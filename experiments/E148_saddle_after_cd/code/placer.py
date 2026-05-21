"""E148 — Cascade saddle escape applied to CD-polished basin.

Different from E128/E138: saddle runs AFTER full CD polish, not after
V4+Gaussian Adam descent. CD takes Adam descent's basin (~0.96-1.28) down
to canonical local optimum (~0.82-1.20); saddle then escapes that local
optimum and a second CD polish polishes the result.

Pipeline:
  1. V4+Gaussian descent + legalize + project_overlaps
  2. CD polish 1 — 700s (full polish to plateau)
  3. Cascade saddle escape on CD-polished position (bounded eigsh,
     `maxiter=50, tol=1e-2`, `saddle_budget=120s, max_iters=2`)
  4. CD polish 2 — 200s (polish saddle output)

Total budget 1500s/bench. Reuses E138 `bounded_saddle.py` unchanged.
DO NOT mutate shared evaluator / placer code.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E138_bounded_saddle" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian
from bounded_saddle import bounded_cascading_saddle_escape


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E148SaddleAfterCdPlacer:
    """V4+Gaussian -> CD polish 1 -> bounded cascade saddle -> CD polish 2."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,  # 25 min/bench
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd1_polish_s: float = 700.0,  # full CD polish to plateau
        cd2_polish_s: float = 200.0,  # short polish after saddle
        saddle_budget_s: float = 120.0,
        saddle_max_iters: int = 2,
        eigsh_maxiter: int = 50,
        eigsh_tol: float = 1e-2,
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd1_polish_s = cd1_polish_s
        self.cd2_polish_s = cd2_polish_s
        self.saddle_budget_s = saddle_budget_s
        self.saddle_max_iters = saddle_max_iters
        self.eigsh_maxiter = eigsh_maxiter
        self.eigsh_tol = eigsh_tol
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s):
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E148 ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # Phase 1: V4+Gaussian descent + legalize
        t_desc = time.time()
        descender = SmoothGlobalPlacerV4Gaussian(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=self.init,
            device=device,
            rng_seed=self.rng_seed,
            verbose=False,
        )
        pos = descender.place(benchmark)
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        if ovl > 0:
            pos, _ = project_overlaps(pos, benchmark)
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_wall = time.time() - t_desc
        self._log(f"  descent: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        # Phase 2: CD polish 1 (full polish to plateau)
        t_cd1 = time.time()
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd1_budget = max(60.0, min(remaining * 0.55, self.cd1_polish_s))
        else:
            cd1_budget = self.cd1_polish_s
        self._log(f"  CD1 polish budget={cd1_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable_idx = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable_idx,
            min_time_s=cd1_budget * 0.5,
            hard_cap_s=cd1_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        pos = evaluator.placement.detach().clone().to(torch.float32)
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        cd1_wall = time.time() - t_cd1
        self._log(f"  CD1: proxy={cd1_proxy:.5f} (delta={cd1_proxy - descent_proxy:+.5f}) "
                  f"wall={cd1_wall:.0f}s")

        # Phase 3: bounded cascade saddle escape on CD-polished position
        t_saddle = time.time()
        cd1_pos = pos.detach().clone()  # baseline for accept/reject
        saddle_accepted = False
        try:
            saddle_pos, saddle_info = bounded_cascading_saddle_escape(
                pos, benchmark, plc,
                max_iters=self.saddle_max_iters,
                total_budget_s=self.saddle_budget_s,
                eigsh_maxiter=self.eigsh_maxiter,
                eigsh_tol=self.eigsh_tol,
                log=lambda s: self._log(f"  [saddle] {s}"),
            )
            saddle_ovl = compute_overlap_metrics(saddle_pos, benchmark)["overlap_count"]
            if saddle_ovl == 0:
                saddle_proxy = float(compute_proxy_cost(saddle_pos, benchmark, plc)["proxy_cost"])
                if saddle_proxy <= cd1_proxy:
                    pos = saddle_pos
                    saddle_accepted = True
                    self._log(f"  saddle ACCEPT: {cd1_proxy:.5f} -> {saddle_proxy:.5f}")
                else:
                    self._log(f"  saddle REJECT: {saddle_proxy:.5f} > {cd1_proxy:.5f}")
            else:
                self._log(f"  saddle output has {saddle_ovl} overlaps; rejecting")
        except Exception as exc:
            self._log(f"  saddle EXCEPTION: {exc}; skipping")
        saddle_wall = time.time() - t_saddle
        self._log(f"  saddle wall={saddle_wall:.0f}s")

        # Phase 4: CD polish 2 (polish saddle output, only meaningful if accepted)
        t_cd2 = time.time()
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd2_budget = max(30.0, min(remaining, self.cd2_polish_s))
        else:
            cd2_budget = self.cd2_polish_s
        self._log(f"  CD2 polish budget={cd2_budget:.0f}s")

        evaluator2 = IncrementalProxyEvaluator(benchmark, plc, pos)
        run_cd_adaptive(
            evaluator2, benchmark, plc, movable_idx,
            min_time_s=cd2_budget * 0.5,
            hard_cap_s=cd2_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        cd2_pos = evaluator2.placement.detach().clone().to(torch.float32)
        cd2_proxy = float(compute_proxy_cost(cd2_pos, benchmark, plc)["proxy_cost"])
        cd2_wall = time.time() - t_cd2
        self._log(f"  CD2: proxy={cd2_proxy:.5f} wall={cd2_wall:.0f}s")

        # Safety: never ship something worse than CD1
        cd1_baseline_proxy = float(compute_proxy_cost(cd1_pos, benchmark, plc)["proxy_cost"])
        if cd2_proxy <= cd1_baseline_proxy:
            final = cd2_pos
            final_proxy = cd2_proxy
            self._log(f"  ship CD2: {cd2_proxy:.5f} (beats CD1 {cd1_baseline_proxy:.5f})")
        else:
            final = cd1_pos
            final_proxy = cd1_baseline_proxy
            self._log(f"  ship CD1: {cd1_baseline_proxy:.5f} (CD2 {cd2_proxy:.5f} regressed)")

        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        total_wall = time.time() - t0
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"saddle_accepted={saddle_accepted} total_wall={total_wall:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final
