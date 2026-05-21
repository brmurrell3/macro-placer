"""E128 — V4+Gaussian basin + cascade saddle escape + CD polish.

Quick test: does the E84 cascading_saddle_escape lift our V4+Gaussian
basin? Same as thinkorplace-v2 + extra saddle step between descent and
CD polish.
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
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian
from cascading_saddle import cascading_saddle_escape


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E128CascadeOnV4GaussPlacer:
    """V4+Gaussian descent -> cascade saddle escape -> CD polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1200.0,  # 20 min/bench (extra for saddle)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 480.0,  # 8 min CD
        saddle_budget_s: float = 240.0,  # 4 min saddle
        saddle_max_iters: int = 3,
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
        self.cd_polish_s = cd_polish_s
        self.saddle_budget_s = saddle_budget_s
        self.saddle_max_iters = saddle_max_iters
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s):
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E128 ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # Phase 1: V4+Gaussian descent + legalize
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
        basin_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  basin: proxy={basin_proxy:.5f} wall={time.time()-t0:.0f}s")

        # Phase 2: Cascade saddle escape on basin
        t_saddle = time.time()
        try:
            saddle_pos, saddle_info = cascading_saddle_escape(
                pos, benchmark, plc,
                max_iters=self.saddle_max_iters,
                total_budget_s=self.saddle_budget_s,
                log=lambda s: self._log(f"  [saddle] {s}"),
            )
            saddle_ovl = compute_overlap_metrics(saddle_pos, benchmark)["overlap_count"]
            if saddle_ovl == 0:
                saddle_proxy = float(compute_proxy_cost(saddle_pos, benchmark, plc)["proxy_cost"])
                if saddle_proxy <= basin_proxy:
                    pos = saddle_pos
                    self._log(f"  saddle accept: {basin_proxy:.5f} -> {saddle_proxy:.5f}")
                else:
                    self._log(f"  saddle reject: {saddle_proxy:.5f} > {basin_proxy:.5f}")
            else:
                self._log(f"  saddle output has {saddle_ovl} overlaps; rejecting")
        except Exception as exc:
            self._log(f"  saddle EXCEPTION: {exc}; skipping")
        self._log(f"  saddle wall={time.time()-t_saddle:.0f}s")

        # Phase 3: CD polish
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5,
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} total_wall={time.time()-t0:.0f}s")
        if final_ovl > 0:
            raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final
