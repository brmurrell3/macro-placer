"""E140 — v1 cascade saddle pipeline with V4 FastDiffProxy + Gaussian basin.

Architecture (descent -> CD1 -> bounded saddle -> CD2):

  1. V4 + Gaussian descent (`SmoothGlobalPlacerV4Gaussian.place()`) — fast
     GPU basin (3-16x faster than V3); produces a legal placement.
  2. CD polish 1 (~300 s) — settle into the local minimum so the saddle
     finds the genuine soft mode at that minimum.
  3. Bounded cascading saddle escape (E138 `bounded_cascading_saddle_escape`,
     `eigsh_maxiter=50, tol=1e-2`). Iterates ±epsilon along the smallest
     eigenvector of the smooth proxy Hessian; CD-polishes each candidate
     with an inner 120 s budget. Bounded eigsh sacrifices eigenvector
     precision but converges 6-8x faster on EPYC than unbounded eigsh.
  4. CD polish 2 (~300 s) — finish on the saddle output (or basin if
     saddle was rejected / failed).

Total budget 1500 s/bench. Within the partcl 60 min/bench cap.

DO NOT mutate E84 / E138 / E127 source — they have downstream callers.
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
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402
from bounded_saddle import bounded_cascading_saddle_escape  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    plateau_threshold: float = 0.001,
):
    """Run CD-adaptive polish under a hard wall budget."""
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=min(30.0, budget_s * 0.5),
        hard_cap_s=budget_s,
        patience=5,
        plateau_threshold=plateau_threshold,
        log_fn=None,
    )
    return evaluator.placement.detach().clone().to(torch.float32)


class E140CascadeFastPlacer:
    """V4+Gaussian basin -> CD1 -> bounded saddle -> CD2."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        # Descent params (mirror v2 thinkorplace-v2)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD params
        cd1_budget_s: float = 300.0,
        cd2_budget_s: float = 300.0,
        cd_plateau_threshold: float = 0.001,
        # Saddle params (E138-style bounded eigsh)
        saddle_budget_s: float = 450.0,
        saddle_max_iters: int = 3,
        saddle_polish_budget: float = 120.0,
        saddle_eps_values=(0.3, 1.0, 3.0),
        eigsh_maxiter: int = 50,
        eigsh_tol: float = 1e-2,
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
        self.cd1_budget_s = cd1_budget_s
        self.cd2_budget_s = cd2_budget_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.saddle_budget_s = saddle_budget_s
        self.saddle_max_iters = saddle_max_iters
        self.saddle_polish_budget = saddle_polish_budget
        self.saddle_eps_values = tuple(saddle_eps_values)
        self.eigsh_maxiter = eigsh_maxiter
        self.eigsh_tol = eigsh_tol
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Phase-wall record (populated by place()).
        self.last_phase_walls: dict = {}
        self.last_phase_proxies: dict = {}
        self.last_saddle_stats: dict = {}

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E140CascadeFastPlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds}s; descent + CD1={self.cd1_budget_s}s "
            f"+ saddle={self.saddle_budget_s}s + CD2={self.cd2_budget_s}s"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")

        # ------- Phase 1: V4 + Gaussian descent + legalize + project_overlaps
        t_descent = time.time()
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(
                    f"  Phase 1 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}"
                )
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"    ok: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"    ok: ovl=0 after project_overlaps")
                    break
                self._log(f"    still {ovl_try} overlaps; retrying")
            except Exception as exc:
                self._log(f"    attempt {attempt+1} EXCEPTION: {exc}")
                continue
            if deadline is not None and time.time() > deadline - 600:
                break
        if pos is None:
            self._log("  Phase 1 fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        descent_wall = time.time() - t_descent
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Phase 1 done: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s"
        )

        # ------- Phase 2: CD polish 1
        if deadline is not None:
            remaining = deadline - time.time()
            # Reserve room for saddle + CD2 + slack.
            reserve = self.saddle_budget_s + self.cd2_budget_s + 60.0
            cd1_budget = max(30.0, min(self.cd1_budget_s, remaining - reserve))
        else:
            cd1_budget = self.cd1_budget_s
        t_cd1 = time.time()
        self._log(f"  Phase 2: CD1 budget={cd1_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd1_budget, self.cd_plateau_threshold)
        cd1_wall = time.time() - t_cd1
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        cd1_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  Phase 2 done: proxy={cd1_proxy:.5f} (Δ={cd1_proxy - descent_proxy:+.5f}) "
            f"ovl={cd1_ovl} wall={cd1_wall:.0f}s"
        )
        if cd1_ovl > 0:
            self._log(f"  WARN: CD1 produced {cd1_ovl} overlaps; project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)

        # ------- Phase 3: bounded cascading saddle escape
        if deadline is not None:
            remaining = deadline - time.time()
            saddle_budget = max(
                60.0, min(self.saddle_budget_s, remaining - self.cd2_budget_s - 30.0)
            )
        else:
            saddle_budget = self.saddle_budget_s
        t_saddle = time.time()
        saddle_pos = pos
        saddle_proxy_pre = cd1_proxy
        saddle_accepted = False
        saddle_info = {}
        if saddle_budget < 60.0:
            self._log(f"  Phase 3 SKIPPED (budget {saddle_budget:.0f}s)")
        else:
            self._log(
                f"  Phase 3: bounded saddle budget={saddle_budget:.0f}s "
                f"max_iters={self.saddle_max_iters} polish={self.saddle_polish_budget}s"
            )
            try:
                cand, saddle_info = bounded_cascading_saddle_escape(
                    pos, benchmark, plc,
                    max_iters=self.saddle_max_iters,
                    eps_values=self.saddle_eps_values,
                    polish_budget=self.saddle_polish_budget,
                    total_budget_s=saddle_budget,
                    eigsh_maxiter=self.eigsh_maxiter,
                    eigsh_tol=self.eigsh_tol,
                    log=lambda s: self._log(f"    [saddle] {s}"),
                )
                cand_ovl = compute_overlap_metrics(cand, benchmark)["overlap_count"]
                if cand_ovl == 0:
                    cand_proxy = float(compute_proxy_cost(cand, benchmark, plc)["proxy_cost"])
                    if cand_proxy < cd1_proxy:
                        pos = cand
                        saddle_pos = cand
                        saddle_accepted = True
                        self._log(
                            f"  Phase 3 ACCEPT: {cd1_proxy:.5f} -> {cand_proxy:.5f}"
                        )
                    else:
                        self._log(
                            f"  Phase 3 REJECT: {cand_proxy:.5f} >= {cd1_proxy:.5f}"
                        )
                else:
                    self._log(f"  Phase 3 REJECT: saddle output has {cand_ovl} ovl")
            except Exception as exc:
                self._log(f"  Phase 3 EXCEPTION: {exc}; skipping")
        saddle_wall = time.time() - t_saddle
        saddle_proxy_after = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  Phase 3 done: proxy={saddle_proxy_after:.5f} wall={saddle_wall:.0f}s")

        # ------- Phase 4: CD polish 2
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd2_budget = max(30.0, min(self.cd2_budget_s, remaining))
        else:
            cd2_budget = self.cd2_budget_s
        t_cd2 = time.time()
        self._log(f"  Phase 4: CD2 budget={cd2_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd2_budget, self.cd_plateau_threshold)
        cd2_wall = time.time() - t_cd2
        final = pos.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        total_wall = time.time() - t0
        self._log(
            f"  Phase 4 done: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"wall={cd2_wall:.0f}s"
        )
        self._log(
            f"  TOTAL: descent={descent_wall:.0f}s CD1={cd1_wall:.0f}s "
            f"saddle={saddle_wall:.0f}s (accepted={saddle_accepted}) "
            f"CD2={cd2_wall:.0f}s total={total_wall:.0f}s "
            f"final_proxy={final_proxy:.5f}"
        )

        # Record stats for harness/inspection.
        self.last_phase_walls = {
            "descent": descent_wall,
            "cd1": cd1_wall,
            "saddle": saddle_wall,
            "cd2": cd2_wall,
            "total": total_wall,
        }
        self.last_phase_proxies = {
            "descent": descent_proxy,
            "cd1": cd1_proxy,
            "saddle_pre": saddle_proxy_pre,
            "saddle_after": saddle_proxy_after,
            "final": final_proxy,
            "saddle_accepted": saddle_accepted,
        }
        self.last_saddle_stats = saddle_info or {}

        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Alias for harness convenience (`Placer` is conventional but we provide both).
Placer = E140CascadeFastPlacer
