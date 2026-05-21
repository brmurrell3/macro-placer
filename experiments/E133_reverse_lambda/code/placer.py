"""E133 — Reverse-time overlap_lambda schedule on V4Gaussian.

Hypothesis: with Gaussian density the basin is smooth, so HIGH lambda
early (50) cleanly separates macros while they can still re-arrange,
and LOW lambda late (5) lets WL + congestion dominate the final basin
choice. This is the reverse of the default forward-time schedule
(start=0 → end=10/50/100) used by thinkorplace-v2 / E127.

No modifications to shipped V4 / V4Gaussian — `SmoothGlobalPlacerV4`
already accepts `overlap_lambda_start` kwarg (line 58 of
`smooth_global_placer_v4.py`) and `SmoothGlobalPlacerV4Gaussian` forwards
`*args, **kwargs` to the parent. The descend loop already computes:

    overlap_lambda = start + ramp_t * (end - start)

so `start=50, end=5` produces a strictly decreasing schedule.

Retry chain: attempt 1 = reverse-time (the hypothesis), attempts 2/3
fall back to forward-time strong overlap as a safety net so we still
produce a legal placement if reverse-time leaves overlaps.
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
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
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

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E133ReverseLambdaPlacer:
    """V4 + Gaussian density with reverse-time overlap_lambda schedule + CD polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        # REVERSE-TIME schedule (the hypothesis).
        overlap_lambda_start: float = 50.0,
        overlap_lambda_end: float = 5.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 600.0,
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_start = overlap_lambda_start
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E133 reverse_lambda ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        pos = None
        # Attempt 1 = the hypothesis (reverse-time start=50, end=5).
        # Attempts 2/3 = forward-time strong-overlap safety nets so we
        # always produce a legal placement.
        for attempt, cfg in enumerate([
            dict(
                overlap_lambda_start=self.overlap_lambda_start,
                overlap_lambda_end=self.overlap_lambda_end,
            ),
            dict(overlap_lambda_start=0.0, overlap_lambda_end=50.0),
            dict(overlap_lambda_start=0.0, overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_start = cfg["overlap_lambda_start"]
                ovl_end = cfg["overlap_lambda_end"]
                self._log(
                    f"  attempt {attempt+1}: "
                    f"ovl={ovl_start}->{ovl_end} steps={num_steps}"
                )
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_start=ovl_start,
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
                    self._log(f"  attempt {attempt+1}: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"  attempt {attempt+1}: ovl=0 after project_overlaps")
                    break
                self._log(f"  attempt {attempt+1}: still {ovl_try} overlaps, retrying...")
            except Exception as exc:
                self._log(f"  attempt {attempt+1} EXCEPTION: {exc}")
                continue

            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            self._log("  fallback: SDF + project_overlaps + CD polish")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  basin: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s"
        )

        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
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
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time()-t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final
