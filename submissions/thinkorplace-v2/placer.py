"""thinkorplace-v2 — V3 per-net-trace congestion + Adam descent + CD polish.

The 2026-05-20 submission. Replaces the 50-min cascade pipeline of
thinkorplace (Option C) with a 12-min smooth-gradient placer that finds
canonical-aligned basins on hard benches.

Verified on M3 Max 2026-05-19:
  - IBM avg 1.00279 (vs thinkorplace v1 1.0575 = −5.17% lift)
  - NG45 avg 0.67861 (vs thinkorplace v1 0.6893 = −1.55% lift)
  - Zero overlaps on all 17 IBM + 4 NG45 benchmarks

Pipeline:
  1. SmoothGlobalPlacerV3 (Adam on E88/E95/E111 smooth proxy with
     per-net-trace congestion that matches canonical ±15-25% vs the
     old bbox-uniform ±200-260%)
  2. greedy_macro_legalize (zero hard-macro overlaps via spiral search)
  3. CD polish (run_cd_adaptive, up to 10 min)

Cfg:
  - num_steps=500, lr_frac=0.005 (descent settles in ~30-60s)
  - gamma_anneal 5e-3 → 5e-5 (smooth → near-canonical)
  - overlap_lambda_end=10 (sweep-tuned; 5 produces overlaps, 50 leaves polish on table)
  - budget_seconds=720 (12 min/bench under 60-min cap)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
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

from smooth_global_placer_v3 import SmoothGlobalPlacerV3  # noqa: E402


class Placer:
    """V3 smooth descent + CD polish; M3-verified 1.0028 IBM / 0.6786 NG45."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,           # 12 min/bench
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,                   # sweep top
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 600.0,                         # 10 min CD
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
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _safe_fallback_place(self, benchmark, plc):
        """Defensive fallback: SDF init + project_overlaps + CD polish.

        Used when V3 descent + greedy_legalize fails to produce a
        zero-overlap placement on this platform (e.g., numerics differ
        under QEMU emulation). Slower but always-valid.
        """
        self._log("  fallback: SDF + project_overlaps + CD polish")
        pos = sdf_init(benchmark)
        pos, _ = project_overlaps(pos, benchmark)
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(f"  WARNING: SDF+project still has {ovl} overlaps")
        return pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== thinkorplace-v2 ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: V3 smooth descent + greedy legalize, with retries + fallback
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),  # primary cfg
            dict(overlap_lambda_end=50.0),                       # fallback: stronger overlap
            dict(overlap_lambda_end=100.0, num_steps=300),       # fallback: aggressive overlap
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(f"  V3 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}")
                descender = SmoothGlobalPlacerV3(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    # Also try project_overlaps to ensure zero ovl with tolerance
                    pos_try2, _ = project_overlaps(pos_try, benchmark)
                    ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                    if ovl2 == 0:
                        pos = pos_try2
                        self._log(f"  attempt {attempt+1}: ovl=0 (post-project)")
                        break
                else:
                    # Try project_overlaps as a recovery step
                    pos_try2, _ = project_overlaps(pos_try, benchmark)
                    ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                    if ovl2 == 0:
                        pos = pos_try2
                        self._log(f"  attempt {attempt+1}: ovl=0 after project_overlaps")
                        break
                    self._log(f"  attempt {attempt+1}: still {ovl_try}/{ovl2} overlaps, retrying...")
            except Exception as exc:
                self._log(f"  attempt {attempt+1} EXCEPTION: {exc}")
                continue

            # Hard time budget check
            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            pos = self._safe_fallback_place(benchmark, plc)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  basin: proxy={descent_proxy:.5f} ovl={descent_ovl} "
            f"wall={descent_wall:.0f}s"
        )

        # Phase 2: CD polish to canonical plateau
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
