"""E111MinimalSteiner3Pin — V3+Steiner descent + CD polish (E122).

Drop-in variant of submissions/_archive/e111_minimal_ovl10_720s/ that
swaps the per-net-trace L-route congestion (E111) for the 3-pin
Steiner T-route variant (E122). Everything else identical (SDF init,
Adam descent, greedy legalize, CD polish at 600s).

Hypothesis (per docs/research/2026-05-20_congestion_model_survey.md
§3 IMPROVEMENT 1): closing the canonical-vs-smooth bias on the
25-40% of 3-pin nets from +14% to +8-10% should translate to a
0.5-2% lift on ibm17. Target: ibm17 + V3+CD600s ≤ 1.18 (baseline 1.20).
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
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E122_3pin_steiner" / "code",
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

from smooth_global_placer_v3_steiner import SmoothGlobalPlacerV3Steiner  # noqa: E402


class E111MinimalSteiner3PinPlacer:
    """V3+Steiner descent + CD polish.

    Pipeline:
      1. SmoothGlobalPlacerV3Steiner (Adam on E122 per-net-trace+Steiner proxy)
      2. greedy_macro_legalize — zero overlaps
      3. CD polish — 600s default

    Same hyperparameters as e111_minimal_ovl10_720s, only swapping
    PerNetTraceCongestion → PerNetTraceCongestionSteiner3Pin.
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 600.0,
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
        beta_steiner: float = 6.0,
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
        self.trace_kwargs = {"beta_steiner": float(beta_steiner)}

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E111MinimalSteiner3Pin ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: V3+Steiner descent + legalize
        placer = SmoothGlobalPlacerV3Steiner(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=self.init,
            rng_seed=self.rng_seed,
            verbose=False,
            trace_kwargs=self.trace_kwargs,
        )
        pos = placer.place(benchmark)
        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  V3Steiner+legalize: proxy={descent_proxy:.5f} ovl={descent_ovl} "
            f"wall={descent_wall:.0f}s"
        )

        if descent_ovl > 0:
            self._log(f"  Recovery: project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)
            descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if descent_ovl > 0:
                raise RuntimeError(
                    f"V3Steiner produced {descent_ovl} overlaps despite legalize+project"
                )

        # Phase 2: CD polish (cap at deadline)
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
            raise RuntimeError(
                f"E111MinimalSteiner3Pin produced {final_ovl} overlaps"
            )
        return final


# Alias for evaluate harness
class Placer(E111MinimalSteiner3PinPlacer):
    pass
