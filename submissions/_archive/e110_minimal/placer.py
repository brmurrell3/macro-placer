"""E110Minimal — gradient placer + CD polish only, no cascade.

Goal: test if E110 + CD polish alone (no E25/E41/cascade/portfolio) gets
comparable quality to Option C at a fraction of the wall.

If quality within 2% of Option C, this is the path to Carrotato-class
throughput (~4 min/bench vs current ~55 min/bench). Lets us run
multi-seed ensembles in the same budget.

Architecture:
  1. SmoothGlobalPlacer.descend() — Adam descent on E95 DiffProxy
  2. greedy_macro_legalize — zero-overlap projection
  3. run_cd_adaptive — CD polish to canonical proxy plateau

Total wall ~3-5 min per bench depending on size.

Hyperparams from sweep top: ovl_lambda_end=10, lr=5e-3, steps=500.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer


class E110MinimalPlacer:
    """SmoothGlobalPlacer + CD polish; no cascade. Fast lane experiment."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 3300.0,
        # E110 (sweep top cfg)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        include_congestion: bool = True,
        legalize_step_frac: float = 0.005,
        legalize_radius_steps: int = 200,
        init: str = "sdf",
        # CD polish: bulk of remaining time
        cd_min_polish_s: float = 600.0,
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
        self.include_congestion = include_congestion
        self.legalize_step_frac = legalize_step_frac
        self.legalize_radius_steps = legalize_radius_steps
        self.init = init
        self.cd_min_polish_s = cd_min_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E110MinimalPlacer ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: E110 descent + legalize
        placer = SmoothGlobalPlacer(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            include_congestion=self.include_congestion,
            legalize_step_frac=self.legalize_step_frac,
            legalize_radius_steps=self.legalize_radius_steps,
            init=self.init,
            rng_seed=self.rng_seed,
            verbose=False,
        )
        pos = placer.place(benchmark)
        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(f"  E110+legalize: proxy={descent_proxy:.5f} ovl={descent_ovl} "
                  f"wall={descent_wall:.0f}s")

        if descent_ovl > 0:
            raise RuntimeError(
                f"E110 produced {descent_ovl} overlaps despite legalize"
            )

        # Phase 2: CD polish to plateau
        if deadline is not None:
            remaining = deadline - time.time() - 30.0
            cd_budget = max(self.cd_min_polish_s, min(remaining, self.budget_seconds - descent_wall - 60.0))
        else:
            cd_budget = self.cd_min_polish_s
        self._log(f"  CD polish: budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5,  # plateau check after 50% of budget
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
                  f"total_wall={time.time()-t0:.0f}s")

        if final_ovl > 0:
            raise RuntimeError(
                f"E110Minimal produced {final_ovl} overlaps in final placement"
            )
        return final
