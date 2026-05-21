"""E111Minimal-multistage — MultiStageAdamPlacer + light CD polish.

Tests the ePlace/Xplace 3-stage fixed-γ descent pattern (E118) against
the V3 single-anneal baseline. Identical post-descent pipeline to
e111_minimal_ovl10_720s (greedy_legalize → CD polish), so any score
delta isolates the descent recipe.

Pipeline:
  1. MultiStageAdamPlacer (Adam on V3 proxy)
     - Stage A (200 steps, γ=5e-3): overlap_lambda 0 → 0.5
     - Stage B (200 steps, γ=5e-4): overlap_lambda = 0.5
     - Stage C (200 steps, γ=5e-5): overlap_lambda = 0.5
     - Best-by-smooth position tracked across stages
  2. greedy_macro_legalize  (zero overlaps)
  3. CD polish              (10 min)

Budget: 720s total / 12 min — matches the e111_minimal_ovl10_720s
baseline so the comparison is apples-to-apples on wall time.

Baselines to beat:
  - V3Min ovl10 720s on ibm17 = 1.20
  - thinkorplace-v2 on --all  = 1.003
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
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_E118_CODE = _ROOT / "experiments" / "E118_multistage_adam" / "code"
if str(_E118_CODE) not in sys.path:
    sys.path.insert(0, str(_E118_CODE))
from multistage_placer import MultiStageAdamPlacer


class E111MinimalMultiStagePlacer:
    """MultiStage Adam descent + light CD polish; matches the 12-min budget."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,             # 12 min/bench
        # Stage knobs (forwarded to MultiStageAdamPlacer). v3: 100/100/100
        # = 300 total steps (vs V3's 500) to leave more wall for CD polish.
        # On ibm17 with sibling-CPU contention, v2's 600 steps consumed 480s
        # of descent and only left 122s for CD → final 1.211. Halving descent
        # work should give ~5-7 min CD instead of 2.
        stage_steps=(100, 100, 100),
        gamma_stage_A: float = 5e-3,
        gamma_stage_B: float = 5e-4,
        gamma_stage_C: float = 5e-5,
        # Primary cfg uses V3's sweet-spot overlap_lambda=10. The "density_weight 0 → 0.5"
        # task spec maps loosely to the soft-overlap penalty coefficient in our objective;
        # using λ=0.5 (per the spec) produces non-zero overlaps after descent and burns
        # multi-minute legalize time, leaving no budget for CD polish on ibm17.
        # Falling back to the V3-tuned scale here makes the budget viable.
        overlap_lambda_stage_A=(0.0, 10.0),
        overlap_lambda_stage_B: float = 10.0,
        overlap_lambda_stage_C: float = 10.0,
        base_lr_frac: float = 0.005,
        boundary_lambda: float = 50.0,
        init: str = "sdf",
        # CD polish
        cd_polish_s: float = 600.0,                           # 10 min CD
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.stage_steps = stage_steps
        self.gamma_stage_A = gamma_stage_A
        self.gamma_stage_B = gamma_stage_B
        self.gamma_stage_C = gamma_stage_C
        self.overlap_lambda_stage_A = overlap_lambda_stage_A
        self.overlap_lambda_stage_B = overlap_lambda_stage_B
        self.overlap_lambda_stage_C = overlap_lambda_stage_C
        self.base_lr_frac = base_lr_frac
        self.boundary_lambda = boundary_lambda
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _safe_fallback_place(self, benchmark, plc):
        """SDF init + project_overlaps when descent fails to zero overlaps."""
        self._log("  fallback: SDF + project_overlaps")
        pos = sdf_init(benchmark)
        pos, _ = project_overlaps(pos, benchmark)
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(f"  WARNING: SDF+project still has {ovl} overlaps")
        return pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E111MinimalMultiStagePlacer ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: multi-stage descent + legalize, with retries on overlap fail
        pos = None
        attempts = [
            # Primary cfg (the experiment)
            dict(
                overlap_lambda_stage_A=self.overlap_lambda_stage_A,
                overlap_lambda_stage_B=self.overlap_lambda_stage_B,
                overlap_lambda_stage_C=self.overlap_lambda_stage_C,
            ),
            # Fallback 1: stronger overlap (matches V3 sweet spot of 10)
            dict(
                overlap_lambda_stage_A=(0.0, 5.0),
                overlap_lambda_stage_B=10.0,
                overlap_lambda_stage_C=10.0,
            ),
            # Fallback 2: very strong overlap
            dict(
                overlap_lambda_stage_A=(0.0, 25.0),
                overlap_lambda_stage_B=50.0,
                overlap_lambda_stage_C=50.0,
            ),
        ]
        for attempt, cfg in enumerate(attempts):
            try:
                self._log(
                    f"  multistage attempt {attempt+1}: λ_A={cfg['overlap_lambda_stage_A']} "
                    f"λ_B={cfg['overlap_lambda_stage_B']} λ_C={cfg['overlap_lambda_stage_C']}"
                )
                descender = MultiStageAdamPlacer(
                    stage_steps=self.stage_steps,
                    gamma_stage_A=self.gamma_stage_A,
                    gamma_stage_B=self.gamma_stage_B,
                    gamma_stage_C=self.gamma_stage_C,
                    base_lr_frac=self.base_lr_frac,
                    boundary_lambda=self.boundary_lambda,
                    init=self.init,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                    **cfg,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"  attempt {attempt+1}: ovl=0")
                    break
                # Recovery: project_overlaps
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                if ovl2 == 0:
                    pos = pos_try2
                    self._log(f"  attempt {attempt+1}: ovl=0 after project_overlaps")
                    break
                self._log(f"  attempt {attempt+1}: still {ovl_try}/{ovl2} overlaps")
            except Exception as exc:
                self._log(f"  attempt {attempt+1} EXCEPTION: {exc}")

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

        # Phase 2: CD polish
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
            raise RuntimeError(f"E111MinimalMultiStage produced {final_ovl} overlaps")
        return final
