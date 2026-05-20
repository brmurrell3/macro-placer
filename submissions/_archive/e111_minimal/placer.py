"""E111Minimal — V3 (per-net trace) descent + light CD polish only.

Target: 2-4 min/bench for fast iteration. No cascade pipeline.

Pipeline:
  1. SmoothGlobalPlacerV3 (Adam on per-net trace proxy) — ~30-60s
  2. greedy_macro_legalize — zero overlaps
  3. CD polish — 60-180s

Hypothesis: V3's canonical-aligned basin polishes well WITHOUT the
50-min cascade saturation. Where V3 wins on hard benches (ibm14, ibm17),
this should match or beat the full cascade pipeline at 1/20 the wall.
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

_E111_CODE = _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code"
if str(_E111_CODE) not in sys.path:
    sys.path.insert(0, str(_E111_CODE))
from smooth_global_placer_v3 import SmoothGlobalPlacerV3


class E111MinimalPlacer:
    """V3 descent + light CD polish; no cascade. Fast lane for iteration."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 240.0,  # 4 min default
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 50.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 120.0,
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

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E111MinimalPlacer ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: V3 descent + legalize
        placer = SmoothGlobalPlacerV3(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=self.init,
            rng_seed=self.rng_seed,
            verbose=False,
        )
        pos = placer.place(benchmark)
        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(f"  V3+legalize: proxy={descent_proxy:.5f} ovl={descent_ovl} "
                  f"wall={descent_wall:.0f}s")

        if descent_ovl > 0:
            raise RuntimeError(f"V3 produced {descent_ovl} overlaps despite legalize")

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
        self._log(f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
                  f"total_wall={time.time()-t0:.0f}s")

        if final_ovl > 0:
            raise RuntimeError(f"E111Minimal produced {final_ovl} overlaps")
        return final
