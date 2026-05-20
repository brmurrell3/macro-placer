"""Variant A + E111 V3 (per-net trace congestion) as Lane-5.

Extends `cd_lns_sa_cascade_stacked_periphery_e110` by adding a SECOND
smooth-gradient lane that uses the canonical-matching per-net trace
congestion (E111). Pipeline:

  1. stacked_periphery (Option C base) — 2100s budget
  2. E110 lane (original smooth proxy) — 450s budget
  3. E111 V3 lane (per-net trace smooth proxy) — 450s budget
  4. Plateau pick: min of {stacked_periphery, E110+CD, V3+CD}

V3 standalone wins on benches where canonical-congestion alignment
matters (ibm14 -4.4%, ibm17 -5.9%) and loses where cascade saturates
(NG45 +5%). Plateau-pick selects the right one per bench.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

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

# Base: Variant A (already adds E110 lane to stacked_periphery).
_BASE = _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery_e110" / "placer.py"
_spec = importlib.util.spec_from_file_location("variant_a_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CDLNSSACascadeStackedPeripheryE110Placer = _mod.CDLNSSACascadeStackedPeripheryE110Placer

# E111 V3 smooth placer
_E111_CODE = _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code"
if str(_E111_CODE) not in sys.path:
    sys.path.insert(0, str(_E111_CODE))
from smooth_global_placer_v3 import SmoothGlobalPlacerV3  # noqa: E402


class CDLNSSACascadeStackedPeripheryE110V3LanePlacer(CDLNSSACascadeStackedPeripheryE110Placer):
    """Variant A + E111 V3 as additional plateau-pick candidate."""

    def __init__(
        self,
        # Override default budgets to make room for V3 lane
        e110_lane_budget_s: float = 450.0,
        e110_cd_polish_s: float = 180.0,
        v3_lane_budget_s: float = 450.0,
        v3_cd_polish_s: float = 180.0,
        # V3 hyperparameters (start with E111 defaults)
        v3_num_steps: int = 500,
        v3_lr_frac: float = 0.005,
        v3_overlap_lambda_end: float = 50.0,
        v3_init: str = "sdf",
        **kwargs,
    ):
        # Set defaults that account for V3 lane
        kwargs.setdefault("e110_lane_budget_s", e110_lane_budget_s)
        kwargs.setdefault("e110_cd_polish_s", e110_cd_polish_s)
        super().__init__(**kwargs)
        self.v3_lane_budget_s = v3_lane_budget_s
        self.v3_cd_polish_s = v3_cd_polish_s
        self.v3_num_steps = v3_num_steps
        self.v3_lr_frac = v3_lr_frac
        self.v3_overlap_lambda_end = v3_overlap_lambda_end
        self.v3_init = v3_init

    def _run_v3_lane(
        self,
        benchmark: Benchmark,
        plc,
        budget_s: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[int]]:
        """Run E111 V3 descent + CD polish under budget_s."""
        t0 = time.time()
        try:
            placer = SmoothGlobalPlacerV3(
                num_steps=self.v3_num_steps,
                lr_frac=self.v3_lr_frac,
                overlap_lambda_end=self.v3_overlap_lambda_end,
                init=self.v3_init,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos = placer.place(benchmark)
        except Exception as exc:
            self._log(f"  V3 descent FAILED: {exc}")
            return None, None, None

        descent_wall = time.time() - t0
        cd_budget = max(60.0, budget_s - descent_wall - 30.0)
        cd_budget = min(cd_budget, self.v3_cd_polish_s)
        self._log(
            f"  V3 descent: wall={descent_wall:.0f}s; "
            f"CD polish budget={cd_budget:.0f}s"
        )

        try:
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [
                i for i in range(benchmark.num_macros)
                if not bool(benchmark.macro_fixed[i])
            ]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd_budget, hard_cap_s=cd_budget,
                patience=3, plateau_threshold=0.001,
            )
            pos = evaluator.placement.detach().clone().to(torch.float32)
        except Exception as exc:
            self._log(f"  V3+CD polish FAILED: {exc}")
            return None, None, None

        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        return pos, proxy, ovl

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = self._log
        log(f"=== CDLNSSACascadeStackedPeripheryE110V3LanePlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        # Run base (which gives us Variant A: stacked_periphery + E110 lane plateau-pick)
        base_pos = super().place(benchmark)
        base_t = time.time() - t0

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        base_proxy = float(compute_proxy_cost(base_pos, benchmark, plc)["proxy_cost"])
        base_ovl = compute_overlap_metrics(base_pos, benchmark)["overlap_count"]
        log(f"  Variant A base: proxy={base_proxy:.5f} ovl={base_ovl} wall={base_t:.0f}s")

        # Run V3 lane on remaining time
        if deadline is not None:
            remaining = deadline - time.time()
        else:
            remaining = self.v3_lane_budget_s
        if remaining < 90.0:
            log(f"  V3 lane SKIPPED (only {remaining:.0f}s left)")
            return base_pos

        v3_budget = min(self.v3_lane_budget_s, remaining - 30.0)
        log(f"  V3 lane: budget={v3_budget:.0f}s")
        v3_pos, v3_proxy, v3_ovl = self._run_v3_lane(
            benchmark, plc, budget_s=v3_budget,
        )

        if v3_pos is None or v3_ovl is None:
            log(f"  V3 lane returned no valid output; keeping base")
            return base_pos
        log(f"  V3+CD: proxy={v3_proxy:.5f} ovl={v3_ovl} "
            f"(total wall={time.time() - t0:.0f}s)")

        # Plateau pick: strictly improve + zero overlaps
        if v3_ovl == 0 and v3_proxy < base_proxy:
            log(f"  ACCEPT V3 lane: {base_proxy:.5f} → {v3_proxy:.5f} "
                f"({(v3_proxy - base_proxy) / base_proxy * 100:+.2f}%)")
            return v3_pos
        log(f"  REJECT V3 lane (ovl={v3_ovl}, Δ={v3_proxy - base_proxy:+.5f}); keeping base")
        return base_pos
