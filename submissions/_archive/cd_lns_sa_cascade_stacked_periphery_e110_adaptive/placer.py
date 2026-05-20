"""Variant A + adaptive num_steps (scales with bench size).

Hypothesis: ibm17 (760 macros) needs more descent steps than ibm01 (246).
With adaptive_num_steps=True and steps_per_macro=2.0, ibm01 gets 492
steps and ibm17 gets 1520 steps. Targets the ibm17 +2.95% LOSS from
the default Variant A.

Also uses overlap_lambda_end=10 (sweep top cfg).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer  # noqa: E402

_BASE = (
    _ROOT / "submissions"
    / "cd_lns_sa_cascade_stacked_periphery_e110" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("e110_lane4_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_BasePlacer = _mod.CDLNSSACascadeStackedPeripheryE110Placer


class CDLNSSACascadeStackedPeripheryE110AdaptivePlacer(_BasePlacer):
    """Lane-4 with adaptive num_steps + overlap_lambda_end=10."""

    def __init__(self, **kwargs):
        kwargs.setdefault("e110_overlap_lambda_end", 10.0)
        # Note: SmoothGlobalPlacer has adaptive_num_steps but base placer
        # constructs it with fixed args. We override _run_e110_lane.
        super().__init__(**kwargs)

    def _run_e110_lane(self, benchmark, plc, budget_s):
        from macro_place.cd_core import run_cd_adaptive
        from macro_place.incremental_evaluator import IncrementalProxyEvaluator
        from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

        t0 = time.time()
        try:
            placer = SmoothGlobalPlacer(
                num_steps=self.e110_num_steps,  # used as min; auto-scales up
                lr_frac=self.e110_lr_frac,
                gamma_start_frac=self.e110_gamma_start_frac,
                gamma_end_frac=self.e110_gamma_end_frac,
                overlap_lambda_end=self.e110_overlap_lambda_end,
                overlap_ramp_pct=self.e110_overlap_ramp_pct,
                legalize_step_frac=self.e110_legalize_step_frac,
                legalize_radius_steps=self.e110_legalize_radius_steps,
                init=self.e110_init,
                rng_seed=self.rng_seed,
                adaptive_num_steps=True,
                steps_per_macro=2.0,
                verbose=False,
            )
            pos = placer.place(benchmark)
        except Exception as exc:
            self._log(f"  E110(adaptive) descent FAILED: {exc}")
            return None, None, None

        descent_wall = time.time() - t0
        cd_budget = max(60.0, budget_s - descent_wall - 30.0)
        cd_budget = min(cd_budget, self.e110_cd_polish_s)
        self._log(f"  E110(adaptive) descent: wall={descent_wall:.0f}s; "
                  f"CD polish budget={cd_budget:.0f}s")

        try:
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros)
                       if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd_budget, hard_cap_s=cd_budget,
                patience=3, plateau_threshold=0.001,
            )
            pos = evaluator.placement.detach().clone().to(torch.float32)
        except Exception as exc:
            self._log(f"  E110(adaptive)+CD polish FAILED: {exc}")
            return None, None, None

        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        return pos, proxy, ovl
