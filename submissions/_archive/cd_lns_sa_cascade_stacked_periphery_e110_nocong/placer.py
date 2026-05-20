"""Variant A (Lane-4) with sweep-winner cfg: include_congestion=False.

Sweep --fast Δ_avg=-3.92% (4/4 wins). Hypothesis: dropping smooth
congestion from descent removes the noisy ABU-5% RUDY approximation
that's known to diverge 3-4× from canonical on hard benches; CD polish
fixes congestion combinatorially.

Also uses lr=3e-3, num_steps=500, overlap_lambda_end=30 (sweep cfg).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Patch SmoothGlobalPlacer to plumb include_congestion through.
_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer  # noqa

_BASE = (
    _ROOT / "submissions"
    / "cd_lns_sa_cascade_stacked_periphery_e110" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("e110_lane4_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_BasePlacer = _mod.CDLNSSACascadeStackedPeripheryE110Placer


class CDLNSSACascadeStackedPeripheryE110NoCongPlacer(_BasePlacer):
    """Lane-4 with include_congestion=False, lr=3e-3, ovl=30."""

    def __init__(self, **kwargs):
        kwargs.setdefault("e110_lr_frac", 0.003)
        kwargs.setdefault("e110_overlap_lambda_end", 30.0)
        super().__init__(**kwargs)

    def _run_e110_lane(self, benchmark, plc, budget_s):
        # Override to construct SmoothGlobalPlacer with include_congestion=False.
        import time
        import torch
        from macro_place.cd_core import run_cd_adaptive
        from macro_place.incremental_evaluator import IncrementalProxyEvaluator
        from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

        t0 = time.time()
        try:
            placer = SmoothGlobalPlacer(
                num_steps=self.e110_num_steps,
                lr_frac=self.e110_lr_frac,
                gamma_start_frac=self.e110_gamma_start_frac,
                gamma_end_frac=self.e110_gamma_end_frac,
                overlap_lambda_end=self.e110_overlap_lambda_end,
                overlap_ramp_pct=self.e110_overlap_ramp_pct,
                include_congestion=False,
                legalize_step_frac=self.e110_legalize_step_frac,
                legalize_radius_steps=self.e110_legalize_radius_steps,
                init=self.e110_init,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos = placer.place(benchmark)
        except Exception as exc:
            self._log(f"  E110 (nocong) descent FAILED: {exc}")
            return None, None, None

        descent_wall = time.time() - t0
        cd_budget = max(60.0, budget_s - descent_wall - 30.0)
        cd_budget = min(cd_budget, self.e110_cd_polish_s)
        self._log(f"  E110(nocong) descent: wall={descent_wall:.0f}s; "
                  f"CD polish budget={cd_budget:.0f}s")
        try:
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros)
                       if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(evaluator, benchmark, plc, movable,
                            min_time_s=cd_budget, hard_cap_s=cd_budget,
                            patience=3, plateau_threshold=0.001)
            pos = evaluator.placement.detach().clone().to(torch.float32)
        except Exception as exc:
            self._log(f"  E110(nocong)+CD polish FAILED: {exc}")
            return None, None, None

        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        return pos, proxy, ovl
