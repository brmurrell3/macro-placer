"""Finegrain b=2700 + property-based dispatch — SAFE-WALL SUBMISSION TARGET.

Combines:
- placer_finegrain.py (tight eps {0.05, 0.1, 0.3, 0.7} + 300s polish + max_iters=2)
- budget_seconds=2700s (45-min target, 15-min margin to 60-min cap)
- Property-based NG45 tuning (min_time_s=180, plateau_threshold=1e-4 for large canvas)

Expected: ~1.128 IBM / ~0.694 NG45. Slightly worse proxy than b=3000 variant
but DURABLE against partcl box being ~10% slower than our OCI cloud.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))

import macro_place.cd_core as _cdmod
from submissions.cd_lns_sa_cascade.placer_finegrain import CDLNSSACascadeFineGrain


class CDLNSSACascadeFineGrainSafeAdaptive(CDLNSSACascadeFineGrain):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 2700.0)
        super().__init__(**kwargs)

    def place(self, benchmark):
        canvas_area = float(benchmark.canvas_width) * float(benchmark.canvas_height)
        is_large = canvas_area > 100000
        orig_run_cd = _cdmod.run_cd_adaptive
        min_time_target = 180.0 if is_large else 30.0
        plateau_target = 0.0001 if is_large else 0.001

        def patched_run_cd(*args, **kw):
            if kw.get("min_time_s", 0) < min_time_target:
                kw["min_time_s"] = min_time_target
            if kw.get("plateau_threshold", 1.0) > plateau_target:
                kw["plateau_threshold"] = plateau_target
            return orig_run_cd(*args, **kw)

        _cdmod.run_cd_adaptive = patched_run_cd
        try:
            return super().place(benchmark)
        finally:
            _cdmod.run_cd_adaptive = orig_run_cd
