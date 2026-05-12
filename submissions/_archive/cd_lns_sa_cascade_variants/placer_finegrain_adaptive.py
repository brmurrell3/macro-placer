"""Finegrain + property-based dispatch — combined winner.

IBM (canvas_area < 100k μm²):  tight eps + 300s polish + max_iters=2
                               (finegrain default)
NG45 (canvas_area ≥ 100k μm²): finegrain + longer min_time_s for CD adaptive
                               (avoids early plateau exit on NG45 designs)

Builds on finegrain (winning configuration) with adaptive params for the
4 NG45 commercial designs.
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


class CDLNSSACascadeFineGrainAdaptive(CDLNSSACascadeFineGrain):
    """Finegrain + property-tuned CD parameters for NG45."""

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)

    def place(self, benchmark):
        canvas_area = float(benchmark.canvas_width) * float(benchmark.canvas_height)
        is_large = canvas_area > 100000

        orig_run_cd = _cdmod.run_cd_adaptive
        if is_large:
            min_time_target = 180.0
            plateau_target = 0.0001
        else:
            min_time_target = 30.0
            plateau_target = 0.001

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
