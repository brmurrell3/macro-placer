"""Property-tuned cascade placer (rule-compliant — no per-bench-NAME dispatch).

Dispatches CD-polish parameters based on BENCHMARK PROPERTIES (n_macros,
canvas size) rather than benchmark identity. NG45-class designs (≥130
macros, large canvas) get longer min_time_s to avoid premature plateau
exit; IBM-class designs use the default tuning.

Rule note: competition prohibits dispatching on bench NAME but explicitly
permits dispatching on bench PROPERTIES (per TODO.md P5 note).
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Need experiments path for cascading_saddle
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer
import macro_place.cd_core as _cdmod


class CDLNSSACascadeAdaptivePlacer(CDLNSSACascadePlacer):
    """Cascade with bench-property-tuned CD parameters."""

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)

    def place(self, benchmark):
        # Detect bench properties. NG45 designs have canvas areas >2M μm²;
        # IBM ICCAD04 designs are <2000 μm² (different physical scales).
        # Threshold canvas_area > 100k clearly separates the two classes.
        canvas_area = float(benchmark.canvas_width) * float(benchmark.canvas_height)
        is_large = canvas_area > 100000  # μm²: NG45-class threshold

        # Monkey-patch run_cd_adaptive with property-tuned params
        orig_run_cd = _cdmod.run_cd_adaptive

        if is_large:
            # NG45-class: longer per-iter CD, tighter plateau threshold.
            min_time_target = 180.0
            plateau_target = 0.0001
        else:
            # IBM-class: default behavior
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
