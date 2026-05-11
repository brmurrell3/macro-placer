"""NG45-tuned cascade: longer per-iter CD polish (avoid early plateau exit).

NG45 designs are larger than IBM (ariane133 ~270 macros, complex netlist).
The default min_time_s=30 in cascading_saddle_escape causes early plateau
detection on these benches, leaving cascade budget unused. Bump polish_budget
+ min_time_s for NG45.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Need to also patch cascading_saddle import
_E84 = _ROOT / 'experiments' / 'E84_cascading_saddle' / 'code'
sys.path.insert(0, str(_E84))

# Monkey-patch cascading_saddle's run_cd_adaptive call params
import cascading_saddle as _csmod
_orig = _csmod.cascading_saddle_escape

def cascading_saddle_escape_ng45(*args, **kwargs):
    # Override polish_budget to 300s (was 180s)
    kwargs.setdefault('polish_budget', 300.0)
    # Other kwargs untouched
    return _orig(*args, **kwargs)

# Inject the wrapper
_csmod.cascading_saddle_escape = cascading_saddle_escape_ng45

# Also monkey-patch the run_cd_adaptive call inside to use min_time_s=180
import macro_place.cd_core as _cdmod
_orig_cd = _cdmod.run_cd_adaptive
def run_cd_adaptive_ng45(*args, **kwargs):
    if kwargs.get('min_time_s', 0) < 100:
        kwargs['min_time_s'] = 180.0
    if kwargs.get('plateau_threshold', 1.0) > 0.0001:
        kwargs['plateau_threshold'] = 0.0001  # tighter — fewer false plateaus
    return _orig_cd(*args, **kwargs)
_cdmod.run_cd_adaptive = run_cd_adaptive_ng45

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeNG45Placer(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault('budget_seconds', 3300.0)
        super().__init__(**kwargs)
