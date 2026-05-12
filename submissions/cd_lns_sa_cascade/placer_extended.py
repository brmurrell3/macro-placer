"""Extended-budget cascade (b=5400s = 90min/bench) — EPYC ceiling test.

Deliberately overshoots 60-min hard cap to measure what cascade can reach
on EPYC if PATH A2 (CD speedup) succeeded. Tells us whether the gap from
1.137 (wall-safe) → 1.06 (cached uncapped) is closable via CD acceleration.

NOT submittable.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeExtended(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 5400.0)
        super().__init__(**kwargs)
