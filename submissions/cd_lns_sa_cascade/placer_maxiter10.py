"""Cascade with max_iters=10 (vs default 5) and budget_seconds=3000s.

Hypothesis: more cascade iters per bench may extract additional lift IF
the proxy hasn't truly saturated by iter 5. Adaptive plateau detection
inside cascading_saddle_escape (min_improvement=1e-5) will still cap.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeMaxIter10Placer(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault('budget_seconds', 3000.0)
        kwargs.setdefault('max_iters', 10)
        super().__init__(**kwargs)
