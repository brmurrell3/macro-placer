"""Micro-grain cascade: ultra-tight eps grid + deep polish.

Hypothesis: finegrain's eps {0.05, 0.1, 0.3, 0.7} works. What if even tighter:
{0.02, 0.05, 0.1, 0.2, 0.5}? Catches even more subtle modes.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeMicroGrain(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        kwargs.setdefault("polish_budget", 300.0)
        kwargs.setdefault("eps_values", (0.02, 0.05, 0.1, 0.2, 0.5))
        kwargs.setdefault("max_iters", 2)
        super().__init__(**kwargs)
