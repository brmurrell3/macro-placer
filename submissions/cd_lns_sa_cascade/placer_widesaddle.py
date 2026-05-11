"""Cascade WIDER SADDLE search: k=4 eigvecs, 6 eps values, shorter polish/trial.

Hypothesis: current cascade (k=2, 3 eps, 180s polish) only probes 12 trials
per iter with deep polishing. Wider search (k=4 eigvecs × 6 eps × 2 signs =
48 trials × 60s) covers more of the smooth-proxy basin. Same total cascade
budget; just reallocated.

If lift, this is a free win — no algorithmic change.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeWideSaddle(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        kwargs.setdefault("polish_budget", 60.0)
        kwargs.setdefault("eps_values", (0.1, 0.3, 1.0, 3.0, 10.0, 30.0))
        kwargs.setdefault("max_iters", 3)
        super().__init__(**kwargs)
