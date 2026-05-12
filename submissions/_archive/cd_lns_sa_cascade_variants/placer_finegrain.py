"""Cascade with FINE-GRAIN saddle (tight eps + deep polish per trial).

Hypothesis: existing eps_values=(0.3, 1, 3) jumps too far; tight eps grid
(0.05, 0.1, 0.3, 0.7) catches subtle modes that big jumps overshoot.
Deep polish (300s/trial) ensures each trial converges before measuring.

Fewer trials (4 eps × 2 sign × 2 eigvecs default = 16) but each one is
high-fidelity.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeFineGrain(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        kwargs.setdefault("polish_budget", 300.0)
        kwargs.setdefault("eps_values", (0.05, 0.1, 0.3, 0.7))
        kwargs.setdefault("max_iters", 2)
        super().__init__(**kwargs)
