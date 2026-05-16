import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Lévy variant with rng_seed=300 (different ε draws)."""
from submissions.cd_lns_sa_cascade_levy.placer import CDLNSSACascadeLevyPlacer

class CDLNSSACascadeLevyS300Placer(CDLNSSACascadeLevyPlacer):
    def __init__(self):
        super().__init__(rng_seed=300)
