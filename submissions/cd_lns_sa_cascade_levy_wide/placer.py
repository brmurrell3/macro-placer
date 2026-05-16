import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Lévy variant with eps_scale=2.0 (wider Cauchy → larger typical jumps)."""
from submissions.cd_lns_sa_cascade_levy.placer import CDLNSSACascadeLevyPlacer

class CDLNSSACascadeLevyWidePlacer(CDLNSSACascadeLevyPlacer):
    def __init__(self):
        super().__init__(K_eps=3, eps_scale=2.0)
