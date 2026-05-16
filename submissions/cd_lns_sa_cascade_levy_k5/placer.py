import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Lévy variant with K_eps=5 (more magnitude candidates per direction)."""
from submissions.cd_lns_sa_cascade_levy.placer import CDLNSSACascadeLevyPlacer

class CDLNSSACascadeLevyK5Placer(CDLNSSACascadeLevyPlacer):
    def __init__(self):
        super().__init__(K_eps=5)
