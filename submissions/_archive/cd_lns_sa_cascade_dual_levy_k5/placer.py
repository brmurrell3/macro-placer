import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Dual (canonical + cong-focus) with K_eps=5 each direction."""
from submissions.cd_lns_sa_cascade_dual_levy.placer import CDLNSSACascadeDualLevyPlacer

class CDLNSSACascadeDualLevyK5Placer(CDLNSSACascadeDualLevyPlacer):
    def __init__(self):
        super().__init__(K_eps=5)
