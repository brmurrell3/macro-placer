import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""4-weight portfolio with K_eps=3 (heavier per-direction sampling)."""
from submissions.cd_lns_sa_cascade_portfolio_levy.placer import CDLNSSACascadePortfolioLevyPlacer

class CDLNSSACascadePortfolioLevyK3Placer(CDLNSSACascadePortfolioLevyPlacer):
    def __init__(self):
        super().__init__(K_eps=3)
