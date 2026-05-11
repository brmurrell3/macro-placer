"""Wall-safe E48 hybrid at budget=3000s (no saddle escape).

Tests whether the simpler hybrid beats cascade under 60-min cap.
E48 hybrid runs E25 and E41 each in their default order, picks best.
Without saddle, each lane has budget for fuller CD/LNS/SA convergence.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_lns_sa_hybrid.placer import CDLNSSAHybridPlacer


class CDLNSSAHybridPlacerB3000(CDLNSSAHybridPlacer):
    """E48 hybrid with each phase budget set to fit 3000s total wall."""

    def __init__(self, **kwargs):
        # Split 3000s evenly across E25 and E41 (~1450s each + overhead).
        # E25: cd=1000, lns=225, sa=225 = 1450s
        # E41: cd=900, lns=200, sa=200, kjoint=150 = 1450s
        common = dict(cd_hard_cap_s=900, lns_budget_s=200, sa_budget_s=200)
        e41 = dict(kjoint_budget_s=150)
        kwargs.setdefault("common_kwargs", common)
        kwargs.setdefault("e41_kwargs", e41)
        super().__init__(**kwargs)
