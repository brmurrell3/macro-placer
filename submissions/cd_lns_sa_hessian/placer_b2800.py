"""Wall-safe E74 with budget_seconds=2800s for tight 60-min cap.

Used for cloud --all to avoid the ibm17-style 60min overshoot.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_lns_sa_hessian.placer import CDLNSSAHessianPlacer


class CDLNSSAHessianPlacer2800(CDLNSSAHessianPlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 2800.0)
        super().__init__(**kwargs)
