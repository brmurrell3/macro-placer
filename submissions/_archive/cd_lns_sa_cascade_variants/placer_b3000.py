"""Wall-safe cascade with budget_seconds=3000s for safe 60-min cap fit.

Tuned for EPYC: cascade b=3300 on cloud hit 3622s (ibm17) overshooting
the 60-min hard cap. b=3000 gives ~10-min margin while preserving most
of the cascade lift over b=2800.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadePlacer3000(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)
