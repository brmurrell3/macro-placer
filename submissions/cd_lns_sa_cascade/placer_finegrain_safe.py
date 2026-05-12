"""Finegrain with budget_seconds=2700s — safer wall margin for partcl box.

Finegrain at b=3000 hit max wall 3354s = 55.9 min (4 min margin to 60-min
cap). If partcl EPYC is even 10% slower than OCI EPYC, ibm18-class benches
risk exceeding cap. b=2700 gives 15-min margin.

Trade-off: slightly less CD work → maybe +0.3-0.5% on proxy.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from submissions.cd_lns_sa_cascade.placer_finegrain import CDLNSSACascadeFineGrain


class CDLNSSACascadeFineGrainSafe(CDLNSSACascadeFineGrain):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 2700.0)
        super().__init__(**kwargs)
