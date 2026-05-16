import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Multi-iter Lévy variant: short polish (60s) → fit 3+ saddle iters per bench.

Rationale: cd_lns_sa_cascade_levy on --all stopped at iters=1 every bench because
polish_budget=180s × 6 candidates = 1080s per iter, leaving <120s for iter 2.
CD-adaptive plateaus at ~30-90s anyway (per smoke + log inspection); polish=60s
gives equivalent polish quality. With 6 × 60 = 360s/iter, the saddle phase
(~1000-1100s on the 3300s budget) fits ~3 iters. Each iter explores a NEW
eigvec direction (state moves → Hessian changes), so the lift compounds.

No bench-specific hyperparameters. Same Lévy distribution, same K_eps=3.
Only the per-candidate polish budget changes.
"""
from submissions.cd_lns_sa_cascade_levy.placer import CDLNSSACascadeLevyPlacer

class CDLNSSACascadeLevyMultiIterPlacer(CDLNSSACascadeLevyPlacer):
    def __init__(self):
        super().__init__(polish_budget=60.0, max_iters=6)
