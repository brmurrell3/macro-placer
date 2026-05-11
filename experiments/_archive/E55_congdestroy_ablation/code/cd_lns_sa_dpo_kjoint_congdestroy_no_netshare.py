"""E55 — E54 ablation: congestion-aware destroy WITHOUT net-share weighting.

Tests whether the per-net 1/n_pins term in `_congestion_aware_destroy`
is load-bearing for E54's --fast lift. If `include_net_share=False`
gives equivalent results, the simpler direct-macro-routing-only
ranking wins.

Subclass of E54's placer with the flag flipped. All other
hyperparameters identical.

Reference:
- E54 — parent. `experiments/E54_congestion_destroy/code/cd_lns_sa_dpo_kjoint_congdestroy.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.E54_congestion_destroy.code.cd_lns_sa_dpo_kjoint_congdestroy import (
    CDLNSSADPOKJointCongDestroyPlacer,
)


class CDLNSSADPOKJointCongDestroyNoNetShare(CDLNSSADPOKJointCongDestroyPlacer):
    """E55: E54 with net-share weighting disabled."""

    def __init__(self, **kwargs):
        # Force the flag off; leave all other defaults from E54.
        kwargs["lns_include_net_share"] = False
        super().__init__(**kwargs)
