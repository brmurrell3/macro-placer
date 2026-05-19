"""v2 submission — frozen-extended variant.

Forces H2=1 with H2_VARIANT=extended at import time so the eval harness
(which instantiates placers with no args and no env-var setup) gets the
extended-SA post-cascade polish without any external configuration.

H1 (LP-dual destroy) is intentionally NOT enabled — Day 1 isolated
smokes showed LP-dual lifts within noise (-0.24% to +0.087% across
three configurations on ibm01/ibm13), so it doesn't justify the LP
solve overhead.

To toggle features at runtime for experimentation, use the parent
placer at submissions/cd_lns_sa_cascade_v2/placer.py with env vars
MPC_V2_H1, MPC_V2_H2, MPC_V2_H2_VARIANT.
"""
from __future__ import annotations

import os as _os

# Pin env vars BEFORE importing the composer so its module-load-time
# patch detection sees the right config.
_os.environ["MPC_V2_H1"] = "0"
_os.environ["MPC_V2_H2"] = "1"
_os.environ["MPC_V2_H2_VARIANT"] = "extended"

import importlib.util as _il
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

_COMPOSER_PATH = _ROOT / "submissions" / "cd_lns_sa_cascade_v2" / "placer.py"
_spec = _il.spec_from_file_location("_v2_composer", str(_COMPOSER_PATH))
_mod = _il.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class CDLNSSACascadeV2ExtendedPlacer:
    """Frozen-extended v2 placer for leaderboard submission.

    Wraps the v2 composer with H2=1 + variant=extended forced at import
    time. No constructor args needed; eval harness instantiates with `()`.
    """

    def __init__(self, **kwargs):
        # Default to the standard challenge budget (60-min/bench).
        if "budget_seconds" not in kwargs:
            kwargs["budget_seconds"] = 3300.0
        self._inner = _mod.CDLNSSACascadeV2Placer(**kwargs)

    def place(self, benchmark):
        return self._inner.place(benchmark)
