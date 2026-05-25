"""E153 — E143 with portfolio_max_iters=3.

For large benches, tests if E148's deeper portfolio compounds with
K-joint+SA stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

import importlib.util as _ilu
_E143 = _ROOT / "experiments" / "E143_v4_full_kjoint_sa" / "code" / "placer.py"
_e143_spec = _ilu.spec_from_file_location("_e143_for_e153", str(_E143))
_e143_mod = _ilu.module_from_spec(_e143_spec)
_e143_spec.loader.exec_module(_e143_mod)
E143Placer = _e143_mod.Placer


class Placer(E143Placer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3300.0)
        kwargs.setdefault("portfolio_max_iters", 3)
        kwargs.setdefault("portfolio_reserve_s", 750.0)
        super().__init__(**kwargs)
