"""E169 — E166 with portfolio max_iters=3 (was 2).

Single hyperparameter change to test if portfolio's lift compounds at iter 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E166 = _ROOT / "experiments" / "E166_v4_full_stack" / "code" / "placer.py"

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_e166_placer_e169", str(_E166))
_e166_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_e166_mod)
E166Placer = _e166_mod.Placer


class Placer(E166Placer):
    """E166 with 3 portfolio iters and reallocated budget."""

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 2800.0)
        kwargs.setdefault("cascade_reserve_s", 300.0)
        kwargs.setdefault("portfolio_reserve_s", 900.0)
        kwargs.setdefault("portfolio_max_iters", 3)
        super().__init__(**kwargs)
