"""E168 — E166 with 7-weight portfolio + reallocated budget.

Same pipeline as E166 (multi-init multi-seed + cascade + portfolio) but
extends portfolio to 7 weights (adds single-component extremes) and
shifts budget from cascade to portfolio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E166 = _ROOT / "experiments" / "E166_v4_full_stack" / "code" / "placer.py"

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_e166_placer", str(_E166))
_e166_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_e166_mod)
E166Placer = _e166_mod.Placer


# 7-weight portfolio: E166's 4 + 3 single-component extremes
WIDER_PORTFOLIO = [
    (1.0, 0.5, 0.5),  # canonical
    (1.0, 0.0, 1.0),  # cong-focus
    (1.0, 1.0, 0.0),  # density-focus
    (0.0, 1.0, 1.0),  # non-WL
    (1.0, 0.0, 0.0),  # pure WL
    (0.0, 1.0, 0.0),  # pure density
    (0.0, 0.0, 1.0),  # pure congestion
]


class Placer(E166Placer):
    """E166 with 7-weight portfolio and budget shifted to portfolio."""

    def __init__(self, **kwargs):
        # Defaults that override E166
        kwargs.setdefault("budget_seconds", 2400.0)
        kwargs.setdefault("cascade_reserve_s", 200.0)
        kwargs.setdefault("cascade_max_iters", 1)
        kwargs.setdefault("portfolio_reserve_s", 800.0)
        kwargs.setdefault("portfolio_max_iters", 2)
        super().__init__(**kwargs)


# E166Placer doesn't currently read portfolio from a kwarg — it hardcodes
# DEFAULT_PORTFOLIO. We monkey-patch its DEFAULT_PORTFOLIO module-level
# only for the duration of this experiment's runs.
_e166_mod.DEFAULT_PORTFOLIO = WIDER_PORTFOLIO
