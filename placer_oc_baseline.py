"""Baseline launcher pointing at Option C (no WireMask).

Used for `--all` baseline runs at the same budget/parallelism as the
WireMask-integrated launcher, to give apples-to-apples comparison.

Respects PLACER_BUDGET env var (same as `placer.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import importlib.util  # noqa: E402

_PLACER_PATH = (
    _REPO / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("_oc_inner", str(_PLACER_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

import os  # noqa: E402

_DEFAULT_BUDGET = float(os.environ.get("PLACER_BUDGET", "3300"))


class Placer(_mod.CDLNSSACascadeStackedPeripheryPlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", _DEFAULT_BUDGET)
        super().__init__(**kwargs)
