"""Variant B (plateau-pick) with sweep-winner cfg overlap_lambda_end=10.

If ovl_end=10 lifts the gradient lane on --all, and plateau-pick lets
cascade compound that lift, this should beat both Variant A_ovl10 and
Option C.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BASE = (
    _ROOT / "submissions"
    / "cd_lns_sa_cascade_stacked_periphery_e110_plateau" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("e110_plateau_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_BasePlacer = _mod.CDLNSSACascadeStackedPeripheryE110PlateauPlacer


class CDLNSSACascadeStackedPeripheryE110PlateauOvl10Placer(_BasePlacer):
    """Variant B with overlap_lambda_end=10."""

    def __init__(self, **kwargs):
        kwargs.setdefault("e110_overlap_lambda_end", 10.0)
        super().__init__(**kwargs)
