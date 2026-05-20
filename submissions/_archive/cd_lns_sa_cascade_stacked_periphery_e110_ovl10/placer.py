"""Variant A (Lane-4) with sweep-winner cfg: overlap_lambda_end=10.

Sweep --fast Δ_avg=-4.39% (4/4 wins). Default cfg only gave -1.27%.
Hypothesis: lower overlap penalty during descent lets Adam pack
macros tighter; greedy_macro_legalize handles the residuals.
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
    / "cd_lns_sa_cascade_stacked_periphery_e110" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("e110_lane4_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_BasePlacer = _mod.CDLNSSACascadeStackedPeripheryE110Placer


class CDLNSSACascadeStackedPeripheryE110Ovl10Placer(_BasePlacer):
    """Lane-4 with overlap_lambda_end=10 (sweep top cfg)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("e110_overlap_lambda_end", 10.0)
        super().__init__(**kwargs)
