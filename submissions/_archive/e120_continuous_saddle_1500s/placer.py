"""E120 Hessian saddle at 1500s budget (25 min/bench).

E120 at 720s wins on easy benches but the saddle stages crowd out CD
polish on hard benches (ibm17). Increasing budget gives both saddle
escape AND enough CD polish time. Still well under judges' 3600s cap.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BASE = _ROOT / "submissions" / "e120_continuous_saddle" / "placer.py"
_spec = importlib.util.spec_from_file_location("e120_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class E120ContinuousSaddle1500sPlacer(_mod.E120ContinuousSaddlePlacer):
    """E120 at 1500s budget: enough for saddle escape + 12-min CD polish."""

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 1500.0)
        kwargs.setdefault("cd_polish_s", 720.0)  # 12 min CD (champion's CD time)
        super().__init__(**kwargs)
