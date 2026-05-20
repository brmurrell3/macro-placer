"""V3Minimal with 8-min budget per bench (480s).

V3Minimal at 4-min gave avg 1.030. Doubling budget should let CD polish
extract more, hopefully 1.020-1.025.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BASE = _ROOT / "submissions" / "e111_minimal" / "placer.py"
_spec = importlib.util.spec_from_file_location("base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class E111Minimal480sPlacer(_mod.E111MinimalPlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 480.0)
        kwargs.setdefault("cd_polish_s", 360.0)  # 6 min CD polish
        super().__init__(**kwargs)
