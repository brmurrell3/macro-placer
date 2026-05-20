"""V3Min ovl5 12-min — even lower overlap penalty."""
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


class E111MinimalOvl5_720sPlacer(_mod.E111MinimalPlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 720.0)
        kwargs.setdefault("cd_polish_s", 600.0)
        kwargs.setdefault("overlap_lambda_end", 5.0)
        super().__init__(**kwargs)
