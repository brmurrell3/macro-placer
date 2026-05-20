"""V3Min ovl10 multi-seed ensemble at 12-min total budget."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BASE = _ROOT / "submissions" / "e111_minimal_multiseed" / "placer.py"
_spec = importlib.util.spec_from_file_location("base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class E111MinimalOvl10MultiseedPlacer(_mod.E111MinimalMultiseedPlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 720.0)
        kwargs.setdefault("n_seeds", 3)
        kwargs.setdefault("overlap_lambda_end", 10.0)
        super().__init__(**kwargs)
