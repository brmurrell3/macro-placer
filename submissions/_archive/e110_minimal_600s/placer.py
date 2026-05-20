"""E110Minimal600s — minimal placer with 600s budget per bench.

For testing Carrotato-class throughput. Total --all wall ~2 hr (jobs=4).
If quality holds vs 3300s budget, this is the path to fast iteration.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BASE = _ROOT / "submissions" / "e110_minimal" / "placer.py"
_spec = importlib.util.spec_from_file_location("e110_minimal_base", str(_BASE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class E110Minimal600sPlacer(_mod.E110MinimalPlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 600.0)
        kwargs.setdefault("cd_min_polish_s", 480.0)  # ~8 min CD; rest descent + buffer
        super().__init__(**kwargs)
