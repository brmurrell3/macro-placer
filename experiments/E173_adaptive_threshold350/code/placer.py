"""E173 — E171 with threshold lowered to 350.

Routes ibm15 (393) and ibm11 (373) to E143 K-joint stack instead of
E166 extended stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

import importlib.util as _ilu
_E171 = _ROOT / "experiments" / "E171_adaptive_kjoint_gate" / "code" / "placer.py"
_e171_spec = _ilu.spec_from_file_location("_e171_for_e173", str(_E171))
_e171_mod = _ilu.module_from_spec(_e171_spec)
_e171_spec.loader.exec_module(_e171_mod)
E171Placer = _e171_mod.Placer


class Placer(E171Placer):
    def __init__(self, **kwargs):
        kwargs.setdefault("large_bench_threshold", 350)
        super().__init__(**kwargs)
