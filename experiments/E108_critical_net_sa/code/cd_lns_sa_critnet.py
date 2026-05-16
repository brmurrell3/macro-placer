"""E108 — CDLNSSACritNet: E25 with critical-net-weighted SA polish.

Monkey-patches `run_sa_polish_v2` in `submissions.cd_lns_sa.placer` BEFORE
loading `CDLNSSAPlacer`, then exports a `CDLNSSACritNetPlacer` that is
identical to E25 in every way except the SA macro-selection distribution.

Default weight_power=2.0 (quadratic in net degree). mix_uniform=0.0 (pure
critical-net sampling). These are tunable via the placer constructor.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import E25 placer module first
_E25_SPEC = importlib.util.spec_from_file_location(
    "_e25_critnet", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py"))
_E25 = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25)

# Pull in critnet replacement
from critical_net_sa import run_sa_polish_v2_critnet

# Monkey-patch: subsequent calls to _E25.run_sa_polish_v2 will dispatch to critnet.
# We wrap it with default critnet params (weight_power=2.0, mix_uniform=0.0).
def _patched_sa(*args, **kwargs):
    kwargs.setdefault("weight_power", _PATCH_CONFIG["weight_power"])
    kwargs.setdefault("mix_uniform", _PATCH_CONFIG["mix_uniform"])
    return run_sa_polish_v2_critnet(*args, **kwargs)

_PATCH_CONFIG = {"weight_power": 2.0, "mix_uniform": 0.0}
_E25.run_sa_polish_v2 = _patched_sa


class CDLNSSACritNetPlacer(_E25.CDLNSSAPlacer):
    """E25 with critical-net-weighted SA polish.

    Extra params:
      weight_power: exponent on net degree in per-macro weight (default 2.0).
      mix_uniform: probability of uniform fallback per move (default 0.0).
    """
    def __init__(
        self,
        weight_power: float = 2.0,
        mix_uniform: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _PATCH_CONFIG["weight_power"] = float(weight_power)
        _PATCH_CONFIG["mix_uniform"] = float(mix_uniform)
        self.weight_power = float(weight_power)
        self.mix_uniform = float(mix_uniform)

    def place(self, benchmark):
        return super().place(benchmark)


# Top-level alias for the evaluator harness.
Placer = CDLNSSACritNetPlacer
