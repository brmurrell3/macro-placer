"""E120 — ContinuousHessianSaddle placer (submission wrapper).

V3 Adam descent + Hessian-eigvec saddle escape in CONTINUOUS space +
CD polish. Novel: extends the E84 cascade saddle escape from the
combinatorial CD plateau into the smooth-proxy basin chosen by Adam.

Default budget 720s/bench (matches the V3Min ovl10 720s champion's
budget for a fair head-to-head). The saddle escape stages consume the
descent budget rather than extending it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_E120 = _ROOT / "experiments" / "E120_continuous_saddle" / "code"
_spec = importlib.util.spec_from_file_location(
    "_e120_continuous_saddle", str(_E120 / "continuous_saddle_placer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ContinuousHessianSaddlePlacer = _mod.ContinuousHessianSaddlePlacer


class E120ContinuousSaddlePlacer(ContinuousHessianSaddlePlacer):
    """Default config tuned for fair comparison against V3Min ovl10 720s.

    `place(benchmark)` returns a [num_macros, 2] tensor of (x, y) center
    coordinates. Hard macros are non-overlapping; fixed macros are not
    moved; all macros are within canvas bounds.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 720.0)
        kwargs.setdefault("num_steps_phaseA", 300)
        kwargs.setdefault("num_steps_resume", 120)
        kwargs.setdefault("max_saddle_stages", 2)
        kwargs.setdefault("overlap_lambda_end", 10.0)
        kwargs.setdefault("cd_polish_s", 300.0)
        kwargs.setdefault("eigsh_maxiter", 300)
        kwargs.setdefault("verbose", False)
        super().__init__(**kwargs)
