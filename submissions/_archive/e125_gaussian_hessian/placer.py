"""E125 — Composed Gaussian density + Hessian saddle escape placer.

Composition of two validated architectural wins:
  - E117 Gaussian-smoothed density (-2.71% avg on dense benches)
  - E120 Continuous Hessian saddle escape (-1 to -2.2% per bench)

Both lift via architectural changes (not fidelity). The Gaussian density
gives Adam (and the saddle eigenvector calculation) a C^inf smooth
landscape, hypothetically letting the saddle escape find better
escape directions than on the C^0 grid-bin density.

Default budget: 1500s/bench (matches e120_continuous_saddle_1500s).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_E125_CODE = _ROOT / "experiments" / "E125_gaussian_hessian" / "code"
_spec = importlib.util.spec_from_file_location(
    "_e125_composed", str(_E125_CODE / "composed_placer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GaussianHessianComposedPlacer = _mod.GaussianHessianComposedPlacer


class E125GaussianHessianPlacer(GaussianHessianComposedPlacer):
    """Default config tuned for 1500s budget.

    `place(benchmark)` returns a [num_macros, 2] tensor of (x, y) center
    coordinates. Hard macros are non-overlapping; fixed macros are not
    moved; all macros are within canvas bounds.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 1500.0)
        kwargs.setdefault("num_steps_phaseA", 300)
        kwargs.setdefault("num_steps_resume", 150)
        kwargs.setdefault("max_saddle_stages", 2)
        kwargs.setdefault("overlap_lambda_end", 10.0)
        kwargs.setdefault("cd_polish_s", 720.0)  # 12 min CD polish
        kwargs.setdefault("eigsh_maxiter", 300)
        # Gaussian density defaults (from E117)
        kwargs.setdefault("sigma_scale", 1.0)
        kwargs.setdefault("sigma_floor_frac", 0.5)
        kwargs.setdefault("verbose", False)
        super().__init__(**kwargs)
