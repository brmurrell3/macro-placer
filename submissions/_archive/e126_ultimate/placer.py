"""E126 — UltimatePlacer (submission wrapper).

3-way composition of:
- E115 FastDiffProxy (index_select + single-pass trace: 3-16x per-step speedup)
- E117 Gaussian density (C^inf erf-smeared: -2.71% avg on dense benches)
- E120 Continuous Hessian saddle escape (-1 to -2.2% per bench)

The V4-fast backbone enables more Adam steps within the budget, which
unlocks deeper saddle cascades and longer resume descents than E125
(which uses the slower V3 backbone).

Default budget: 1500s/bench (matches E120-1500s and E125 budgets).
Device defaults to CPU; CUDA can be selected via constructor or env var.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_E126_CODE = _ROOT / "experiments" / "E126_fast_gaussian_hessian" / "code"
_spec = importlib.util.spec_from_file_location(
    "_e126_ultimate", str(_E126_CODE / "ultimate_placer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
UltimatePlacer = _mod.UltimatePlacer


class E126UltimatePlacer(UltimatePlacer):
    """Default config for 1500s/bench with V4 backbone.

    `place(benchmark)` returns a [num_macros, 2] tensor of (x, y) center
    coordinates. Hard macros are non-overlapping; fixed macros are not
    moved; all macros are within canvas bounds. Zero overlaps guaranteed
    (legalize + project_overlaps + CD polish enforces this; raises if any
    are found at end).
    """

    def __init__(self, **kwargs):
        # Saddle cascade
        kwargs.setdefault("budget_seconds", 1500.0)
        kwargs.setdefault("num_steps_phaseA", 500)
        kwargs.setdefault("num_steps_resume", 200)
        kwargs.setdefault("max_saddle_stages", 3)
        # Adam descent
        kwargs.setdefault("overlap_lambda_end", 10.0)
        # CD polish
        kwargs.setdefault("cd_polish_s", 720.0)
        # Gaussian density (E117 defaults — validated on dense benches)
        kwargs.setdefault("sigma_scale", 1.0)
        kwargs.setdefault("sigma_floor_frac", 0.5)
        # Misc
        kwargs.setdefault("eigsh_maxiter", 300)
        kwargs.setdefault("verbose", False)
        kwargs.setdefault("device", "cpu")  # explicit CPU default for harness
        super().__init__(**kwargs)
