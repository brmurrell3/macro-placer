"""E12 ablation — random-destroy variant of grid-bin LNS.

Same algorithm as `cd_lns_gridbin.py` but the destroy step picks K macros
uniformly at random instead of cost-aware ranking.

Goal: tells us whether the cost-aware destroy step is load-bearing.
  * If random destroy gets the same gain as cost_aware → simplify the design
    (skip the O(num_hard) cost-ranking pass, save ~10% wall).
  * If random destroy is meaningfully worse → cost_aware is doing real work.

This is an ablation, not a candidate champion. Run on --fast only.
"""
from __future__ import annotations

import sys
from pathlib import Path

# This file lives at experiments/E12_grid_bin_lns/code/, so repo root is 4 levels up.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# After the 2026-04-28 promotion (ADR-007), the production placer lives at
# submissions/cd_lns_gridbin/placer.py — load it from there. This ablation
# stays in experiments/ as the experiment's record of "is cost-aware destroy
# load-bearing?" (it isn't — see ../notes.md).
import importlib.util as _ilu
_target = _ROOT / "submissions" / "cd_lns_gridbin" / "placer.py"
_spec = _ilu.spec_from_file_location("cd_lns_gridbin", str(_target))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CDLNSGridBinPlacer = _mod.CDLNSGridBinPlacer


class CDLNSGridBinRandomPlacer(CDLNSGridBinPlacer):
    """Random-destroy ablation."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(
            lns_destroy_strategy="random",
            verbose=verbose,
        )
