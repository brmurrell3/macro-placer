"""E77 CDLNSSAHessianSharper — E74 with k=5 eigvecs and finer ε grid.

Strategy: same architecture as E79 (parallel E25⊥E41 + Hessian saddle), but
with broader spectrum (k=5) and finer ε grid {0.1, 0.3, 1.0, 3.0, 10.0}.

Per bench: ~30 min E25/E41 (parallel) + 50 polish trials × 180 s Hessian =
~3 hr/bench. Run only on ibm12-18 in normal use; --all is allowed but slow.

Reference: experiments/E77_sharper_hessian/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Patch loader before any downstream imports trigger plc_client_os.
import macro_place.loader as _loader_mod
_orig_load_benchmark = _loader_mod.load_benchmark


def _patched_load_benchmark(netlist_file, plc_file=None, name=None):
    netlist_file = str(netlist_file).replace("\\", "/")
    if plc_file is not None:
        plc_file = str(plc_file).replace("\\", "/")
    return _orig_load_benchmark(netlist_file, plc_file, name)


_loader_mod.load_benchmark = _patched_load_benchmark

# Reuse the entire E79 placer machinery — only the hyperparameters change.
_E79_PATH = (
    _ROOT
    / "experiments"
    / "E79_hardware_portability"
    / "code"
    / "cd_lns_sa_hessian_fast.py"
)
_E79_SPEC = importlib.util.spec_from_file_location("e79_placer_ref", str(_E79_PATH))
_E79_MOD = importlib.util.module_from_spec(_E79_SPEC)
_E79_SPEC.loader.exec_module(_E79_MOD)

CDLNSSAHessianFastPlacer = _E79_MOD.CDLNSSAHessianFastPlacer


class CDLNSSAHessianSharperPlacer(CDLNSSAHessianFastPlacer):
    """E77 — wider Hessian spectrum + finer ε grid (proxy improvement target).

    Subclasses E79; only the saddle-escape hyperparameters change.
    """

    def __init__(
        self,
        n_eigvecs: int = 5,
        eps_values: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0),
        polish_budget: float = 180.0,
        parallel: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            n_eigvecs=n_eigvecs,
            eps_values=eps_values,
            polish_budget=polish_budget,
            parallel=parallel,
            verbose=verbose,
        )
