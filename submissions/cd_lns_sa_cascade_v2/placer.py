"""v2 submission — composes Option C (cascade saddle + portfolio saddle +
periphery wrapper) with H1 (LP-dual destroy ranking in LNS) and/or H2
(population annealing replacing SA-v2). Toggled at import time via env
vars MPC_V2_H1=1 / MPC_V2_H2=1.

The composer uses monkey-patches over the existing `cd_lns_sa.placer`
module so neither macro_place/ nor any existing submission is modified.
After May 19's kill gate, the v2 submission ships with whichever flags
the gate cleared.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-use Option C unchanged. Load via importlib because submissions/ is not
# a package and the dependency chain is non-trivial.
_OPTC_PATH = _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
_spec = importlib.util.spec_from_file_location("_cd_lns_sa_cascade_stacked_periphery", str(_OPTC_PATH))
_optc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_optc)

CDLNSSACascadeStackedPeripheryPlacer = _optc.CDLNSSACascadeStackedPeripheryPlacer

# Feature flags.
H1_ENABLED = os.environ.get("MPC_V2_H1", "0") == "1"
H2_ENABLED = os.environ.get("MPC_V2_H2", "0") == "1"

_patches_applied = []


def _apply_patches():
    """Install monkey-patches into cd_lns_sa.placer based on env flags.

    Idempotent — multiple imports of this module won't double-patch.
    """
    if _patches_applied:
        return

    # We must import cd_lns_sa as a module to patch it.
    cdlns_path = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
    cdlns_spec = importlib.util.spec_from_file_location("submissions.cd_lns_sa.placer", str(cdlns_path))
    cdlns_mod = importlib.util.module_from_spec(cdlns_spec)
    # Register before exec so internal `from submissions.cd_lns_sa.placer import ...`
    # references resolve consistently.
    sys.modules["submissions.cd_lns_sa.placer"] = cdlns_mod
    cdlns_spec.loader.exec_module(cdlns_mod)

    if H1_ENABLED:
        h1_dir = _ROOT / "experiments" / "E110_lp_dual_congestion" / "code"
        if str(h1_dir) not in sys.path:
            sys.path.insert(0, str(h1_dir))
        from lp_destroy_rank import lp_dual_destroy
        cdlns_mod._cost_aware_destroy = lp_dual_destroy  # type: ignore[attr-defined]
        _patches_applied.append("H1:lp_dual_destroy")

    if H2_ENABLED:
        h2_dir = _ROOT / "experiments" / "E111_population_annealing" / "code"
        if str(h2_dir) not in sys.path:
            sys.path.insert(0, str(h2_dir))
        from pa_core import run_pa_polish
        cdlns_mod.run_sa_polish_v2 = run_pa_polish  # type: ignore[attr-defined]
        _patches_applied.append("H2:run_pa_polish")


_apply_patches()


class CDLNSSACascadeV2Placer:
    """v2 = Option C wrapped with H1/H2 monkey-patches selected by env flags.

    All Option C kwargs forward unchanged; v2 differs only via the patches
    applied above at import time.
    """

    def __init__(self, **kwargs):
        self._inner = CDLNSSACascadeStackedPeripheryPlacer(**kwargs)
        self.h1 = H1_ENABLED
        self.h2 = H2_ENABLED
        if H1_ENABLED or H2_ENABLED:
            print(
                f"[v2] active patches: {','.join(_patches_applied) if _patches_applied else 'none'}",
                flush=True,
            )

    def place(self, benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
