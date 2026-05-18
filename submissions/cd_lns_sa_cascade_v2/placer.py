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


def _collect_target_modules():
    """Find every module that holds a reference to either
    `_cost_aware_destroy` or `run_sa_polish_v2`. Returns two lists:
    (modules_with_destroy, modules_with_sa).

    Three sources:
      1. Private modules from `_optc._stacked._E25_MOD` (cd_lns_sa loaded
         as "e25_placer" via importlib, NOT in sys.modules).
      2. `cd_lns_sa_kjoint` (E39, in sys.modules) — has its own inlined
         copies of both functions.
      3. `cd_lns_sa_dpo_kjoint` (E41, in sys.modules) — imports
         `run_sa_polish_v2` from E39 by reference, so we must patch
         E41's module-level name independently (Python's `from X import Y`
         copies the reference at import time).
    """
    import types

    targets_destroy: list = []
    targets_sa: list = []
    visited = set()
    max_depth = 6

    def visit_private(obj, depth: int):
        if depth > max_depth:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)
        if not isinstance(obj, types.ModuleType):
            return
        if hasattr(obj, "_cost_aware_destroy"):
            if obj not in targets_destroy:
                targets_destroy.append(obj)
        if hasattr(obj, "run_sa_polish_v2"):
            if obj not in targets_sa:
                targets_sa.append(obj)
        for attr_name in vars(obj):
            if attr_name.startswith("__"):
                continue
            if not (attr_name.startswith("_") or attr_name in ("stacked", "cascade")):
                continue
            try:
                val = getattr(obj, attr_name)
            except Exception:
                continue
            if isinstance(val, types.ModuleType):
                visit_private(val, depth + 1)

    # Source 1: private cascade-chain modules.
    visit_private(_optc, 0)

    # Sources 2 + 3: sys.modules scan.
    for name, mod in list(sys.modules.items()):
        if not isinstance(mod, types.ModuleType):
            continue
        if name.startswith("torch.") or name.startswith("numpy.") or name.startswith("scipy."):
            continue
        # Restrict to our repo's modules (filename-based filter avoids
        # touching anything unrelated).
        f = getattr(mod, "__file__", None)
        if not f or "/home/user/macro-placer/" not in f:
            continue
        try:
            if hasattr(mod, "_cost_aware_destroy") and mod not in targets_destroy:
                targets_destroy.append(mod)
        except Exception:
            pass
        try:
            if hasattr(mod, "run_sa_polish_v2") and mod not in targets_sa:
                targets_sa.append(mod)
        except Exception:
            pass

    return targets_destroy, targets_sa


def _apply_patches():
    """Install monkey-patches into every cascade-chain module that holds
    `_cost_aware_destroy` or `run_sa_polish_v2`. Idempotent.
    """
    if _patches_applied:
        return

    destroy_targets, sa_targets = _collect_target_modules()

    if H1_ENABLED:
        if not destroy_targets:
            print("[v2] WARNING: no _cost_aware_destroy targets found", flush=True)
        else:
            h1_dir = _ROOT / "experiments" / "E110_lp_dual_congestion" / "code"
            if str(h1_dir) not in sys.path:
                sys.path.insert(0, str(h1_dir))
            from lp_destroy_rank import lp_dual_destroy
            patched_names = []
            for m in destroy_targets:
                m._cost_aware_destroy = lp_dual_destroy
                patched_names.append(getattr(m, "__name__", "?"))
            _patches_applied.append(f"H1:lp_dual_destroy[{','.join(patched_names)}]")

    if H2_ENABLED:
        if not sa_targets:
            print("[v2] WARNING: no run_sa_polish_v2 targets found", flush=True)
        else:
            h2_dir = _ROOT / "experiments" / "E111_population_annealing" / "code"
            if str(h2_dir) not in sys.path:
                sys.path.insert(0, str(h2_dir))
            from pa_core import run_pa_polish
            patched_names = []
            for m in sa_targets:
                m.run_sa_polish_v2 = run_pa_polish
                patched_names.append(getattr(m, "__name__", "?"))
            _patches_applied.append(f"H2:run_pa_polish[{','.join(patched_names)}]")


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
