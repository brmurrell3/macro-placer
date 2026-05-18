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

    H1 (LP-dual destroy) is applied via monkey-patches inside the cascade
    pipeline at import time. H2 (population annealing) runs as a DEDICATED
    post-cascade polish phase with its own time budget — the inner cascade
    pipeline gets a reduced budget so PA has room to do real work.

    Why H2 needs its own budget: PA's replica build alone is ~1s × N; the
    SA budget inside Option C's E25 phase is only ~2-5% of the total. Even
    at the default 3300s budget, inner SA gets ~90s — too tight for 24
    replicas × 40 ladder steps. Reserving a dedicated 600s for PA yields
    a meaningful ladder + resampling.
    """

    def __init__(self, pa_polish_budget: float = 600.0, **kwargs):
        self.h1 = H1_ENABLED
        self.h2 = H2_ENABLED
        self.pa_polish_budget = float(pa_polish_budget)

        # Carve the PA budget out of total budget_seconds so total wall is
        # preserved (judge box has a 60-min/bench cap; we can't overrun).
        total_budget = kwargs.get("budget_seconds", 3300.0)
        if H2_ENABLED and total_budget is not None:
            reserved = min(self.pa_polish_budget, total_budget * 0.25)
            self.pa_polish_budget = reserved
            kwargs["budget_seconds"] = max(total_budget - reserved, 60.0)
        else:
            self.pa_polish_budget = 0.0

        self._inner = CDLNSSACascadeStackedPeripheryPlacer(**kwargs)
        if H1_ENABLED or H2_ENABLED:
            print(
                f"[v2] active patches: {','.join(_patches_applied) if _patches_applied else 'none'}",
                flush=True,
            )
            print(
                f"[v2] budget split: cascade={kwargs.get('budget_seconds', 'default'):.0f}s "
                f"PA-polish={self.pa_polish_budget:.0f}s",
                flush=True,
            )

    def place(self, benchmark) -> torch.Tensor:
        placement = self._inner.place(benchmark)
        if not H2_ENABLED or self.pa_polish_budget < 30.0:
            return placement

        # Dedicated post-cascade PA polish.
        from macro_place.bench_paths import find_benchmark_dir
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

        try:
            bench_dir = find_benchmark_dir(benchmark.name)
            _bench_reload, plc = load_benchmark_from_dir(str(bench_dir))
        except Exception as exc:
            print(f"[v2] PA polish skipped (couldn't reload plc): {exc!r}", flush=True)
            return placement

        from macro_place.incremental_evaluator import IncrementalProxyEvaluator
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
        pre_pa_proxy = float(evaluator.current_cost()["proxy"])
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]

        sys.path.insert(0, str(_ROOT / "experiments" / "E111_population_annealing" / "code"))
        from pa_core import run_pa_polish

        print(f"[v2] post-cascade PA polish: budget={self.pa_polish_budget:.0f}s "
              f"pre_proxy={pre_pa_proxy:.5f}", flush=True)

        try:
            stats = run_pa_polish(
                evaluator, benchmark, plc, hard_movable,
                time_budget_s=self.pa_polish_budget,
                log_fn=lambda s: print(s, flush=True),
            )
        except Exception as exc:
            print(f"[v2] PA polish raised: {exc!r}; using pre-PA placement", flush=True)
            return placement

        new_placement = evaluator.placement.detach().clone().to(torch.float32)
        new_proxy = float(compute_proxy_cost(new_placement, benchmark, plc)["proxy_cost"])
        new_ovl = int(compute_overlap_metrics(new_placement, benchmark)["overlap_count"])

        if new_ovl > 0:
            print(f"[v2] PA produced {new_ovl} overlaps; reverting to pre-PA placement", flush=True)
            return placement
        if new_proxy > pre_pa_proxy + 1e-5:
            print(f"[v2] PA regressed ({pre_pa_proxy:.5f} → {new_proxy:.5f}); "
                  "reverting", flush=True)
            return placement
        print(f"[v2] PA accepted: {pre_pa_proxy:.5f} → {new_proxy:.5f} "
              f"({(new_proxy-pre_pa_proxy)/pre_pa_proxy*100:+.3f}%)", flush=True)
        return new_placement
