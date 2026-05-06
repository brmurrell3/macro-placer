"""E84 DREAMPlaceSaddlePlacer — DREAMPlace plateau + Hessian saddle escape.

Pipeline per benchmark:
  1. Convert benchmark → Bookshelf (LF, tab-separated, matches adaptec1).
  2. Run DREAMPlace global place via Docker (GPU, ~10 sec).
  3. project_overlaps to legalize DREAMPlace output.
  4. Hessian saddle escape (E74 mechanism, k=1, ε={0.3, 1.0, 3.0}, polish 180s).
  5. Validate zero overlaps; return.

Wall budget on RTX 4070 / RTX 6000 Ada:
  Phase 1+2 (DREAMPlace):     ~10-30 sec/bench
  Phase 3 (project_overlaps): ~5 sec
  Phase 4 (Hessian saddle):   ~18 min full (6 polishes × 180s)
  Total:                       ~20 min/bench  ✓ massive headroom under 60-min cap

Reference: experiments/E84_dreamplace_saddle/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Patch loader for Windows path normalization (matches E83/E79 pattern).
import macro_place.loader as _loader_mod
_orig_load_benchmark = _loader_mod.load_benchmark


def _patched_load_benchmark(netlist_file, plc_file=None, name=None):
    netlist_file = str(netlist_file).replace("\\", "/")
    if plc_file is not None:
        plc_file = str(plc_file).replace("\\", "/")
    return _orig_load_benchmark(netlist_file, plc_file, name)


_loader_mod.load_benchmark = _patched_load_benchmark

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse Hessian saddle escape from E74.
_E74_PATH = _REPO_ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_for_e84", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape

# DREAMPlace wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dreamplace import run_dreamplace


class DREAMPlaceSaddlePlacer:
    """E84 — DREAMPlace plateau + Hessian saddle escape.

    Single algorithm, same hyperparameters every benchmark.  GPU-accelerated
    plateau in <30 sec leaves ~57 min of the 60-min cap for the saddle escape
    on every bench.
    """

    def __init__(
        self,
        n_eigvecs: int = 1,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        gpu: bool = True,
        verbose: bool = True,
    ):
        self.n_eigvecs = n_eigvecs
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.gpu = gpu
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== DREAMPlaceSaddlePlacer ({benchmark.name}) ===")
        t0 = time.time()

        # Reload plc for proxy computation and saddle escape.
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: DREAMPlace plateau.
        work_dir = (
            _REPO_ROOT / "experiments" / "E84_dreamplace_saddle" / "dp_runs"
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        log("  Phase 1: DREAMPlace global placement (GPU)")
        dp_placement = run_dreamplace(benchmark, work_dir, gpu=self.gpu, log_fn=log)
        dp_proxy_raw = float(
            compute_proxy_cost(dp_placement, benchmark, plc)["proxy_cost"]
        )
        dp_overlaps_raw = compute_overlap_metrics(dp_placement, benchmark)["overlap_count"]
        log(f"  DREAMPlace raw: proxy={dp_proxy_raw:.5f}, "
            f"overlaps={dp_overlaps_raw} (wall={time.time() - t0:.0f}s)")

        # Phase 2: project_overlaps to legalize DREAMPlace output.
        log("  Phase 2: project_overlaps")
        dp_legal, proj_iters = project_overlaps(dp_placement, benchmark)
        dp_proxy = float(compute_proxy_cost(dp_legal, benchmark, plc)["proxy_cost"])
        dp_overlaps = compute_overlap_metrics(dp_legal, benchmark)["overlap_count"]
        log(f"  Legalized: proxy={dp_proxy:.5f}, overlaps={dp_overlaps}, "
            f"proj iters={proj_iters} (wall={time.time() - t0:.0f}s)")

        if dp_overlaps > 0:
            log(f"  WARNING: {dp_overlaps} residual overlaps after project; "
                f"saddle escape may amplify them")

        # Phase 3: Hessian saddle escape on the legalized DREAMPlace plateau.
        log(f"  Phase 3: Hessian saddle escape "
            f"(n_eigvecs={self.n_eigvecs}, eps={self.eps_values}, "
            f"polish={self.polish_budget:.0f}s)")
        try:
            saddle_state, saddle_proxy = _saddle_escape(
                dp_legal, benchmark, plc,
                n_eigvecs=self.n_eigvecs,
                eps_values=self.eps_values,
                polish_budget=self.polish_budget,
                log=log if self.verbose else None,
            )
        except Exception as exc:
            log(f"  Saddle escape failed: {exc}; using DREAMPlace plateau")
            saddle_state = dp_legal
            saddle_proxy = dp_proxy

        # Best of (DREAMPlace, saddle).
        if saddle_proxy < dp_proxy - 1e-7:
            best_state, best_proxy, best_name = saddle_state, saddle_proxy, "saddle"
        else:
            best_state, best_proxy, best_name = dp_legal, dp_proxy, "DREAMPlace"
        total_wall = time.time() - t0
        log(
            f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"(DREAMPlace={dp_proxy:.5f}, saddle={saddle_proxy:.5f})  "
            f"total wall={total_wall:.0f}s"
        )

        ovl = compute_overlap_metrics(best_state, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"E84 winner has {ovl} hard-macro overlaps"
            )
        return best_state
