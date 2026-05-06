"""E78 CDLNSE61V2HessianPlacer — E61V2 plateau + compressed Hessian saddle.

Pipeline:
  1. Run CDLNSGACrossoverPlacer (E61V2): E25 → E41 → GA crossover → polish.
  2. Apply compressed Hessian saddle escape on the E61V2 plateau.
  3. Return min-proxy among {E61V2 raw, Hessian-layered}.

Target benches: ibm08, ibm14, ibm16, ibm17, ibm18 (where E74 used the E48
hybrid plateau, not E61V2 — E78 tests if swapping the plateau source unlocks
extra lift, as it did for ibm12 and ibm15 in E74's per-bench layering).

Per bench: ~5 hr E61V2 + 18 min Hessian (1 eig × 2 signs × 3 eps × 180s) ≈
~5.3 hr/bench. Five-bench wave ~27 hr serial; ~7-8 hr `--jobs 4`.

Reference: experiments/E78_layered_e61v2_e74/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Patch loader for Windows path normalization (avoid modifying loader.py).
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
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E61V2 placer (multi-stage pipeline).
from experiments.E61_ga_crossover.code.cd_lns_ga_crossover import CDLNSGACrossoverPlacer

# Reuse E74's saddle escape.
_E74_PATH = _ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_placer_ref", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape


class CDLNSE61V2HessianPlacer:
    """E78 — E61V2 plateau + compressed Hessian saddle escape."""

    def __init__(
        self,
        n_eigvecs: int = 1,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        verbose: bool = True,
    ):
        self.n_eigvecs = n_eigvecs
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== CDLNSE61V2HessianPlacer ({benchmark.name}) ===")
        t0 = time.time()

        # Reload plc for proxy computation and saddle escape.
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: E61V2 plateau (E25 → E41 → GA crossover → polish).
        log("  Phase 1: E61V2 (CDLNSGACrossoverPlacer)")
        e61v2_state = CDLNSGACrossoverPlacer().place(benchmark)
        e61v2_proxy = float(compute_proxy_cost(e61v2_state, benchmark, plc)["proxy_cost"])
        log(f"  E61V2 done: proxy={e61v2_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # Phase 2: Compressed Hessian saddle escape.
        log(f"  Phase 2: Hessian saddle escape (n_eigvecs={self.n_eigvecs}, "
            f"eps={self.eps_values}, polish={self.polish_budget:.0f}s)")
        try:
            saddle_state, saddle_proxy = _saddle_escape(
                e61v2_state, benchmark, plc,
                n_eigvecs=self.n_eigvecs,
                eps_values=self.eps_values,
                polish_budget=self.polish_budget,
                log=log if self.verbose else None,
            )
        except Exception as exc:
            log(f"  Hessian saddle escape failed: {exc}; using E61V2 plateau")
            saddle_state = e61v2_state
            saddle_proxy = e61v2_proxy

        # Phase 3: Best of (E61V2, layered).
        if saddle_proxy < e61v2_proxy - 1e-7:
            best_state, best_proxy, best_name = saddle_state, saddle_proxy, "saddle"
        else:
            best_state, best_proxy, best_name = e61v2_state, e61v2_proxy, "E61V2"
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"(E61V2={e61v2_proxy:.5f}, saddle={saddle_proxy:.5f})  "
            f"total wall={time.time() - t0:.0f}s")

        # Validate zero overlap.
        ovl = compute_overlap_metrics(best_state, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"E78 winner has {ovl} hard-macro overlaps — falling back to E61V2"
            )
        return best_state
