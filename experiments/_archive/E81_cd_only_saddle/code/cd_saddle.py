"""E81 CDSaddlePlacer — minimal pipeline test for Hessian saddle hypothesis.

Tests whether the Hessian negative-eigenvalue saddle escape (the proxy-lift
mechanism that distinguished E74 from E48) still delivers when started from
a CHEAP plateau rather than the expensive E25/E41 plateau.

Pipeline per benchmark:
  1. SDF init.
  2. project_overlaps.
  3. CD adaptive (cap 1800 s, plateau-threshold 0.001).
  4. Hessian saddle escape (k=1, ε ∈ {0.3, 1.0, 3.0}, polish 180 s).
  5. Validate zero overlaps; preserve fixed macros.

NO LNS, NO SA, NO K-joint, NO DPO/E41 lane.

Target wall on M3 baseline: ~40-50 min/bench (vs E74 ~75 min, E79 ~65 min).
On AMD EPYC (probe scales 1.5×): ~60-75 min — fits 60-min cap on the
benches that don't blow CD's reduced budget.

Reference: experiments/E81_cd_only_saddle/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
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
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse E74's Hessian saddle machinery.
_E74_PATH = _ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_for_e81", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape


def _probe_hardware_scale() -> float:
    """scale = wall / baseline; clamp [1.0, 1.5] (no reduction below baseline)."""
    A = np.random.randn(512, 512).astype(np.float32)
    t0 = time.time()
    for _ in range(50):
        B = A @ A.T
        A = B / (float(np.linalg.norm(B)) + 1e-9)
    wall = time.time() - t0
    M3_MAX_BASELINE_S = 1.2
    scale = wall / M3_MAX_BASELINE_S
    return float(np.clip(scale, 1.0, 1.5))


class CDSaddlePlacer:
    """E81 — SDF + project + CD + Hessian saddle escape.

    Tests the hypothesis that the saddle escape carries the proxy lift,
    independent of the plateau depth.
    """

    def __init__(
        self,
        cd_cap_s: float = 1800.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        n_eigvecs: int = 1,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        verbose: bool = True,
    ):
        self.cd_cap_s = float(cd_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.n_eigvecs = int(n_eigvecs)
        self.eps_values = tuple(eps_values)
        self.polish_budget = float(polish_budget)
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== CDSaddlePlacer ({benchmark.name}) ===")
        t0 = time.time()

        # Hardware probe.
        hw_scale = _probe_hardware_scale()
        scaled_cd_cap = self.cd_cap_s * hw_scale
        scaled_polish = self.polish_budget * hw_scale
        log(f"  Hardware probe: scale={hw_scale:.3f}, "
            f"cd_cap {self.cd_cap_s:.0f}s → {scaled_cd_cap:.0f}s, "
            f"polish_budget {self.polish_budget:.0f}s → {scaled_polish:.0f}s")

        # Reload plc.
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: SDF init.
        log("  Phase 1: SDF init")
        placement = sdf_init(benchmark)
        log(f"    init wall = {time.time() - t0:.1f}s")

        # Phase 2: project_overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        log(f"  Phase 2: project_overlaps {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}")

        # Phase 3: CD adaptive.
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        log(f"  Phase 3: CD adaptive (cap={scaled_cd_cap:.0f}s, "
            f"init proxy={init_cost['proxy']:.5f})")
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=scaled_cd_cap,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        log(f"    CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, proxy={cd_proxy:.5f}")

        # Pull placement back; preserve fixed macros (saddle escape needs
        # float32 tensors with fixed macros pinned).
        cd_placement = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            cd_placement[fixed_mask] = original_positions[fixed_mask]
        cd_placement = cd_placement.to(torch.float32)

        # Recompute proxy in float32 (the evaluator used float64 internally).
        plateau_proxy = float(
            compute_proxy_cost(cd_placement, benchmark, plc)["proxy_cost"]
        )
        log(f"  CD plateau (float32 recompute): proxy={plateau_proxy:.5f}")

        # Phase 4: Hessian saddle escape.
        log(f"  Phase 4: Hessian saddle escape "
            f"(n_eigvecs={self.n_eigvecs}, eps={self.eps_values}, "
            f"polish={scaled_polish:.0f}s)")
        try:
            saddle_state, saddle_proxy = _saddle_escape(
                cd_placement, benchmark, plc,
                n_eigvecs=self.n_eigvecs,
                eps_values=self.eps_values,
                polish_budget=scaled_polish,
                log=log if self.verbose else None,
            )
        except Exception as exc:
            log(f"  Saddle escape failed: {exc}; using CD plateau")
            saddle_state = cd_placement
            saddle_proxy = plateau_proxy

        # Best of (CD plateau, saddle).
        if saddle_proxy < plateau_proxy - 1e-7:
            best_state, best_proxy, best_name = saddle_state, saddle_proxy, "saddle"
        else:
            best_state, best_proxy, best_name = cd_placement, plateau_proxy, "CD"
        log(
            f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"(CD={plateau_proxy:.5f}, saddle={saddle_proxy:.5f})  "
            f"total wall={time.time() - t0:.0f}s"
        )

        ovl = compute_overlap_metrics(best_state, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"E81 winner has {ovl} hard-macro overlaps"
            )
        return best_state
