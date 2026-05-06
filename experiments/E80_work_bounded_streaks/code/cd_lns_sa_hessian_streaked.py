"""E80 CDLNSSAHessianStreaked — E79 + work-bounded streaks + drop K-joint.

Combines:
  * E79's parallel E25⊥E41 + compressed Hessian saddle escape
  * §Derisk Mitigation #3: 5-streak LNS + 1000-move SA-v2 streak termination
  * §Derisk Mitigation #4: drop K-joint K=3 phase from E41 (Hessian saddle
    finds deeper minima anyway)

Expected wall on M3 Max: 30-50 min/bench (depending on phase plateau timing).
Expected wall on AMD EPYC 9655P: ≤60 min via probe-scaled budgets + streaks.

The streaked LNS/SA functions are monkey-patched into the E25 and E41
modules in the parallel subprocess workers (see `_worker.py`), so the
champion's E25/E41 source files are NOT modified.

Reference: experiments/E80_work_bounded_streaks/manifest.md
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
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
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse Hessian saddle from E74.
_E74_PATH = _ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_for_e80", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape

_WORKER_SCRIPT = Path(__file__).parent / "_worker.py"


# ── Hardware probe (same as E79 fixed version) ─────────────────────────────


def _probe_hardware_scale() -> float:
    """scale = wall / baseline; clamp [1.0, 1.5]; never reduce below baseline."""
    A = np.random.randn(512, 512).astype(np.float32)
    t0 = time.time()
    for _ in range(50):
        B = A @ A.T
        A = B / (float(np.linalg.norm(B)) + 1e-9)
    wall = time.time() - t0
    M3_MAX_BASELINE_S = 1.2
    scale = wall / M3_MAX_BASELINE_S
    return float(np.clip(scale, 1.0, 1.5))


# ── Parallel E25⊥E41 subprocess workers ────────────────────────────────────


def _launch_worker(mode: str, bench_name: str) -> Tuple[subprocess.Popen, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + existing if existing else "")
    proc = subprocess.Popen(
        [sys.executable, str(_WORKER_SCRIPT), mode, bench_name, str(_ROOT), tmp.name],
        env=env,
    )
    return proc, tmp.name


def _run_parallel_e25_e41(bench_name: str, log) -> Tuple[torch.Tensor, torch.Tensor]:
    log("  Phases 1+2: launching E25 ‖ E41 streaked subprocesses...")
    t0 = time.time()
    proc25, path25 = _launch_worker("e25", bench_name)
    proc41, path41 = _launch_worker("e41", bench_name)
    rc25 = proc25.wait()
    rc41 = proc41.wait()
    elapsed = time.time() - t0
    log(f"  Parallel E25+E41 (streaked) done in {elapsed:.0f}s "
        f"(rc25={rc25}, rc41={rc41})")
    if rc25 != 0 or rc41 != 0:
        for p in (path25, path41):
            try:
                os.unlink(p)
            except OSError:
                pass
        raise RuntimeError(f"E80 worker failure: rc_e25={rc25}, rc_e41={rc41}")
    try:
        e25_np = np.load(path25)
        e41_np = np.load(path41)
    finally:
        for p in (path25, path41):
            try:
                os.unlink(p)
            except OSError:
                pass
    return (torch.tensor(e25_np, dtype=torch.float32),
            torch.tensor(e41_np, dtype=torch.float32))


# ── Placer ──────────────────────────────────────────────────────────────────


class CDLNSSAHessianStreakedPlacer:
    """E80 — E79 + work-bounded streaks + dropped K-joint.

    Default hyperparameters target ≤ 60 min/bench on AMD EPYC 9655P.
    """

    def __init__(
        self,
        n_eigvecs: int = 1,
        eps_values: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        verbose: bool = True,
    ):
        # Note: eps_values now includes 0.1 from E77's finding (small ε
        # delivered the only saddle improvement on ibm12).
        self.n_eigvecs = n_eigvecs
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== CDLNSSAHessianStreakedPlacer ({benchmark.name}) ===")
        t0 = time.time()

        hw_scale = _probe_hardware_scale()
        scaled_polish = self.polish_budget * hw_scale
        log(f"  Hardware probe: scale={hw_scale:.3f}, "
            f"polish_budget {self.polish_budget:.0f}s → {scaled_polish:.0f}s")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phases 1+2: streaked E25 ‖ E41.
        try:
            e25, e41 = _run_parallel_e25_e41(benchmark.name, log)
        except Exception as exc:
            raise RuntimeError(
                f"E80 parallel E25⊥E41 failed: {exc}. "
                f"No fallback in E80 — investigate worker logs."
            ) from exc

        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
        log(f"  E25 proxy={e25_proxy:.5f}, E41 proxy={e41_proxy:.5f} "
            f"(parallel wall={time.time() - t0:.0f}s)")

        if e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  Plateau = {plateau_label} ({plateau_proxy:.5f})")

        # Phase 3: Hessian saddle (compressed; ε now includes 0.1 from E77).
        log(f"  Phase 3: Hessian saddle escape "
            f"(n_eigvecs={self.n_eigvecs}, eps={self.eps_values}, "
            f"polish={scaled_polish:.0f}s)")
        try:
            saddle_state, saddle_proxy = _saddle_escape(
                plateau, benchmark, plc,
                n_eigvecs=self.n_eigvecs,
                eps_values=self.eps_values,
                polish_budget=scaled_polish,
                log=log if self.verbose else None,
            )
        except Exception as exc:
            log(f"  Hessian saddle failed: {exc}; using plateau")
            saddle_state = plateau
            saddle_proxy = plateau_proxy

        candidates = [
            (e25_proxy, e25, "E25"),
            (e41_proxy, e41, "E41"),
            (saddle_proxy, saddle_state, "saddle"),
        ]
        candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = candidates[0]
        log(
            f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s"
        )

        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"E80 winner has {ovl} hard-macro overlaps"
            )
        return best_placement
