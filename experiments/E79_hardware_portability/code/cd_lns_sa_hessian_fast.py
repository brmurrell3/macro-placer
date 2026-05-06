"""E79 CDLNSSAHessianFast — E74 + hardware portability optimizations.

Changes vs E74 (CDLNSSAHessian):
  1. E25 and E41 run in PARALLEL as subprocesses — saves ~25 min/bench.
     Wall = max(E25, E41) ≈ 30 min instead of E25 + E41 ≈ 55 min.
  2. Hessian phase COMPRESSED: n_eigvecs=1 (from 2), polish_budget=180s
     (from 240s) — saves ~30 min. ~80 % of lift comes from eig0.
  3. Hardware PROBE at startup scales the polish budget to hardware speed
     (matmul probe vs M3 Max baseline; clamp ∈ [0.5, 1.5]).

Expected wall on M3 Max: max(25, 30) + Hessian 18 + misc ≈ 50 min/bench.
Expected wall on AMD EPYC 9655P: ≈ 50 × 1.5 / probe_scale ≈ 55–60 min.
Within the 60-min judging cap.

Reference: experiments/E79_hardware_portability/manifest.md
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

# Monkey-patch macro_place.loader.load_benchmark to handle Windows backslash
# paths.  Upstream plc_client_os.py uses rsplit('/', -1)[-2] which crashes on
# Windows-style paths. We normalize at the loader entry point so every call
# chain (including sdf_init → _load_plc → load_benchmark_from_dir → load_benchmark)
# gets posix paths without modifying macro_place/loader.py itself.
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
from macro_place.loader import load_benchmark  # picks up patched module-level name
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse the Hessian saddle-escape machinery from E74.
_E74_PATH = _ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_placer_ref", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape

_WORKER_SCRIPT = Path(__file__).parent / "_worker.py"


def _load_plc_for_bench(bench_name: str):
    """Load (benchmark, plc) for a benchmark, normalizing Windows paths."""
    bench_dir = find_benchmark_dir(bench_name)
    netlist_file = str(Path(bench_dir) / "netlist.pb.txt").replace("\\", "/")
    plc_path = Path(bench_dir) / "initial.plc"
    plc_file = str(plc_path).replace("\\", "/") if plc_path.exists() else None
    return load_benchmark(netlist_file, plc_file)


# ── Hardware probe ──────────────────────────────────────────────────────────

def _probe_hardware_scale() -> float:
    """Estimate hardware speed vs M3 Max baseline.

    Runs 50 iterations of 512×512 float32 matmul.  Returns scale ∈ [0.5, 1.5]:
      >1.0 → hardware SLOWER than M3 Max → expand budgets so slow per-core
             EPYC (the judging target) still gets enough sweeps.
      <1.0 → hardware FASTER than M3 Max → contract budgets to avoid waste.

    Direction: scale = wall / baseline (higher for slower HW).  This matches
    the §Derisk intent in TODO.md: "If partcl is 2× slower per-core, all
    caps relax to 2× without changing the algorithm."

    Floor at 1.0 (not 0.5): the M3 Max polish budget is already tuned to
    deliver the lift; reducing it on faster hardware is pure proxy regression
    with no meaningful wall savings (Hessian phase is small fraction of total
    wall).  Empirical data: polish=90s on ibm01 gave proxy 0.882 vs polish=270s
    smoke 0.864 — a 2.1 % regression for ~13-min wall savings, bad trade.
    """
    A = np.random.randn(512, 512).astype(np.float32)
    t0 = time.time()
    for _ in range(50):
        B = A @ A.T
        A = B / (float(np.linalg.norm(B)) + 1e-9)
    wall = time.time() - t0
    M3_MAX_BASELINE_S = 1.2
    scale = wall / M3_MAX_BASELINE_S
    return float(np.clip(scale, 1.0, 1.5))


# ── Parallel E25 ⊥ E41 via subprocesses ────────────────────────────────────

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


def _run_parallel_e25_e41(
    bench_name: str, log,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log("  Phases 1+2: launching E25 ‖ E41 subprocesses in parallel...")
    t0 = time.time()
    proc25, path25 = _launch_worker("e25", bench_name)
    proc41, path41 = _launch_worker("e41", bench_name)
    rc25 = proc25.wait()
    rc41 = proc41.wait()
    elapsed = time.time() - t0
    log(f"  Parallel E25+E41 done in {elapsed:.0f}s (rc25={rc25}, rc41={rc41})")
    if rc25 != 0 or rc41 != 0:
        for p in (path25, path41):
            try:
                os.unlink(p)
            except OSError:
                pass
        raise RuntimeError(f"E79 worker failure: rc_e25={rc25}, rc_e41={rc41}")
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


class CDLNSSAHessianFastPlacer:
    """E79 — E74 with hardware portability: parallel E25⊥E41 + compressed Hessian.

    Key parameters vs E74:
      n_eigvecs: 1 (was 2) — ~80 % of lift comes from eig0
      polish_budget: 180.0 s (was 240.0 s)
      parallel: True — E25 and E41 run as parallel subprocesses

    Wall target: ≤50 min/bench on M3 Max, ≤60 min on AMD EPYC 9655P.
    """

    def __init__(
        self,
        n_eigvecs: int = 1,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        parallel: bool = True,
        verbose: bool = True,
    ):
        self.n_eigvecs = n_eigvecs
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.parallel = parallel
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== CDLNSSAHessianFastPlacer ({benchmark.name}) ===")
        t0 = time.time()

        hw_scale = _probe_hardware_scale()
        scaled_polish = self.polish_budget * hw_scale
        log(f"  Hardware probe: scale={hw_scale:.3f}, "
            f"polish_budget {self.polish_budget:.0f}s → {scaled_polish:.0f}s")

        # Reload plc with path normalization (avoid loader.py modification).
        _, plc = _load_plc_for_bench(benchmark.name)

        # Phases 1+2: E25 ‖ E41.
        if self.parallel:
            try:
                e25, e41 = _run_parallel_e25_e41(benchmark.name, log)
            except Exception as exc:
                log(f"  Parallel E25‖E41 failed ({exc}); falling back to sequential")
                self.parallel = False

        if not self.parallel:
            _E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
            _e25_spec = importlib.util.spec_from_file_location("e25_placer_seq", str(_E25_PATH))
            _e25_mod = importlib.util.module_from_spec(_e25_spec)
            _e25_spec.loader.exec_module(_e25_mod)
            log("  Phase 1: E25 (sequential)")
            e25 = _e25_mod.CDLNSSAPlacer().place(benchmark)
            log(f"  E25 done (wall={time.time() - t0:.0f}s)")

            from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer
            log("  Phase 2: E41 (sequential)")
            e41 = CDLNSSADPOKJointPlacer().place(benchmark)
            log(f"  E41 done (wall={time.time() - t0:.0f}s)")

        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
        log(f"  E25 proxy={e25_proxy:.5f}, E41 proxy={e41_proxy:.5f} "
            f"(wall={time.time() - t0:.0f}s)")

        if e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  E48 plateau = {plateau_label} ({plateau_proxy:.5f})")

        # Phase 3: Hessian saddle escape (compressed).
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
            log(f"  Hessian saddle escape failed: {exc}; using E48 plateau")
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
                f"E79 winner has {ovl} hard-macro overlaps — falling back to E25"
            )
        return best_placement
