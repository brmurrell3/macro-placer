"""E83 CDLNSSAHessianClockPlacer — single algorithm with hard wall cap.

Same hyperparameters on every benchmark; the only "dispatch" is the wall
clock observed during execution.  Compliant with the competition rule that
forbids per-benchmark hardcoded logic.

Pipeline:
  1. Hardware probe (scales polish QUALITY, not total wall budget).
  2. Phase 1+2 — parallel E25 ⊥ E41 with reduced hard caps:
       CD 1500 s, LNS 360 s, SA 360 s, K-joint 0 s.
       Per-lane wall ≤ 37 min; parallel max wall ≤ 37 min.
  3. Phase 3 — Hessian saddle escape, configuration adapts to remaining time:
       remaining ≥ 15 min → full   (k=1, ε={0.3, 1.0, 3.0}, polish 180 s)
       remaining ≥  5 min → minimal (k=1, ε={1.0},          polish  60 s)
       otherwise          → skip   (return E25/E41 plateau as winner)
  4. Validate zero overlaps; return best placement.

Worst-case wall on M3 baseline: ~55 min.  Probe scales polish only, not the
TOTAL_HARD_CAP_S — so the algorithm guarantees ≤ 60 min regardless of probe.

Reference: experiments/E83_clock_aware/manifest.md
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

# Reuse Hessian saddle escape from E74.
_E74_PATH = _ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_for_e83", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape

_WORKER_SCRIPT = Path(__file__).parent / "_worker.py"


# ── Hardware probe (scales polish only, NOT total cap) ──────────────────────

def _probe_hardware_scale() -> float:
    """scale = wall / baseline; clamp [1.0, 1.5]; floor 1.0 (no reduction)."""
    A = np.random.randn(512, 512).astype(np.float32)
    t0 = time.time()
    for _ in range(50):
        B = A @ A.T
        A = B / (float(np.linalg.norm(B)) + 1e-9)
    wall = time.time() - t0
    M3_MAX_BASELINE_S = 1.2
    scale = wall / M3_MAX_BASELINE_S
    return float(np.clip(scale, 1.0, 1.5))


# ── Parallel E25 ⊥ E41 subprocess workers ──────────────────────────────────

def _launch_worker(
    mode: str,
    bench_name: str,
    cd_cap_s: float,
    lns_budget_s: float,
    sa_budget_s: float,
    kjoint_budget_s: float,
) -> Tuple[subprocess.Popen, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + existing if existing else "")
    proc = subprocess.Popen(
        [
            sys.executable, str(_WORKER_SCRIPT),
            mode, bench_name, str(_ROOT), tmp.name,
            str(cd_cap_s), str(lns_budget_s), str(sa_budget_s), str(kjoint_budget_s),
        ],
        env=env,
    )
    return proc, tmp.name


def _run_parallel_e25_e41(
    bench_name: str,
    cd_cap_s: float,
    lns_budget_s: float,
    sa_budget_s: float,
    kjoint_budget_s: float,
    log,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log(
        f"  Phase 1+2 — parallel E25 ⊥ E41 "
        f"(CD={cd_cap_s:.0f}s, LNS={lns_budget_s:.0f}s, "
        f"SA={sa_budget_s:.0f}s, KJoint={kjoint_budget_s:.0f}s)"
    )
    t0 = time.time()
    proc25, path25 = _launch_worker(
        "e25", bench_name, cd_cap_s, lns_budget_s, sa_budget_s, 0.0
    )
    proc41, path41 = _launch_worker(
        "e41", bench_name, cd_cap_s, lns_budget_s, sa_budget_s, kjoint_budget_s
    )
    rc25 = proc25.wait()
    rc41 = proc41.wait()
    elapsed = time.time() - t0
    log(f"  Phase 1+2 done in {elapsed:.0f}s (rc25={rc25}, rc41={rc41})")
    if rc25 != 0 or rc41 != 0:
        for p in (path25, path41):
            try:
                os.unlink(p)
            except OSError:
                pass
        raise RuntimeError(f"E83 worker failure: rc_e25={rc25}, rc_e41={rc41}")
    try:
        e25_np = np.load(path25)
        e41_np = np.load(path41)
    finally:
        for p in (path25, path41):
            try:
                os.unlink(p)
            except OSError:
                pass
    return (
        torch.tensor(e25_np, dtype=torch.float32),
        torch.tensor(e41_np, dtype=torch.float32),
    )


# ── Placer ──────────────────────────────────────────────────────────────────


class CDLNSSAHessianClockPlacer:
    """E83 — single algorithm, wall-clock-aware Hessian degradation.

    Same hyperparameters and code path on every benchmark.  Hessian phase
    adapts ONLY based on observed elapsed time (not benchmark properties).
    """

    # Hard wall cap — 5-min safety margin under 60-min judging cap.
    TOTAL_HARD_CAP_S: float = 55 * 60

    # Phase 1+2 budgets (per-lane).  Sum ≤ 2220 s = 37 min.
    CD_CAP_S: float = 1500.0      # 25 min (vs E79's 2400)
    LNS_BUDGET_S: float = 360.0   #  6 min (vs E79's 600)
    SA_BUDGET_S: float = 360.0    #  6 min (vs E79's 600)
    KJOINT_BUDGET_S: float = 0.0  # disabled (Mitigation #4)

    # Phase 3 (Hessian) thresholds — based on TIME REMAINING after phase 1+2.
    HESSIAN_FULL_THRESHOLD_S: float = 15 * 60   # ≥ 15 min → full
    HESSIAN_MIN_THRESHOLD_S: float = 5 * 60     # ≥ 5 min → minimal
    HESSIAN_FULL_POLISH_S: float = 180.0
    HESSIAN_FULL_EPS: Tuple[float, ...] = (0.3, 1.0, 3.0)
    HESSIAN_MIN_POLISH_S: float = 60.0
    HESSIAN_MIN_EPS: Tuple[float, ...] = (1.0,)

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== CDLNSSAHessianClockPlacer ({benchmark.name}) ===")
        t0 = time.time()

        hw_scale = _probe_hardware_scale()
        log(f"  Hardware probe: scale={hw_scale:.3f} "
            f"(scales polish only; total cap fixed at {self.TOTAL_HARD_CAP_S:.0f}s)")

        # Reload plc.
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1+2: parallel E25 ⊥ E41 with reduced budgets.
        e25, e41 = _run_parallel_e25_e41(
            benchmark.name,
            cd_cap_s=self.CD_CAP_S,
            lns_budget_s=self.LNS_BUDGET_S,
            sa_budget_s=self.SA_BUDGET_S,
            kjoint_budget_s=self.KJOINT_BUDGET_S,
            log=log,
        )

        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
        log(f"  E25 proxy={e25_proxy:.5f}, E41 proxy={e41_proxy:.5f} "
            f"(elapsed={time.time() - t0:.0f}s)")

        if e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  Plateau = {plateau_label} ({plateau_proxy:.5f})")

        # Phase 3: Hessian saddle escape — clock-aware configuration.
        elapsed = time.time() - t0
        remaining = self.TOTAL_HARD_CAP_S - elapsed
        log(f"  Time accounting: elapsed={elapsed:.0f}s, "
            f"remaining={remaining:.0f}s "
            f"(full ≥ {self.HESSIAN_FULL_THRESHOLD_S:.0f}s, "
            f"min ≥ {self.HESSIAN_MIN_THRESHOLD_S:.0f}s)")

        saddle_state = plateau
        saddle_proxy = plateau_proxy
        saddle_mode = "skip"

        if remaining >= self.HESSIAN_FULL_THRESHOLD_S:
            saddle_mode = "full"
            polish = self.HESSIAN_FULL_POLISH_S * hw_scale
            eps_values = self.HESSIAN_FULL_EPS
            log(f"  Phase 3 — Hessian saddle (FULL): "
                f"k=1, ε={eps_values}, polish={polish:.0f}s")
        elif remaining >= self.HESSIAN_MIN_THRESHOLD_S:
            saddle_mode = "minimal"
            polish = self.HESSIAN_MIN_POLISH_S * hw_scale
            eps_values = self.HESSIAN_MIN_EPS
            log(f"  Phase 3 — Hessian saddle (MINIMAL): "
                f"k=1, ε={eps_values}, polish={polish:.0f}s")
        else:
            log(f"  Phase 3 — SKIPPED ({remaining:.0f}s remaining < "
                f"{self.HESSIAN_MIN_THRESHOLD_S:.0f}s threshold)")
            polish = 0.0
            eps_values = ()

        if saddle_mode != "skip":
            try:
                saddle_state, saddle_proxy = _saddle_escape(
                    plateau, benchmark, plc,
                    n_eigvecs=1,
                    eps_values=eps_values,
                    polish_budget=polish,
                    log=log if self.verbose else None,
                )
            except Exception as exc:
                log(f"  Hessian saddle failed: {exc}; using plateau")
                saddle_state = plateau
                saddle_proxy = plateau_proxy

        # Best of (E25, E41, saddle).
        candidates = [
            (e25_proxy, e25, "E25"),
            (e41_proxy, e41, "E41"),
            (saddle_proxy, saddle_state, f"saddle({saddle_mode})"),
        ]
        candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = candidates[0]
        total_wall = time.time() - t0
        log(
            f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={total_wall:.0f}s (cap={self.TOTAL_HARD_CAP_S:.0f}s, "
            f"saddle={saddle_mode})"
        )

        # Hard-wall sanity check (should never fire if budgets correct).
        if total_wall > 60 * 60:
            log(f"  WARNING: total wall {total_wall:.0f}s exceeds 60-min cap")

        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"E83 winner has {ovl} hard-macro overlaps"
            )
        return best_placement
