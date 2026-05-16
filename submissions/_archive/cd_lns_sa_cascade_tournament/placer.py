"""CDLNSSACascadeTournamentPlacer — best-of-3 over DP-lane + dual_levy + adaptive.

Tournament approach: runs three full placers in parallel subprocesses, each
with its own budget. Returns the placement with lowest canonical proxy
(zero overlaps required).

Rule compliance: same three placers on every benchmark — not per-bench
hardcoded. The "algorithm" is "run these three independent strategies
in parallel and pick the best."

Expected aggregate (from per-bench best-of-N analysis 2026-05-14):
  IBM avg ≈ 1.060 (vs shipped 1.078; vs best single placer DP-lane 1.067)
  NG45 avg ≈ 0.677

Wall budget per bench: 3 placers in parallel, each ~50 min, total wall
~50 min (dominated by slowest). Within 60-min cap.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


# Each "lane" = a subprocess command that loads a placer module and runs
# its place() method. We pass the bench_name and let each lane re-load
# the benchmark (avoids pickling Benchmark across processes).

LANES = [
    # (label, placer module path relative to repo root)
    ("dp_lane", "submissions/cd_lns_sa_cascade_dp_lane/placer.py"),
    ("dual_levy", "submissions/cd_lns_sa_cascade_dual_levy/placer.py"),
    ("adaptive", "submissions/cd_lns_sa_cascade/placer_adaptive.py"),
]


_LANE_RUNNER = Path(__file__).resolve().parent / "_lane_runner.py"


def _run_one_lane(bench_name: str, placer_path: str, label: str,
                  out_pkl: str, budget_seconds: float) -> dict:
    """Run a single placer in a subprocess and write the placement to a pickle.

    Returns metadata dict with proxy, overlap_count, wall.
    """
    cmd = [
        sys.executable, str(_LANE_RUNNER),
        bench_name, placer_path, label, out_pkl, str(budget_seconds),
    ]
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=budget_seconds + 900,
        )
    except subprocess.TimeoutExpired:
        return {"label": label, "error": "timeout", "wall_seconds": time.time() - t0}
    wall = time.time() - t0
    if proc.returncode != 0:
        return {
            "label": label,
            "error": f"exit={proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-2000:],
            "wall_seconds": wall,
        }
    if not Path(out_pkl).exists():
        return {"label": label, "error": "no output pkl", "wall_seconds": wall}
    with open(out_pkl, "rb") as f:
        result = pickle.load(f)
    return result


class CDLNSSACascadeTournamentPlacer:
    """Tournament: 3 placers in parallel, pick best."""

    def __init__(self, budget_seconds: float = 2700.0, verbose: bool = True):
        # Per-lane budget. Total wall ≈ this (dominated by slowest lane).
        # 2700s = 45 min. With 60-min outer cap that's plenty of margin.
        self.budget_seconds = float(budget_seconds)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._log(f"=== CDLNSSACascadeTournamentPlacer ({benchmark.name}) ===")
        self._log(f"  budget per lane: {self.budget_seconds:.0f}s, "
                  f"running {len(LANES)} lanes in parallel")

        with tempfile.TemporaryDirectory(prefix=f"tournament_{benchmark.name}_") as tmp:
            tmp_path = Path(tmp)
            results = []

            # Launch all lanes as subprocesses in parallel (raw Popen).
            procs = []
            for label, placer_path in LANES:
                pkl = str(tmp_path / f"{label}.pkl")
                stderr_log = tmp_path / f"{label}.stderr"
                stdout_log = tmp_path / f"{label}.stdout"
                cmd = [
                    sys.executable, str(_LANE_RUNNER),
                    benchmark.name, placer_path, label, pkl,
                    str(self.budget_seconds),
                ]
                env = dict(os.environ)
                env["OPENBLAS_NUM_THREADS"] = "4"
                env["OMP_NUM_THREADS"] = "4"
                env["MKL_NUM_THREADS"] = "4"
                stdout_f = open(stdout_log, "w")
                stderr_f = open(stderr_log, "w")
                p = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=stderr_f)
                procs.append({
                    "label": label, "pkl": pkl, "proc": p,
                    "stdout_log": str(stdout_log), "stderr_log": str(stderr_log),
                    "stdout_f": stdout_f, "stderr_f": stderr_f,
                    "t0": time.time(),
                })

            # Wait for all (with global timeout)
            global_deadline = time.time() + self.budget_seconds + 900
            for p in procs:
                remaining = max(1, int(global_deadline - time.time()))
                try:
                    p["proc"].wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    p["proc"].kill()
                    p["proc"].wait()
                p["stdout_f"].close()
                p["stderr_f"].close()

            for p in procs:
                wall = time.time() - p["t0"]
                if p["proc"].returncode != 0:
                    stderr_tail = ""
                    try:
                        with open(p["stderr_log"]) as f:
                            stderr_tail = f.read()[-2000:]
                    except Exception:
                        pass
                    results.append({
                        "label": p["label"],
                        "error": f"exit={p['proc'].returncode}",
                        "stderr_tail": stderr_tail,
                        "wall_seconds": wall,
                    })
                    self._log(f"  [lane {p['label']}] ERROR exit={p['proc'].returncode}")
                    continue
                if not Path(p["pkl"]).exists():
                    results.append({
                        "label": p["label"],
                        "error": "no output pkl",
                        "wall_seconds": wall,
                    })
                    self._log(f"  [lane {p['label']}] ERROR no pkl")
                    continue
                with open(p["pkl"], "rb") as f:
                    result = pickle.load(f)
                results.append(result)
                self._log(f"  [lane {p['label']}] proxy={result['proxy_cost']:.5f} "
                          f"ovl={result['overlap_count']} wall={result['wall_seconds']:.0f}s")

        # Pick best (lowest proxy, zero overlaps required)
        valid = [r for r in results if "error" not in r and r.get("overlap_count", 1) == 0]
        if not valid:
            # Fallback: pick lowest proxy regardless of overlap (will likely fail validation)
            valid = [r for r in results if "error" not in r]
        if not valid:
            raise RuntimeError("Tournament: all lanes failed")

        winner = min(valid, key=lambda r: r["proxy_cost"])
        self._log(f"  WINNER: {winner['label']} proxy={winner['proxy_cost']:.5f} "
                  f"(beats {len(valid)-1} others)")
        return winner["placement"]
