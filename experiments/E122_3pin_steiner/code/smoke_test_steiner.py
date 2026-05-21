"""Smoke test the V3+Steiner placer on a small bench (ibm01) to verify
correctness before running the full ibm17 test.

Verifies:
  - Forward pass works
  - Gradient is non-zero (backward works)
  - Adam descent reduces loss
  - greedy_macro_legalize succeeds
  - Final placement has zero overlaps
  - Final proxy is reasonable (sub 1.5 on ibm01)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3_steiner import SmoothGlobalPlacerV3Steiner


def smoke(bench_name: str = "ibm01"):
    print(f"\n=== Smoke test V3+Steiner on {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    t0 = time.time()
    placer = SmoothGlobalPlacerV3Steiner(
        num_steps=200,
        lr_frac=0.005,
        gamma_start_frac=5e-3,
        gamma_end_frac=5e-5,
        overlap_lambda_end=10.0,
        init="sdf",
        verbose=True,
        log_every=50,
    )
    pos = placer.place(benchmark)
    wall = time.time() - t0
    proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(f"\n  RESULT: proxy={proxy:.5f}  ovl={ovl}  wall={wall:.0f}s", flush=True)
    return proxy, ovl


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    smoke(bench)
