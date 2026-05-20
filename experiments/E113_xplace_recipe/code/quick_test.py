"""E113 — Quick 100-step debug run of V4 on ibm01.

Catches obvious crashes / NaN issues before running the full smoke.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE, _ROOT,
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4 import SmoothGlobalPlacerV4


def main():
    bench_dir = find_benchmark_dir("ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded ibm01: {benchmark.num_macros} macros", flush=True)

    placer = SmoothGlobalPlacerV4(
        num_steps=100, base_lr_frac=0.005, verbose=True, log_every=10,
        nesterov_use_bb=True,
    )
    t0 = time.time()
    pos = placer.place(benchmark)
    wall = time.time() - t0
    proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(f"\n[100-step debug] proxy={proxy:.5f} ovl={ovl} wall={wall:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
