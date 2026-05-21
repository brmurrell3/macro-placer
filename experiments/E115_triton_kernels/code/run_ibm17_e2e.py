"""E115 — end-to-end smoke test of FastDiffProxy on ibm17.

Runs V3 and V4 with identical hyperparameters, compares:
  - final proxy cost (canonical)
  - per-step wall time
  - total descent wall

Usage:
  .venv/bin/python experiments/E115_triton_kernels/code/run_ibm17_e2e.py [bench] [num_steps]
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from smooth_global_placer_v4 import SmoothGlobalPlacerV4


def main(bench_name: str = "ibm17", num_steps: int = 300):
    print(f"\n=== E115 e2e: {bench_name} ({num_steps} steps) ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    common = dict(
        num_steps=num_steps,
        lr_frac=0.005,
        gamma_start_frac=5e-3,
        gamma_end_frac=5e-4,
        overlap_lambda_start=0.0,
        overlap_lambda_end=50.0,
        overlap_ramp_pct=0.7,
        boundary_lambda=50.0,
        include_congestion=True,
        init="sdf",
        rng_seed=42,
        verbose=False,
        log_every=50,
    )

    # V4 first (fastest)
    print(f"\n--- V4 (FastDiffProxy) ---", flush=True)
    placer_v4 = SmoothGlobalPlacerV4(device="cuda", **common)
    t0 = time.perf_counter()
    pos_v4 = placer_v4.place(benchmark)
    wall_v4 = time.perf_counter() - t0
    cost_v4 = float(compute_proxy_cost(pos_v4, benchmark, plc)["proxy_cost"])
    print(f"  v4: proxy={cost_v4:.5f} wall={wall_v4:.1f}s", flush=True)

    # V3 baseline
    print(f"\n--- V3 (DiffProxyV3 baseline) ---", flush=True)
    placer_v3 = SmoothGlobalPlacerV3(device="cuda", **common)
    t0 = time.perf_counter()
    pos_v3 = placer_v3.place(benchmark)
    wall_v3 = time.perf_counter() - t0
    cost_v3 = float(compute_proxy_cost(pos_v3, benchmark, plc)["proxy_cost"])
    print(f"  v3: proxy={cost_v3:.5f} wall={wall_v3:.1f}s", flush=True)

    speedup_total = wall_v3 / wall_v4
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  V3:        proxy={cost_v3:.5f}  wall={wall_v3:.1f}s", flush=True)
    print(f"  V4:        proxy={cost_v4:.5f}  wall={wall_v4:.1f}s", flush=True)
    print(f"  speedup:   {speedup_total:.2f}x", flush=True)
    print(f"  cost diff: {(cost_v4 - cost_v3) / max(abs(cost_v3), 1e-9) * 100:+.3f}%", flush=True)

    results = {
        "bench": bench_name,
        "num_steps": num_steps,
        "v3_proxy": cost_v3,
        "v4_proxy": cost_v4,
        "v3_wall_s": wall_v3,
        "v4_wall_s": wall_v4,
        "speedup_total": speedup_total,
        "cost_diff_pct": (cost_v4 - cost_v3) / max(abs(cost_v3), 1e-9) * 100,
    }
    out_path = _HERE.parent / "results" / f"e2e_{bench_name}_n{num_steps}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(bench, n)
