"""E115 — verify FastDiffProxy gives same final placement quality as DiffProxyV3.

Runs both V3 and V4 with identical seed/config on multiple benchmarks. Checks:
  - Final canonical proxy cost (must be within 0.5%)
  - Final HPWL, density, congestion components
  - Zero overlaps preserved
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
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from smooth_global_placer_v4 import SmoothGlobalPlacerV4


def main(benches, device, num_steps):
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
        log_every=10**6,  # silent
    )

    rows = []
    for bench_name in benches:
        print(f"\n=== {bench_name} ===", flush=True)
        bench_dir = find_benchmark_dir(bench_name)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))

        print(f"  V4 (fast)...", flush=True)
        p4 = SmoothGlobalPlacerV4(device=device, **common)
        t0 = time.perf_counter()
        pos4 = p4.place(benchmark)
        wall4 = time.perf_counter() - t0
        c4 = compute_proxy_cost(pos4, benchmark, plc)
        ov4 = compute_overlap_metrics(pos4, benchmark)["overlap_count"]

        # re-load to get a fresh plc (since plc may have internal state)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        print(f"  V3 (baseline)...", flush=True)
        p3 = SmoothGlobalPlacerV3(device=device, **common)
        t0 = time.perf_counter()
        pos3 = p3.place(benchmark)
        wall3 = time.perf_counter() - t0
        c3 = compute_proxy_cost(pos3, benchmark, plc)
        ov3 = compute_overlap_metrics(pos3, benchmark)["overlap_count"]

        row = {
            "bench": bench_name,
            "v3_proxy": float(c3["proxy_cost"]),
            "v4_proxy": float(c4["proxy_cost"]),
            "v3_overlap": int(ov3),
            "v4_overlap": int(ov4),
            "v3_wl": float(c3.get("wirelength_cost", 0)),
            "v4_wl": float(c4.get("wirelength_cost", 0)),
            "v3_dens": float(c3.get("density_cost", 0)),
            "v4_dens": float(c4.get("density_cost", 0)),
            "v3_cong": float(c3.get("congestion_cost", 0)),
            "v4_cong": float(c4.get("congestion_cost", 0)),
            "v3_wall_s": wall3,
            "v4_wall_s": wall4,
            "speedup_total": wall3 / wall4,
            "proxy_diff_pct": (float(c4["proxy_cost"]) - float(c3["proxy_cost"])) / max(abs(float(c3["proxy_cost"])), 1e-9) * 100,
        }
        rows.append(row)
        print(f"  V3: proxy={row['v3_proxy']:.5f} ovl={row['v3_overlap']} wall={row['v3_wall_s']:.1f}s", flush=True)
        print(f"  V4: proxy={row['v4_proxy']:.5f} ovl={row['v4_overlap']} wall={row['v4_wall_s']:.1f}s", flush=True)
        print(f"  diff: {row['proxy_diff_pct']:+.3f}%  speedup: {row['speedup_total']:.2f}x", flush=True)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  {'bench':<10s} {'V3 proxy':>10s} {'V4 proxy':>10s} {'diff%':>8s} {'V3 wall':>9s} {'V4 wall':>9s} {'speedup':>8s}", flush=True)
    for r in rows:
        print(f"  {r['bench']:<10s} {r['v3_proxy']:10.5f} {r['v4_proxy']:10.5f} {r['proxy_diff_pct']:+8.3f} {r['v3_wall_s']:9.1f} {r['v4_wall_s']:9.1f} {r['speedup_total']:8.2f}", flush=True)

    avg_speedup = sum(r["speedup_total"] for r in rows) / len(rows)
    avg_diff = sum(r["proxy_diff_pct"] for r in rows) / len(rows)
    print(f"\n  avg speedup: {avg_speedup:.2f}x  avg diff: {avg_diff:+.3f}%", flush=True)

    out_path = _HERE.parent / "results" / f"verify_quality_{device}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "avg_speedup": avg_speedup, "avg_diff_pct": avg_diff}, f, indent=2)
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    benches = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ibm01", "ibm07", "ibm10", "ibm17"]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    main(benches, device, num_steps)
