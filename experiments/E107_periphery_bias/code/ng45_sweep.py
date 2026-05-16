"""Sweep periphery push on all 4 NG45 Grand Prize benches.

For each bench: control, periphery push α=0.01, random kick (matched RMS), center push.
Measure Δproxy, Δedge, overlaps. Generalize signal from ariane133.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse helpers from decompose_spike
sys.path.insert(0, str(Path(__file__).parent))
from decompose_spike import (
    edge_dist_mean, push_periphery, push_center, push_random,
    polish_and_score, measure_rms,
)


def run_bench(bench_name, alpha=0.01, cd_budget_s=180.0):
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    placement_path = _ROOT / f"experiments/E91_dp_full_polish/results/{bench_name}_dp_full_polish.pt"
    if not placement_path.exists():
        # Fallback: try v4 variant
        placement_path = _ROOT / f"experiments/E91_dp_full_polish/results/{bench_name}_v4_dp_full_polish.pt"
    if not placement_path.exists():
        print(f"  [{bench_name}] NO CACHED PLACEMENT — skipping")
        return None

    data = torch.load(placement_path, map_location="cpu", weights_only=False)
    baseline = data["placement"].to(torch.float32) if isinstance(data, dict) else data.to(torch.float32)
    base_proxy = float(compute_proxy_cost(baseline, bench, plc)["proxy_cost"])
    base_edge = edge_dist_mean(baseline, cw, ch)

    print(f"\n=== {bench_name} (n={bench.num_macros}, canvas={cw:.0f}×{ch:.0f}) ===")
    print(f"baseline: proxy={base_proxy:.5f} edge_dist={base_edge:.4f}")

    periph = push_periphery(baseline, bench, alpha)
    periph_rms = measure_rms(periph, baseline)
    print(f"α={alpha} RMS={periph_rms:.4f}")

    cases = [
        ("Control", baseline.clone()),
        ("Periphery", periph),
        ("Random", push_random(baseline, bench, periph_rms, seed=42)),
        ("Center", push_center(baseline, bench, alpha)),
    ]

    rows = []
    print(f"{'case':>10} {'proxy':>9} {'Δprx%':>7} {'edge':>7} {'Δedg%':>7} {'ovl':>4} {'wall':>5}")
    for name, perturbed in cases:
        t0 = time.time()
        polished = polish_and_score(perturbed, bench, plc, cd_budget_s=cd_budget_s)
        p = float(compute_proxy_cost(polished, bench, plc)["proxy_cost"])
        e = edge_dist_mean(polished, cw, ch)
        ovl = compute_overlap_metrics(polished, bench)["overlap_count"]
        dp = (p - base_proxy) / base_proxy * 100
        de = (e - base_edge) / base_edge * 100
        wall = time.time() - t0
        print(f"{name:>10} {p:9.5f} {dp:+6.2f}% {e:7.4f} {de:+6.2f}% {ovl:4d} {wall:5.0f}s")
        rows.append((name, p, dp, e, de, ovl, wall))
    return {"bench": bench_name, "baseline_proxy": base_proxy, "baseline_edge": base_edge, "rows": rows}


def main():
    import os
    bench_arg = os.environ.get("BENCHES", "ibm09,ibm10,ibm12,ibm14,ibm17,ariane133")
    benches = bench_arg.split(",")
    all_results = []
    for b in benches:
        try:
            r = run_bench(b, alpha=0.01)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"  [{b}] FAILED: {e}")

    print("\n\n=== ng45 PERIPHERY SUMMARY (α=0.01) ===")
    print(f"{'bench':>16} {'control':>9} {'periphery':>9} {'random':>9} {'center':>9}  -- Δproxy% --")
    for r in all_results:
        ctrl = r["rows"][0][2]
        per = r["rows"][1][2]
        rnd = r["rows"][2][2]
        cen = r["rows"][3][2]
        print(f"{r['bench']:>16} {ctrl:+8.2f}% {per:+8.2f}% {rnd:+8.2f}% {cen:+8.2f}%")

    out = _ROOT / "experiments" / "E107_periphery_bias" / "results" / "ng45_sweep_alpha001.pt"
    torch.save(all_results, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
