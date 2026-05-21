"""Test descent-only on ibm17 for both grid_density (V3) and eDensity.

Compares raw post-legalize proxy from each.
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
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from diff_proxy_v3_edensity import SmoothGlobalPlacerV3eDensity


def run_one(placer_class, label, edensity_kwargs=None, **placer_kwargs):
    print(f"\n=== {label} ===")
    t0 = time.time()
    if edensity_kwargs is not None:
        placer = placer_class(
            num_steps=500,
            lr_frac=0.005,
            gamma_start_frac=5e-3,
            gamma_end_frac=5e-5,
            overlap_lambda_end=10.0,
            overlap_ramp_pct=0.7,
            init="sdf",
            rng_seed=42,
            verbose=True,
            log_every=100,
            edensity_kwargs=edensity_kwargs,
            **placer_kwargs,
        )
    else:
        placer = placer_class(
            num_steps=500,
            lr_frac=0.005,
            gamma_start_frac=5e-3,
            gamma_end_frac=5e-5,
            overlap_lambda_end=10.0,
            overlap_ramp_pct=0.7,
            init="sdf",
            rng_seed=42,
            verbose=True,
            log_every=100,
            **placer_kwargs,
        )
    bench_name = "ibm17"
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    pos = placer.place(benchmark)
    wall = time.time() - t0
    canon = compute_proxy_cost(pos, benchmark, plc)
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(
        f"  RESULT {label}: proxy={canon['proxy_cost']:.5f} "
        f"wl={canon['wirelength_cost']:.4f} d={canon['density_cost']:.4f} "
        f"c={canon['congestion_cost']:.4f} ovl={ovl} wall={wall:.0f}s"
    )
    return {
        "label": label,
        "proxy": canon["proxy_cost"],
        "wl": canon["wirelength_cost"],
        "d": canon["density_cost"],
        "c": canon["congestion_cost"],
        "ovl": ovl,
        "wall": wall,
        "pos": pos,
    }


if __name__ == "__main__":
    results = []

    # Baseline: grid-bin V3
    results.append(run_one(
        SmoothGlobalPlacerV3,
        "V3 grid-density (baseline)",
    ))

    # eDensity: scale 0.10, target 1.0
    results.append(run_one(
        SmoothGlobalPlacerV3eDensity,
        "V3eDensity scale=0.10 target=1.0",
        edensity_kwargs=dict(scale=0.10, target_density=1.0, sigma_frac=0.5),
    ))

    # eDensity: scale 0.05, target 1.0 (gentler)
    results.append(run_one(
        SmoothGlobalPlacerV3eDensity,
        "V3eDensity scale=0.05 target=1.0",
        edensity_kwargs=dict(scale=0.05, target_density=1.0, sigma_frac=0.5),
    ))

    # eDensity: scale 0.20, target 1.0 (stronger)
    results.append(run_one(
        SmoothGlobalPlacerV3eDensity,
        "V3eDensity scale=0.20 target=1.0",
        edensity_kwargs=dict(scale=0.20, target_density=1.0, sigma_frac=0.5),
    ))

    print("\n\n=== SUMMARY (ibm17, descent only, no CD polish) ===")
    print(f"{'config':<45} {'proxy':>9} {'wl':>7} {'d':>7} {'c':>7} {'ovl':>4} {'wall':>5}")
    for r in results:
        print(f"{r['label']:<45} {r['proxy']:>9.5f} {r['wl']:>7.4f} "
              f"{r['d']:>7.4f} {r['c']:>7.4f} {r['ovl']:>4d} {r['wall']:>5.0f}s")
