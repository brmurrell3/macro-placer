"""E117 — ibm17 full comparison: V3 grid vs V3 Gaussian, both with CD polish.

Runs each variant end-to-end (V3 descent + greedy_macro_legalize + CD600s polish)
and reports the final canonical proxy.

Estimated wall: ~27 min (2 x (~200s descent + ~600s CD))
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
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from diff_proxy_v3_gaussian_density import SmoothGlobalPlacerV3GaussianDensity


def descent_and_polish(label, placer, benchmark, plc, cd_budget=600.0):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()

    # Phase 1: descent + greedy_macro_legalize (inside .place())
    pos = placer.place(benchmark)
    descent_wall = time.time() - t0
    descent_canon = compute_proxy_cost(pos, benchmark, plc)
    descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(
        f"  DESCENT {label}: proxy={descent_canon['proxy_cost']:.5f} "
        f"wl={descent_canon['wirelength_cost']:.4f} d={descent_canon['density_cost']:.4f} "
        f"c={descent_canon['congestion_cost']:.4f} ovl={descent_ovl} wall={descent_wall:.0f}s",
        flush=True,
    )

    if descent_ovl != 0:
        print(f"  WARNING: descent produced {descent_ovl} overlaps", flush=True)
        return {"label": label, "descent_proxy": descent_canon['proxy_cost'],
                "polished_proxy": None, "ovl": descent_ovl}

    # Phase 2: CD600s polish
    t_polish = time.time()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
    movable = [i for i in range(benchmark.num_macros)
               if not bool(benchmark.macro_fixed[i])]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=cd_budget * 0.5,
        hard_cap_s=cd_budget,
        patience=5,
        plateau_threshold=0.001,
        log_fn=None,
    )
    cd_wall = time.time() - t_polish

    final = evaluator.placement.detach().clone().to(torch.float32)
    final_canon = compute_proxy_cost(final, benchmark, plc)
    final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
    print(
        f"  POLISHED {label}: proxy={final_canon['proxy_cost']:.5f} "
        f"wl={final_canon['wirelength_cost']:.4f} d={final_canon['density_cost']:.4f} "
        f"c={final_canon['congestion_cost']:.4f} ovl={final_ovl} cd_wall={cd_wall:.0f}s "
        f"total_wall={time.time()-t0:.0f}s",
        flush=True,
    )
    return {
        "label": label,
        "descent_proxy": descent_canon['proxy_cost'],
        "descent_wl": descent_canon['wirelength_cost'],
        "descent_d": descent_canon['density_cost'],
        "descent_c": descent_canon['congestion_cost'],
        "polished_proxy": final_canon['proxy_cost'],
        "polished_wl": final_canon['wirelength_cost'],
        "polished_d": final_canon['density_cost'],
        "polished_c": final_canon['congestion_cost'],
        "ovl": final_ovl,
        "descent_wall": descent_wall,
        "cd_wall": cd_wall,
    }


def main():
    bench_name = "ibm17"
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"bench={benchmark.name} num_macros={benchmark.num_macros} num_hard={benchmark.num_hard_macros}",
          flush=True)

    common_kwargs = dict(
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
    )

    results = []

    # Baseline: V3 grid + CD600s
    placer = SmoothGlobalPlacerV3(**common_kwargs)
    results.append(descent_and_polish("V3 grid-density + CD600s", placer, benchmark, plc, cd_budget=600.0))

    # V3 Gaussian sigma=1.0 + CD600s
    placer = SmoothGlobalPlacerV3GaussianDensity(
        **common_kwargs, sigma_scale=1.0, sigma_floor_frac=0.5,
    )
    results.append(descent_and_polish("V3 Gaussian(1.0) + CD600s", placer, benchmark, plc, cd_budget=600.0))

    print("\n\n=== ibm17 FULL PIPELINE SUMMARY ===", flush=True)
    print(f"{'config':<28} {'descent':>9} {'polished':>9} {'wl':>7} {'d':>7} {'c':>7} {'ovl':>4}",
          flush=True)
    for r in results:
        if r['polished_proxy'] is not None:
            print(f"{r['label']:<28} {r['descent_proxy']:>9.5f} {r['polished_proxy']:>9.5f} "
                  f"{r['polished_wl']:>7.4f} {r['polished_d']:>7.4f} {r['polished_c']:>7.4f} "
                  f"{r['ovl']:>4d}", flush=True)
        else:
            print(f"{r['label']:<28} {r['descent_proxy']:>9.5f} {'N/A':>9} (ovl={r['ovl']})",
                  flush=True)


if __name__ == "__main__":
    main()
