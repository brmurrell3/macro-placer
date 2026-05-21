"""E117 — ibm17 head-to-head: V3 grid-density vs V3 Gaussian-density.

Descent-only (no CD polish) test on ibm17. Reports raw canonical proxy
after legalize for both variants. CD polish budget=0.

Run:
  uv run experiments/E117_gaussian_density/code/test_ibm17.py
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
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from diff_proxy_v3_gaussian_density import SmoothGlobalPlacerV3GaussianDensity


def run_descent(label, placer):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
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
        f"c={canon['congestion_cost']:.4f} ovl={ovl} wall={wall:.0f}s",
        flush=True,
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
        "benchmark": benchmark,
        "plc": plc,
    }


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


def main():
    results = []

    # Baseline: V3 grid-density
    results.append(
        run_descent(
            "V3 grid-density baseline",
            SmoothGlobalPlacerV3(**common_kwargs),
        )
    )

    # V3 Gaussian sigma=1.0
    results.append(
        run_descent(
            "V3 Gaussian sigma_scale=1.0 sigma_floor=0.5",
            SmoothGlobalPlacerV3GaussianDensity(
                **common_kwargs, sigma_scale=1.0, sigma_floor_frac=0.5,
            ),
        )
    )

    # V3 Gaussian sigma=0.5 (sharper, closer to canonical)
    results.append(
        run_descent(
            "V3 Gaussian sigma_scale=0.5 sigma_floor=0.5",
            SmoothGlobalPlacerV3GaussianDensity(
                **common_kwargs, sigma_scale=0.5, sigma_floor_frac=0.5,
            ),
        )
    )

    # V3 Gaussian sigma=1.5 (broader smear)
    results.append(
        run_descent(
            "V3 Gaussian sigma_scale=1.5 sigma_floor=1.0",
            SmoothGlobalPlacerV3GaussianDensity(
                **common_kwargs, sigma_scale=1.5, sigma_floor_frac=1.0,
            ),
        )
    )

    print("\n\n=== SUMMARY (ibm17, descent only, no CD polish) ===", flush=True)
    print(
        f"{'config':<50} {'proxy':>9} {'wl':>7} {'d':>7} {'c':>7} {'ovl':>4} {'wall':>5}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['label']:<50} {r['proxy']:>9.5f} {r['wl']:>7.4f} "
            f"{r['d']:>7.4f} {r['c']:>7.4f} {r['ovl']:>4d} {r['wall']:>5.0f}s",
            flush=True,
        )

    # Save positions for follow-up CD polish test
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r["pos"] is not None:
            safe = r["label"].replace(" ", "_").replace("=", "").replace("/", "_")
            torch.save(
                {
                    "positions": r["pos"],
                    "label": r["label"],
                    "proxy": r["proxy"],
                    "wall": r["wall"],
                },
                out_dir / f"ibm17_descent_{safe}.pt",
            )

    return results


if __name__ == "__main__":
    main()
