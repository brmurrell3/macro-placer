"""Single eDensity descent run on ibm17 with CD polish.

Avoids re-running V3 baseline (already cached at 1.304 raw / 1.246 CD60s).
Compares eDensity raw and CD-polished against those numbers.

Two configs tested:
  - scale=0.10 (calibrated to grid_density magnitude at SDF init)
  - scale=0.05 (gentler)

Note: 4-min descent + 10-min CD polish per config = 28 min total.
"""
from __future__ import annotations

import json
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
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from diff_proxy_v3_edensity import SmoothGlobalPlacerV3eDensity


def run_eDensity(bench_name, scale, target_density, sigma_frac, cd_polish_s=600.0):
    print(f"\n=== eDensity scale={scale} target={target_density} sigma_frac={sigma_frac} ===")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    t0 = time.time()
    placer = SmoothGlobalPlacerV3eDensity(
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
        edensity_kwargs=dict(
            scale=scale,
            target_density=target_density,
            sigma_frac=sigma_frac,
        ),
    )
    pos = placer.place(benchmark)
    raw_wall = time.time() - t0
    raw_canon = compute_proxy_cost(pos, benchmark, plc)
    raw_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(
        f"  RAW: proxy={raw_canon['proxy_cost']:.5f} "
        f"wl={raw_canon['wirelength_cost']:.4f} "
        f"d={raw_canon['density_cost']:.4f} "
        f"c={raw_canon['congestion_cost']:.4f} "
        f"ovl={raw_ovl} wall={raw_wall:.0f}s"
    )

    # CD polish
    print(f"  Starting CD polish budget={cd_polish_s}s")
    t_cd = time.time()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=cd_polish_s * 0.5,
        hard_cap_s=cd_polish_s,
        patience=5,
        plateau_threshold=0.001,
        log_fn=None,
    )
    cd_wall = time.time() - t_cd
    final = evaluator.placement.detach().clone().to(torch.float32)
    cd_canon = compute_proxy_cost(final, benchmark, plc)
    cd_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
    print(
        f"  CD-polished: proxy={cd_canon['proxy_cost']:.5f} "
        f"wl={cd_canon['wirelength_cost']:.4f} "
        f"d={cd_canon['density_cost']:.4f} "
        f"c={cd_canon['congestion_cost']:.4f} "
        f"ovl={cd_ovl} CD_wall={cd_wall:.0f}s"
    )

    return {
        "scale": scale,
        "target_density": target_density,
        "sigma_frac": sigma_frac,
        "raw_proxy": float(raw_canon["proxy_cost"]),
        "raw_wl": float(raw_canon["wirelength_cost"]),
        "raw_d": float(raw_canon["density_cost"]),
        "raw_c": float(raw_canon["congestion_cost"]),
        "raw_ovl": int(raw_ovl),
        "raw_wall_s": raw_wall,
        "cd_proxy": float(cd_canon["proxy_cost"]),
        "cd_wl": float(cd_canon["wirelength_cost"]),
        "cd_d": float(cd_canon["density_cost"]),
        "cd_c": float(cd_canon["congestion_cost"]),
        "cd_ovl": int(cd_ovl),
        "cd_wall_s": cd_wall,
    }


if __name__ == "__main__":
    bench = "ibm17"
    results = []

    # Configuration 1: scale=0.10
    r = run_eDensity(bench, scale=0.10, target_density=1.0, sigma_frac=0.5)
    results.append(r)

    # Save what we have so far
    out = _ROOT / "experiments" / "E114_edensity" / "results" / "ibm17_edensity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Configuration 2: scale=0.05 (gentler)
    r = run_eDensity(bench, scale=0.05, target_density=1.0, sigma_frac=0.5)
    results.append(r)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== ibm17 SUMMARY ===")
    print(f"{'config':<35} {'raw':>9} {'CD600s':>9} {'wl':>7} {'d':>7} {'c':>7} {'ovl':>4}")
    print(f"  REFERENCE V3 ovl10 720s: raw=1.30396 CD60s=1.24601 (ibm17_smoke.json)")
    print(f"  CHAMPION V3Min ovl10 720s: CD600s=1.20032 (spec)")
    for r in results:
        cfg = f"sc={r['scale']:.2f}/td={r['target_density']:.1f}/sf={r['sigma_frac']:.1f}"
        print(f"{cfg:<35} {r['raw_proxy']:>9.5f} {r['cd_proxy']:>9.5f} "
              f"{r['cd_wl']:>7.4f} {r['cd_d']:>7.4f} {r['cd_c']:>7.4f} {r['cd_ovl']:>4d}")

    print(f"\n  Results saved to {out}")
