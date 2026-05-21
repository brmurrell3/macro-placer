"""E117 — ibm17 Gaussian descent test (single variant).

Just V3 Gaussian sigma=1.0, descent-only, with proxy reported by the
placer itself (no redundant compute_proxy_cost call). Compare against
V3 grid-density baseline = 1.28548 raw (from prior test).
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

from diff_proxy_v3_gaussian_density import SmoothGlobalPlacerV3GaussianDensity


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

    # V3 Gaussian sigma=1.0 — placer prints its own final canonical proxy
    print(f"\n=== V3 Gaussian sigma_scale=1.0 ===", flush=True)
    placer = SmoothGlobalPlacerV3GaussianDensity(
        **common_kwargs, sigma_scale=1.0, sigma_floor_frac=0.5,
    )
    pos = placer.place(benchmark)

    # Save positions for follow-up CD polish
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"positions": pos}, out_dir / "ibm17_descent_gaussian_s1_0.pt")
    print(f"saved positions to {out_dir / 'ibm17_descent_gaussian_s1_0.pt'}", flush=True)


if __name__ == "__main__":
    main()
