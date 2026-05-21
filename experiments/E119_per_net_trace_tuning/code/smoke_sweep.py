"""E119 smoke — quick sweep on ibm10 only to validate the script.

Subset of `sweep_hparams.py` for fast smoke test:
  - bench = [ibm10] only (smaller than ibm17)
  - grid = 9 configs around the default
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
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion


def main():
    bench = "ibm10"
    print(f"Loading {bench} ...", flush=True)
    t0 = time.time()
    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
    else:
        start = benchmark.macro_positions.clone().float()
    can_full = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can_full["congestion_cost"])
    print(f"  canonical_cong={can_cong:.5f} wall={time.time()-t0:.1f}s", flush=True)

    print(f"Building proxy ...", flush=True)
    t1 = time.time()
    proxy = PerNetTraceCongestion(benchmark, plc, device="cpu")
    print(f"  n_pairs={proxy.n_pairs} wall={time.time()-t1:.1f}s", flush=True)

    configs = [
        # default
        (2, 0.5, 6, 4),
        # smooth_range sweep
        (1, 0.5, 6, 4),
        (3, 0.5, 6, 4),
        # sigma_frac sweep
        (2, 0.3, 6, 4),
        (2, 0.7, 6, 4),
        # beta_minmax sweep
        (2, 0.5, 4, 4),
        (2, 0.5, 10, 4),
        # beta_range sweep
        (2, 0.5, 6, 6),
        (2, 0.5, 6, 10),
    ]
    print(f"\n  {'sr':>2s} {'σ':>4s} {'β_mm':>4s} {'β_rg':>4s}   {'smooth':>9s}  {'rel%':>6s}  {'wall':>4s}", flush=True)
    for sr, sigma, b_mm, b_rg in configs:
        t2 = time.time()
        proxy.smooth_range = sr
        proxy.sigma_cell_frac = sigma
        proxy.beta_minmax = b_mm
        proxy.beta_range_per_cell = b_rg
        with torch.no_grad():
            sm = float(proxy.compute_congestion(start))
        rel = (sm - can_cong) / max(abs(can_cong), 1e-9) * 100
        wall = time.time() - t2
        print(f"  {sr:>2d} {sigma:>4.1f} {b_mm:>4.1f} {b_rg:>4.1f}   {sm:>9.5f}  {rel:>+6.1f}%  {wall:>4.1f}s", flush=True)
    print(f"\nTotal wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
