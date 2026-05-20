"""Diagnose where E111's trace congestion mismatches canonical.

Strip components one at a time:
  1. Just net trace, no macro, no smoothing
  2. + macro
  3. + smoothing
And compare each to canonical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion


def diagnose(bench_name: str = "ibm17"):
    print(f"\n=== diagnose {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
    else:
        start = benchmark.macro_positions.clone().float()

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    print(f"  canonical:                 {can_cong:.5f}", flush=True)

    cases = [
        ("net only, no smooth",   {"include_macro_routing": False, "include_smoothing": False}),
        ("net only, +smooth",     {"include_macro_routing": False, "include_smoothing": True}),
        ("net + macro, no smooth",{"include_macro_routing": True,  "include_smoothing": False}),
        ("net + macro + smooth",  {"include_macro_routing": True,  "include_smoothing": True}),
    ]
    for label, kw in cases:
        p = PerNetTraceCongestion(benchmark, plc, device="cpu", **kw)
        with torch.no_grad():
            v = float(p.compute_congestion(start))
        print(f"  {label:28s} {v:.5f}  ({(v-can_cong)/can_cong*100:+.1f}%)", flush=True)


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    for b in benches:
        diagnose(b)
