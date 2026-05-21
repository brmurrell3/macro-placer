"""Quick calibration smoke check on ibm17 — compare buggy E111 vs fixed E124.

This is read-only: just compute_congestion on the starting positions and
on cached cascade placement if available. Tells us calibration delta
without running the full Adam descent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion as TraceOrig
from per_net_trace_proxy_fixed import PerNetTraceCongestionFixed as TraceFixed


def main():
    for bench in ("ibm10", "ibm17"):
        print(f"\n=== {bench} ===", flush=True)
        bench_dir = find_benchmark_dir(bench)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        pos = benchmark.macro_positions.clone().float()

        orig = TraceOrig(benchmark, plc, device="cpu")
        fixed = TraceFixed(benchmark, plc, device="cpu")

        with torch.no_grad():
            c_orig = float(orig.compute_congestion(pos))
            c_fix = float(fixed.compute_congestion(pos))
        can = float(compute_proxy_cost(pos, benchmark, plc)["congestion_cost"])
        rel_orig = (c_orig - can) / max(abs(can), 1e-6) * 100
        rel_fix = (c_fix - can) / max(abs(can), 1e-6) * 100
        delta_fix_orig = (c_fix - c_orig) / max(abs(c_orig), 1e-9) * 100
        print(f"  canonical: {can:.5f}", flush=True)
        print(f"  orig    : {c_orig:.5f}  ({rel_orig:+.2f}% vs canonical)", flush=True)
        print(f"  fixed   : {c_fix:.5f}  ({rel_fix:+.2f}% vs canonical)", flush=True)
        print(f"  fix vs orig: {delta_fix_orig:+.3f}%", flush=True)


if __name__ == "__main__":
    main()
