"""Profile Option C on ibm01 at small budget to identify bottlenecks.

What we need to know:
1. Where does wall time go inside cascade pipeline? (per-phase contribution)
2. Within SA-v2, where does it go? (Metropolis loop, move generation, eval, revert)
3. Within cascade saddle escape, what dominates?
4. Per-component cost trajectory: at each phase, what is WL/density/congestion?

Outputs cProfile data + an objective-decomposition trajectory log.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

import importlib.util


def main():
    bench = "ibm01"
    budget = 300.0  # short budget to keep profiling tractable

    print(f"Profiling Option C on {bench} at {budget}s budget...")
    print(f"=" * 72)

    spec = importlib.util.spec_from_file_location(
        "p", str(_ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer = mod.CDLNSSACascadeStackedPeripheryPlacer(
        budget_seconds=budget, verbose=True,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.time()
    placement = placer.place(benchmark)
    wall = time.time() - t0
    profiler.disable()

    proxy_breakdown = compute_proxy_cost(placement, benchmark, plc)
    overlaps = compute_overlap_metrics(placement, benchmark)
    print(f"\n=== Final on {bench}: ===")
    print(f"  proxy={float(proxy_breakdown['proxy_cost']):.5f}")
    print(f"  wl={float(proxy_breakdown.get('wl', 0)):.5f}")
    print(f"  density={float(proxy_breakdown.get('density', 0)):.5f}")
    print(f"  congestion={float(proxy_breakdown.get('congestion', 0)):.5f}")
    print(f"  overlaps={int(overlaps['overlap_count'])}")
    print(f"  wall={wall:.1f}s")

    # Profile output: top 30 by cumulative time.
    print(f"\n{'='*72}")
    print(f"Top 30 functions by CUMULATIVE TIME:")
    print(f"{'='*72}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    print(f"\n{'='*72}")
    print(f"Top 30 functions by SELF TIME (excluding subcalls):")
    print(f"{'='*72}")
    s = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    ps2.print_stats(30)
    print(s.getvalue())

    # Dump for later analysis.
    out_path = _ROOT / "experiments" / "E110_lp_dual_congestion" / "profile_optc_ibm01.prof"
    profiler.dump_stats(str(out_path))
    print(f"Saved cProfile data to {out_path}")


if __name__ == "__main__":
    main()
