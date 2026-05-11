"""Run E61 V2 (CDLNSGACrossoverPlacer) fresh on a benchmark and save the output.

This re-runs E25 + E41 + crossover + polish from scratch INSIDE the placer
(unlike my E72 which used cached E25/E41 — that approach was falsified
because cached pairs are not crossover-compatible).

Usage:
  uv run python experiments/E75_fresh_e61v2_wave/code/run_fresh_e61v2.py <bench>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.E61_ga_crossover.code.cd_lns_ga_crossover import CDLNSGACrossoverPlacer

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    bench_name = sys.argv[1]
    out_pt = _HERE.parent / "results" / f"e61v2_fresh_{bench_name}.pt"
    out_json = _HERE.parent / "results" / f"e61v2_fresh_{bench_name}.json"

    print(f"[E75] loading {bench_name}...")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    print(f"[E75] running E61 V2 (CDLNSGACrossoverPlacer) FRESH on {bench_name}...")
    placer = CDLNSGACrossoverPlacer()
    t0 = time.time()
    placement = placer.place(benchmark)
    wall = time.time() - t0

    metrics = compute_proxy_cost(placement, benchmark, plc)
    overlap = metrics["overlap_count"]
    proxy = metrics["proxy_cost"]
    print(
        f"[E75] DONE E61V2/{bench_name}: proxy={proxy:.5f} overlap={overlap} "
        f"wall={wall:.0f}s"
    )

    torch.save(
        {
            "placement": placement.detach().cpu(),
            "proxy": float(proxy),
            "overlap_count": int(overlap),
            "wall_seconds": float(wall),
            "bench_name": bench_name,
            "macro_sizes": benchmark.macro_sizes.detach().cpu(),
            "num_hard_macros": int(benchmark.num_hard_macros),
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
        },
        out_pt,
    )
    out_json.write_text(json.dumps({
        "bench_name": bench_name,
        "proxy": float(proxy),
        "overlap_count": int(overlap),
        "wall_seconds": float(wall),
    }, indent=2))
    print(f"[E75] saved -> {out_pt}, {out_json}")


if __name__ == "__main__":
    main()
