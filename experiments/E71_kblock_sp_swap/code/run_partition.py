"""Driver for partition-crossover search."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from partition_crossover_placer import partition_crossover_search, cd_polish

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--max-trials", type=int, default=100)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--polish-per-trial", type=float, default=60.0)
    ap.add_argument("--final-polish", type=float, default=600.0)
    ap.add_argument("--grids", type=str, default="2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    grid_sizes = tuple(int(x) for x in args.grids.split(","))
    bench_name = args.bench
    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25 = torch.load(e69 / f"e25_{bench_name}.pt", weights_only=False)
    e41 = torch.load(e69 / f"e41_{bench_name}.pt", weights_only=False)

    print(f"[E71v2] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    if e25["proxy"] <= e41["proxy"]:
        state, other = e25, e41; sl, ol = "E25", "E41"
    else:
        state, other = e41, e25; sl, ol = "E41", "E25"
    print(f"[E71v2] start={sl} ({state['proxy']:.5f}), other={ol} ({other['proxy']:.5f})")
    print(f"[E71v2] grid_sizes={grid_sizes} max_trials={args.max_trials} budget={args.budget}s")

    t0 = time.time()
    result, stats = partition_crossover_search(
        state["placement"], other["placement"], benchmark, plc,
        grid_sizes=grid_sizes,
        max_trials=args.max_trials,
        polish_per_trial=args.polish_per_trial,
        budget_seconds=args.budget,
        seed=args.seed,
    )
    crossover_wall = time.time() - t0

    t1 = time.time()
    polished = cd_polish(result, benchmark, plc, budget_seconds=args.final_polish)
    polish_wall = time.time() - t1
    polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]

    out = {
        "bench_name": bench_name,
        "start_basin": sl,
        "start_proxy": float(state["proxy"]),
        "other_basin_proxy": float(other["proxy"]),
        "stats": {k: v for k, v in stats.items() if k != "proxy_history"},
        "crossover_wall_seconds": crossover_wall,
        "post_search_proxy": stats["final_proxy"],
        "polished_proxy": polished_proxy,
        "polished_overlap": polished_overlap,
        "polish_wall_seconds": polish_wall,
        "best_proxy": min(
            float(state["proxy"]), float(other["proxy"]),
            stats["final_proxy"], polished_proxy,
        ),
        "args": vars(args),
    }
    out_summary = _HERE.parent / "results" / f"partition_{bench_name}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(out, indent=2))
    out_pt = _HERE.parent / "results" / f"partition_{bench_name}.pt"
    torch.save({"placement": polished.cpu(), "stats": stats, "bench_name": bench_name}, out_pt)
    print(f"[E71v2] saved -> {out_summary}, {out_pt}")
    print(f"  start ({sl}): {state['proxy']:.5f}")
    print(f"  other ({ol}): {other['proxy']:.5f}")
    print(f"  post-search:     {stats['final_proxy']:.5f}")
    print(f"  post-polish:     {polished_proxy:.5f} overlap={polished_overlap}")
    print(f"  best:            {out['best_proxy']:.5f}")


if __name__ == "__main__":
    main()
