"""Driver for K-block transplant search."""
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

from kblock_placer import kblock_transplant_search, cd_polish

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--max-trials", type=int, default=500)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--polish", type=float, default=600.0)
    ap.add_argument("--K", type=str, default="10,20,50",
                    help="comma-separated K values to sample")
    ap.add_argument("--density-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    K_values = tuple(int(x) for x in args.K.split(","))
    bench_name = args.bench
    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25 = torch.load(e69 / f"e25_{bench_name}.pt", weights_only=False)
    e41 = torch.load(e69 / f"e41_{bench_name}.pt", weights_only=False)

    print(f"[E71] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    if e25["proxy"] <= e41["proxy"]:
        state, other = e25, e41; sl, ol = "E25", "E41"
    else:
        state, other = e41, e25; sl, ol = "E41", "E25"
    print(f"[E71] start={sl} ({state['proxy']:.5f}), other={ol} ({other['proxy']:.5f})")
    print(f"[E71] K_values={K_values} max_trials={args.max_trials} budget={args.budget}s")

    t0 = time.time()
    result, stats = kblock_transplant_search(
        state["placement"], other["placement"], benchmark, plc,
        K_values=K_values,
        max_trials=args.max_trials,
        budget_seconds=args.budget,
        seed=args.seed,
        density_centered_frac=args.density_frac,
    )
    swap_wall = time.time() - t0

    t1 = time.time()
    polished = cd_polish(result, benchmark, plc, budget_seconds=args.polish)
    polish_wall = time.time() - t1
    polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]

    out = {
        "bench_name": bench_name,
        "start_basin": sl,
        "start_proxy": float(state["proxy"]),
        "other_basin_proxy": float(other["proxy"]),
        "transplant_stats": {k: v for k, v in stats.items() if k != "proxy_history"},
        "swap_wall_seconds": swap_wall,
        "post_transplant_proxy": stats["final_proxy"],
        "polished_proxy": polished_proxy,
        "polished_overlap": polished_overlap,
        "polish_wall_seconds": polish_wall,
        "best_proxy": min(
            float(state["proxy"]), float(other["proxy"]),
            stats["final_proxy"] or float("inf"), polished_proxy,
        ),
        "args": vars(args),
    }
    out_summary = _HERE.parent / "results" / f"kblock_{bench_name}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(out, indent=2))
    out_pt = _HERE.parent / "results" / f"kblock_{bench_name}.pt"
    torch.save({"transplanted": result.cpu(), "polished": polished.cpu(),
                "stats": stats, "bench_name": bench_name}, out_pt)
    print(f"[E71] saved -> {out_summary}, {out_pt}")
    print(f"  start ({sl}): {state['proxy']:.5f}")
    print(f"  other ({ol}): {other['proxy']:.5f}")
    print(f"  post-transplant: {stats['final_proxy']:.5f}")
    print(f"  post-polish:     {polished_proxy:.5f} overlap={polished_overlap}")
    print(f"  best:            {out['best_proxy']:.5f}")


if __name__ == "__main__":
    main()
