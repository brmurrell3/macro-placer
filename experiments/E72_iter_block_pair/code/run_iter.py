"""Driver: run iterative block+pair refinement on a benchmark.

Loads cached E25 + E41, runs iter_block_pair_search, saves placement
+ JSON summary.
"""
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

from iter_block_pair_placer import iter_block_pair_search

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--n-iters", type=int, default=4)
    ap.add_argument("--block-polish", type=float, default=300.0)
    ap.add_argument("--pair-attempts", type=int, default=300)
    ap.add_argument("--pair-budget", type=float, default=600.0)
    ap.add_argument("--pair-polish", type=float, default=300.0)
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    bench_name = args.bench
    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25 = torch.load(e69 / f"e25_{bench_name}.pt", weights_only=False)
    e41 = torch.load(e69 / f"e41_{bench_name}.pt", weights_only=False)

    print(f"[E72] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    print(f"  E25 cached proxy {e25['proxy']:.5f} | E41 cached proxy {e41['proxy']:.5f}")

    t0 = time.time()
    best, stats = iter_block_pair_search(
        e25["placement"], e41["placement"], benchmark, plc,
        n_iters=args.n_iters,
        block_polish_budget=args.block_polish,
        pair_max_attempts=args.pair_attempts,
        pair_budget=args.pair_budget,
        pair_polish_budget=args.pair_polish,
        seed_base=args.seed_base,
    )
    wall = time.time() - t0

    final_proxy = float(compute_proxy_cost(best, benchmark, plc)["proxy_cost"])
    final_overlap = compute_overlap_metrics(best, benchmark)["overlap_count"]

    out = {
        "bench_name": bench_name,
        "e25_proxy": float(e25["proxy"]),
        "e41_proxy": float(e41["proxy"]),
        "min_parent_proxy": min(float(e25["proxy"]), float(e41["proxy"])),
        "final_proxy": final_proxy,
        "final_overlap": final_overlap,
        "lift_vs_min_parent": min(float(e25["proxy"]), float(e41["proxy"])) - final_proxy,
        "lift_frac": (min(float(e25["proxy"]), float(e41["proxy"])) - final_proxy)
                       / min(float(e25["proxy"]), float(e41["proxy"])),
        "wall_seconds": wall,
        "n_iters_run": stats["n_iters_run"],
        "best_iter": stats["best_iter"],
        "iter_log": stats["iter_log"],
        "args": vars(args),
    }
    out_summary = _HERE.parent / "results" / f"iter_{bench_name}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(out, indent=2))
    out_pt = _HERE.parent / "results" / f"iter_{bench_name}.pt"
    torch.save({"placement": best.detach().cpu(), "stats": stats, "bench_name": bench_name},
               out_pt)
    print(f"\n[E72] saved -> {out_summary}, {out_pt}")
    print(f"  start min(E25,E41) = {out['min_parent_proxy']:.5f}")
    print(f"  final              = {final_proxy:.5f} overlap={final_overlap}")
    print(f"  lift               = {out['lift_vs_min_parent']:+.5f} ({100*out['lift_frac']:+.3f}%)")


if __name__ == "__main__":
    main()
