"""Driver: run SP-guided directed-swap search on a benchmark.

Loads cached E25 and E41 placements, runs sp_guided_swap_search with optional
CD polish, saves result to .pt and writes summary JSON.

Usage:
  uv run python experiments/E69_sequence_pair_search/code/run_sp_search.py \
      <bench> [--max-attempts N] [--swap-budget S] [--polish-budget S]
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

from sp_search_placer import sp_guided_swap_search, cd_polish

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--swap-budget", type=float, default=1800.0)
    ap.add_argument("--polish-budget", type=float, default=600.0)
    ap.add_argument("--swap-order", default="spatial_close_first")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bench_name = args.bench
    placements_dir = _HERE.parent / "results" / "placements"
    e25_path = placements_dir / f"e25_{bench_name}.pt"
    e41_path = placements_dir / f"e41_{bench_name}.pt"
    if not e25_path.exists() or not e41_path.exists():
        print(f"[run_sp_search] missing placements for {bench_name}", file=sys.stderr)
        sys.exit(2)

    e25 = torch.load(e25_path, weights_only=False)
    e41 = torch.load(e41_path, weights_only=False)

    print(f"[run_sp_search] loading {bench_name}...")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # If E25 has lower proxy, start from E25; else from E41 (whichever is
    # better is the basin we're trying to escape).
    if e25["proxy"] <= e41["proxy"]:
        start_pl, start_other = e25, e41
        start_label, other_label = "E25", "E41"
    else:
        start_pl, start_other = e41, e25
        start_label, other_label = "E41", "E25"
    print(f"[run_sp_search] start={start_label} ({start_pl['proxy']:.5f}), "
          f"target_basin={other_label} ({start_other['proxy']:.5f})")

    t0 = time.time()
    swapped, stats = sp_guided_swap_search(
        start_pl["placement"], start_other["placement"],
        benchmark, plc,
        max_attempts=args.max_attempts,
        budget_seconds=args.swap_budget,
        seed=args.seed,
        swap_order=args.swap_order,
    )
    swap_wall = time.time() - t0

    # CD polish.
    t1 = time.time()
    polished = cd_polish(
        swapped, benchmark, plc, budget_seconds=args.polish_budget,
    )
    polish_wall = time.time() - t1
    polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]

    result = {
        "bench_name": bench_name,
        "start_basin": start_label,
        "start_proxy": float(start_pl["proxy"]),
        "other_basin_proxy": float(start_other["proxy"]),
        "swap_stats": {k: v for k, v in stats.items() if k != "proxy_history"},
        "swap_wall_seconds": swap_wall,
        "post_swap_proxy": stats["final_proxy"],
        "polished_proxy": polished_proxy,
        "polished_overlap": polished_overlap,
        "polish_wall_seconds": polish_wall,
        "best_proxy": min(
            float(start_pl["proxy"]),
            float(start_other["proxy"]),
            stats["final_proxy"] or float("inf"),
            polished_proxy,
        ),
        "args": {
            "max_attempts": args.max_attempts,
            "swap_budget": args.swap_budget,
            "polish_budget": args.polish_budget,
            "swap_order": args.swap_order,
            "seed": args.seed,
        },
    }
    out_summary = _HERE.parent / "results" / f"sp_search_{bench_name}.json"
    out_summary.write_text(json.dumps(result, indent=2))

    out_placement = _HERE.parent / "results" / f"sp_search_{bench_name}.pt"
    torch.save(
        {
            "swapped_placement": swapped.detach().cpu(),
            "polished_placement": polished.detach().cpu(),
            "stats": stats,
            "polished_proxy": polished_proxy,
            "polished_overlap": polished_overlap,
            "bench_name": bench_name,
        },
        out_placement,
    )
    print(f"[run_sp_search] saved -> {out_summary}, {out_placement}")
    print()
    print(f"  start ({start_label}): {start_pl['proxy']:.5f}")
    print(f"  other ({other_label}): {start_other['proxy']:.5f}")
    print(f"  post-swap:         {stats['final_proxy']:.5f}")
    print(f"  post-polish:       {polished_proxy:.5f}  overlap={polished_overlap}")
    print(f"  best:              {result['best_proxy']:.5f}")
    print(f"  vs E48 ibm01 0.892 / ibm12 1.206: "
          f"Δ = {result['best_proxy'] - {'ibm01': 0.892, 'ibm12': 1.206}.get(bench_name, 0):+.5f}")


if __name__ == "__main__":
    main()
