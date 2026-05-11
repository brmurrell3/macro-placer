"""Driver: run axis-rotation transplant search on a benchmark.

Loads cached E25 + E41 placements (from E69), runs axis_rotation_search
starting from the lower-proxy basin (transplanting from the other),
optionally polishes with short CD, saves placement and JSON summary.

Usage:
  uv run python experiments/E70_axis_rotation/code/run_axis_rotation.py \
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

from axis_rotation_placer import axis_rotation_search, cd_polish_placement

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--swap-budget", type=float, default=2400.0)
    ap.add_argument("--polish-budget", type=float, default=600.0)
    ap.add_argument("--transplant-order", default="spatial_close_first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-from", choices=["best", "e25", "e41"], default="best")
    args = ap.parse_args()

    bench_name = args.bench
    e69_placements = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25_path = e69_placements / f"e25_{bench_name}.pt"
    e41_path = e69_placements / f"e41_{bench_name}.pt"
    if not e25_path.exists() or not e41_path.exists():
        print(f"[E70] missing cached E25/E41 placements for {bench_name}", file=sys.stderr)
        sys.exit(2)

    e25 = torch.load(e25_path, weights_only=False)
    e41 = torch.load(e41_path, weights_only=False)

    print(f"[E70] loading {bench_name}...")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    if args.start_from == "best":
        if e25["proxy"] <= e41["proxy"]:
            start_pl, start_other = e25, e41
            start_label, other_label = "E25", "E41"
        else:
            start_pl, start_other = e41, e25
            start_label, other_label = "E41", "E25"
    elif args.start_from == "e25":
        start_pl, start_other = e25, e41
        start_label, other_label = "E25", "E41"
    else:
        start_pl, start_other = e41, e25
        start_label, other_label = "E41", "E25"

    print(f"[E70] start={start_label} ({start_pl['proxy']:.5f}), "
          f"target_basin={other_label} ({start_other['proxy']:.5f})")

    t0 = time.time()
    result, stats = axis_rotation_search(
        start_pl["placement"], start_other["placement"], benchmark, plc,
        max_attempts=args.max_attempts,
        budget_seconds=args.swap_budget,
        seed=args.seed,
        transplant_order=args.transplant_order,
    )
    swap_wall = time.time() - t0

    t1 = time.time()
    polished = cd_polish_placement(
        result, benchmark, plc, budget_seconds=args.polish_budget,
    )
    polish_wall = time.time() - t1
    polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]

    out = {
        "bench_name": bench_name,
        "start_basin": start_label,
        "start_proxy": float(start_pl["proxy"]),
        "other_basin_proxy": float(start_other["proxy"]),
        "transplant_stats": {k: v for k, v in stats.items() if k != "proxy_history"},
        "swap_wall_seconds": swap_wall,
        "post_transplant_proxy": stats["final_proxy"],
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
            "transplant_order": args.transplant_order,
            "seed": args.seed,
            "start_from": args.start_from,
        },
    }
    out_summary = _HERE.parent / "results" / f"axis_rot_{bench_name}.json"
    out_summary.write_text(json.dumps(out, indent=2))

    out_placement = _HERE.parent / "results" / f"axis_rot_{bench_name}.pt"
    torch.save(
        {
            "transplanted_placement": result.detach().cpu(),
            "polished_placement": polished.detach().cpu(),
            "stats": stats,
            "polished_proxy": polished_proxy,
            "polished_overlap": polished_overlap,
            "bench_name": bench_name,
        },
        out_placement,
    )
    print(f"[E70] saved -> {out_summary}, {out_placement}")
    print()
    print(f"  start ({start_label}): {start_pl['proxy']:.5f}")
    print(f"  other ({other_label}): {start_other['proxy']:.5f}")
    print(f"  post-transplant:   {stats['final_proxy']:.5f}")
    print(f"  post-polish:       {polished_proxy:.5f}  overlap={polished_overlap}")
    print(f"  best:              {out['best_proxy']:.5f}")


if __name__ == "__main__":
    main()
