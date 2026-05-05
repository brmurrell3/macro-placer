"""Run Hessian saddle escape starting from a custom placement .pt.

Used to layer E74 on top of E61V2-fresh outputs (E75 wave).
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

from hessian_saddle import saddle_escape

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--start-pt", required=True, help="path to .pt with 'placement' key")
    ap.add_argument("--start-label", default="custom")
    ap.add_argument("--n-eigvecs", type=int, default=2)
    ap.add_argument("--epsilons", type=str, default="0.3,1.0,3.0")
    ap.add_argument("--polish-budget", type=float, default=240.0)
    ap.add_argument("--out-suffix", default="from_custom")
    args = ap.parse_args()

    epsilon_values = tuple(float(x) for x in args.epsilons.split(","))
    bench_name = args.bench
    print(f"[E74-custom] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    start_pt = torch.load(args.start_pt, weights_only=False)
    start_placement = start_pt["placement"]
    start_proxy = float(compute_proxy_cost(start_placement, benchmark, plc)["proxy_cost"])
    print(f"[E74-custom] start_label={args.start_label} start_proxy={start_proxy:.5f}")

    t0 = time.time()
    result, stats = saddle_escape(
        start_placement, benchmark, plc,
        n_eigvecs=args.n_eigvecs,
        epsilon_values=epsilon_values,
        cd_polish_budget=args.polish_budget,
    )
    wall = time.time() - t0

    final_proxy = float(compute_proxy_cost(result, benchmark, plc)["proxy_cost"])
    final_overlap = compute_overlap_metrics(result, benchmark)["overlap_count"]
    print(f"[E74-custom] DONE: best={stats['best_proxy']:.5f} (vs start {stats['starting_proxy']:.5f}, "
          f"lift {stats['improvement']:+.5f} = {100*stats['improvement_frac']:+.3f}%) wall={wall:.0f}s")

    out = {
        "bench_name": bench_name,
        "start_label": args.start_label,
        "start_proxy": start_proxy,
        "best_proxy": stats["best_proxy"],
        "improvement": stats["improvement"],
        "improvement_frac": stats["improvement_frac"],
        "final_overlap": final_overlap,
        "wall_seconds": wall,
        "n_attempts": len(stats["attempts"]),
        "attempts": stats["attempts"],
    }
    out_summary = _HERE.parent / "results" / f"hessian_{bench_name}_{args.out_suffix}.json"
    out_pt = _HERE.parent / "results" / f"hessian_{bench_name}_{args.out_suffix}.pt"
    out_summary.write_text(json.dumps(out, indent=2))
    torch.save({"placement": result.cpu(), "stats": stats, "bench_name": bench_name}, out_pt)
    print(f"[E74-custom] saved -> {out_summary}, {out_pt}")


if __name__ == "__main__":
    main()
