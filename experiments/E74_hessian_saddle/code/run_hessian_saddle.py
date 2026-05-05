"""Driver for Hessian saddle escape on cached E48-equivalent placements.

Uses cached E25/E41 placements as starting state (E48 hybrid would pick
the lower-proxy of the two per bench).
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
    ap.add_argument("--n-eigvecs", type=int, default=5)
    ap.add_argument("--epsilons", type=str, default="0.1,0.3,0.5,1.0,2.0")
    ap.add_argument("--polish-budget", type=float, default=600.0)
    ap.add_argument("--start-from", choices=["best", "e25", "e41"], default="best")
    args = ap.parse_args()

    epsilon_values = tuple(float(x) for x in args.epsilons.split(","))
    bench_name = args.bench
    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25 = torch.load(e69 / f"e25_{bench_name}.pt", weights_only=False)
    e41 = torch.load(e69 / f"e41_{bench_name}.pt", weights_only=False)

    print(f"[E74] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    if args.start_from == "best":
        if e25["proxy"] <= e41["proxy"]:
            start, sl = e25, "E25"
        else:
            start, sl = e41, "E41"
    elif args.start_from == "e25":
        start, sl = e25, "E25"
    else:
        start, sl = e41, "E41"
    print(f"[E74] start={sl} ({start['proxy']:.5f})")
    print(f"[E74] n_eigvecs={args.n_eigvecs} epsilons={epsilon_values}")

    t0 = time.time()
    result, stats = saddle_escape(
        start["placement"], benchmark, plc,
        n_eigvecs=args.n_eigvecs,
        epsilon_values=epsilon_values,
        cd_polish_budget=args.polish_budget,
    )
    wall = time.time() - t0

    final_proxy = float(compute_proxy_cost(result, benchmark, plc)["proxy_cost"])
    final_overlap = compute_overlap_metrics(result, benchmark)["overlap_count"]
    print(f"[E74] DONE: best={stats['best_proxy']:.5f} (vs start {stats['starting_proxy']:.5f}, "
          f"lift {stats['improvement']:+.5f} = {100*stats['improvement_frac']:+.3f}%) wall={wall:.0f}s")

    out = {
        "bench_name": bench_name,
        "start_basin": sl,
        "start_proxy": float(start["proxy"]),
        "starting_proxy_recompute": stats["starting_proxy"],
        "best_proxy": stats["best_proxy"],
        "improvement": stats["improvement"],
        "improvement_frac": stats["improvement_frac"],
        "final_proxy": final_proxy,
        "final_overlap": final_overlap,
        "n_eigvecs": stats["n_eigvecs"],
        "eigenvalues": stats["eigenvalues"],
        "n_attempts": len(stats["attempts"]),
        "n_feasible": sum(1 for a in stats["attempts"] if a["feasible"]),
        "attempts": stats["attempts"],
        "wall_seconds": wall,
        "args": vars(args),
    }
    out_summary = _HERE.parent / "results" / f"hessian_{bench_name}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(out, indent=2))
    out_pt = _HERE.parent / "results" / f"hessian_{bench_name}.pt"
    torch.save({"placement": result.cpu(), "stats": stats, "bench_name": bench_name}, out_pt)
    print(f"[E74] saved -> {out_summary}, {out_pt}")


if __name__ == "__main__":
    main()
