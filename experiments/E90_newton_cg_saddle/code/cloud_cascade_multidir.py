"""E90 cloud — cascade_multidir end-to-end on cached cascade outputs.

Companion to cloud_validate.py (single-pass). This driver iterates the
multi-direction saddle escape, which compounded -0.614 % lift on local
ibm01 over 3 iters where single-pass got -0.35 %. The hypothesis under
test: does iteration also compound on wall-bound cascade outputs
(ibm03/06/07) where single-pass only finds +0.01-0.10 % per bench?

Sharing the cloud with PATH A's A4 + PATH B's E91 — use small thread
budget (OPENBLAS_NUM_THREADS=2) per attempt.
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from cascade_multidir import cascade_multidir_escape

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=["ibm03"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--polish-budget-s", type=float, default=60.0)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--total-budget-s", type=float, default=2400.0)
    ap.add_argument("--only-rank-at-least", type=int, default=2)
    ap.add_argument("--cascade-dir",
                    default="experiments/E84_cascading_saddle/results")
    ap.add_argument("--out-dir",
                    default="experiments/E90_newton_cg_saddle/results/cloud_iter")
    args = ap.parse_args()

    cascade_dir = Path(args.cascade_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = []
    t_total = time.time()
    for bench_name in args.benches:
        cache = cascade_dir / f"cascade_{bench_name}.pt"
        if not cache.exists():
            print(f"[{bench_name}] SKIP — no cached cascade output at {cache}")
            continue
        bench_dir = find_benchmark_dir(bench_name)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        cached = torch.load(cache, weights_only=False, map_location="cpu")
        plateau = cached["placement"].clone().float()
        init_proxy = float(compute_proxy_cost(plateau, benchmark, plc)["proxy_cost"])
        init_ovl = int(compute_overlap_metrics(plateau, benchmark)["overlap_count"])
        print(f"[{bench_name}] plateau: init={init_proxy:.5f} ovl={init_ovl}")
        if init_ovl != 0:
            continue
        t0 = time.time()
        best_state, stats = cascade_multidir_escape(
            plateau, benchmark, plc,
            K=args.K, eps_values=tuple(args.eps),
            polish_budget_s=args.polish_budget_s,
            max_iters=args.max_iters,
            total_budget_s=args.total_budget_s,
            only_rank_at_least=args.only_rank_at_least,
        )
        wall = time.time() - t0
        best_proxy = float(stats["best_proxy"])
        best_ovl = int(compute_overlap_metrics(best_state, benchmark)["overlap_count"])
        lift = (init_proxy - best_proxy) / init_proxy * 100.0
        print(f"[{bench_name}] cascade_multidir done: "
              f"{init_proxy:.5f} -> {best_proxy:.5f}  "
              f"({lift:+.3f}%)  ovl={best_ovl}  "
              f"iters={stats['iters_run']}  wall={wall:.0f}s")
        summary = {
            "bench": bench_name,
            "init_proxy": init_proxy,
            "best_proxy": best_proxy,
            "lift_pct": lift,
            "final_overlap": best_ovl,
            "iters_run": stats["iters_run"],
            "iter_log": stats["iter_log"],
            "wall_seconds": wall,
        }
        aggregate.append(summary)
        (out_dir / f"iter_{bench_name}.json").write_text(json.dumps(summary, indent=2))
        torch.save(
            {"placement": best_state.cpu(), "bench_name": bench_name, "label": "E90_cascade_multidir"},
            out_dir / f"iter_{bench_name}.pt",
        )

    init_sum = sum(r["init_proxy"] for r in aggregate)
    best_sum = sum(r["best_proxy"] for r in aggregate)
    avg_init = init_sum / max(1, len(aggregate))
    avg_best = best_sum / max(1, len(aggregate))
    avg_lift = (avg_init - avg_best) / max(1e-12, avg_init) * 100.0
    print(f"\n=== AGGREGATE ({len(aggregate)} benches) ===")
    print(f"avg init={avg_init:.5f} -> avg best={avg_best:.5f}  avg lift={avg_lift:+.3f}%")
    (out_dir / "iter_aggregate.json").write_text(json.dumps({
        "benches": args.benches,
        "avg_init": avg_init,
        "avg_best": avg_best,
        "avg_lift_pct": avg_lift,
        "per_bench": aggregate,
        "wall_total_seconds": time.time() - t_total,
    }, indent=2))


if __name__ == "__main__":
    main()
