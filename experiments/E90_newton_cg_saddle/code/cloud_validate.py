"""E90 cloud validation — multi-direction saddle escape on cached cascade outputs.

For each cached cascade bench (experiments/E84_cascading_saddle/results/cascade_<bench>.pt),
run multi-direction saddle escape with full polish budget. Compare to cached cascade
input proxy. Promote-decision: aggregate lift > 0.3% across --fast (4 IBM).

Designed to run on cloud with single-job to avoid CPU contention with PATH A's
parallel CD-cascade-adaptive runs (--jobs=4).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from multi_saddle import multi_saddle_escape

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


# Curated configs based on local-spike tuning.
# At cascade plateau, rank-2/rank-3 combinations with eps ~0.5-1.5 beat the cache.
# Use a coarse eps grid with two scales (0.5 the sweet spot, 1.5 for sign-vecs
# that benefit from larger steps).
DEFAULT_K = 3
DEFAULT_EPS = (0.5, 1.0, 2.0)
DEFAULT_POLISH_S = 180.0           # 3 minutes per attempt (cloud-class)
DEFAULT_BUDGET_PER_BENCH_S = 1800   # 30 min per bench
DEFAULT_ONLY_RANK_AT_LEAST = 2     # multi-direction only — single-direction is what cascade did


def run_one(
    bench_name: str,
    cache_path: Path,
    out_dir: Path,
    *,
    K: int = DEFAULT_K,
    eps_values=DEFAULT_EPS,
    polish_budget_s: float = DEFAULT_POLISH_S,
    only_rank_at_least: int = DEFAULT_ONLY_RANK_AT_LEAST,
    total_budget_s: float = DEFAULT_BUDGET_PER_BENCH_S,
) -> dict:
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    cached = torch.load(cache_path, weights_only=False, map_location="cpu")
    plateau = cached["placement"].clone().float()
    init_proxy = float(compute_proxy_cost(plateau, benchmark, plc)["proxy_cost"])
    init_ovl = int(compute_overlap_metrics(plateau, benchmark)["overlap_count"])
    print(f"[{bench_name}] plateau loaded: init={init_proxy:.5f} ovl={init_ovl}")
    if init_ovl != 0:
        print(f"[{bench_name}] SKIP (input has {init_ovl} overlaps)")
        return {"bench": bench_name, "skipped": True, "reason": f"{init_ovl} overlaps in input"}

    t0 = time.time()
    best_state, stats = multi_saddle_escape(
        plateau, benchmark, plc,
        K=K, eps_values=eps_values,
        polish_budget_s=polish_budget_s,
        only_rank_at_least=only_rank_at_least,
        total_budget_s=total_budget_s,
    )
    wall = time.time() - t0
    best_proxy = stats["best_proxy"]
    best_ovl = int(compute_overlap_metrics(best_state, benchmark)["overlap_count"])

    summary = {
        "bench": bench_name,
        "cache_input": str(cache_path),
        "K": K, "eps_values": list(eps_values),
        "polish_budget_s": polish_budget_s,
        "only_rank_at_least": only_rank_at_least,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_overlap": best_ovl,
        "lift_pct": (init_proxy - best_proxy) / init_proxy * 100.0,
        "eigvals": stats["eigvals"],
        "K_actual": stats["K_actual"],
        "n_attempts": len(stats["attempts"]),
        "wall_seconds": wall,
        "attempts": stats["attempts"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"validate_{bench_name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    out_pt = out_dir / f"validate_{bench_name}.pt"
    torch.save({"placement": best_state.cpu(), "bench_name": bench_name, "label": "E90"}, out_pt)
    print(f"[{bench_name}] done: {init_proxy:.5f} -> {best_proxy:.5f}  "
          f"(Δ={best_proxy - init_proxy:+.5f} = {summary['lift_pct']:+.3f}%)  "
          f"ovl={best_ovl}  wall={wall:.0f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benches", nargs="+",
        default=["ibm01", "ibm03", "ibm06", "ibm07"],  # --fast set
        help="Bench names to validate.",
    )
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--polish-budget-s", type=float, default=DEFAULT_POLISH_S)
    ap.add_argument("--total-budget-s", type=float, default=DEFAULT_BUDGET_PER_BENCH_S)
    ap.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--only-rank-at-least", type=int, default=DEFAULT_ONLY_RANK_AT_LEAST)
    ap.add_argument("--cascade-dir",
                    default="experiments/E84_cascading_saddle/results")
    ap.add_argument("--out-dir", default="experiments/E90_newton_cg_saddle/results/cloud")
    args = ap.parse_args()

    cascade_dir = Path(args.cascade_dir)
    out_dir = Path(args.out_dir)

    aggregate = []
    t_total = time.time()
    for bench in args.benches:
        cache = cascade_dir / f"cascade_{bench}.pt"
        if not cache.exists():
            print(f"[{bench}] SKIP — no cached cascade output at {cache}")
            continue
        result = run_one(
            bench, cache, out_dir,
            K=args.K, eps_values=tuple(args.eps),
            polish_budget_s=args.polish_budget_s,
            only_rank_at_least=args.only_rank_at_least,
            total_budget_s=args.total_budget_s,
        )
        aggregate.append(result)

    # Aggregate report.
    agg_path = out_dir / "validate_aggregate.json"
    init_sum = sum(r["init_proxy"] for r in aggregate if "init_proxy" in r)
    best_sum = sum(r["best_proxy"] for r in aggregate if "best_proxy" in r)
    avg_init = init_sum / max(1, sum(1 for r in aggregate if "init_proxy" in r))
    avg_best = best_sum / max(1, sum(1 for r in aggregate if "best_proxy" in r))
    agg_lift = (avg_init - avg_best) / max(1e-12, avg_init) * 100.0
    agg_path.write_text(json.dumps({
        "benches": args.benches,
        "n_runs": len(aggregate),
        "avg_init": avg_init,
        "avg_best": avg_best,
        "avg_lift_pct": agg_lift,
        "per_bench": aggregate,
        "wall_total_seconds": time.time() - t_total,
    }, indent=2))
    print(f"\n=== AGGREGATE ({len(aggregate)} benches) ===")
    print(f"avg init={avg_init:.5f} -> avg best={avg_best:.5f}  "
          f"avg lift={agg_lift:+.3f}%")
    print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
