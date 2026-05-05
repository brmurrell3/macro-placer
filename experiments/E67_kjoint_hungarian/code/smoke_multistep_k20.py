"""E67 multi-step plateau-lift with K=20 (smaller cluster for tighter plateau).

K=50 multi-step (`smoke_multistep.py`) hit MARGINAL verdict at avg -0.11 %.
Most benches saturate around step 30 with 15 consecutive rejections —
the top-50-adjacency pool is exhausted even with diversification. K=20
trades coupling depth for slot pool coverage: smaller cluster ⇒ smaller
cost matrix ⇒ faster per-step ⇒ more steps in budget AND each cluster's
candidate slots have a denser feasible subset (n_no_op should drop).

Same N=50 budget, same 4 winning-lane benches. Compare aggregate Δ vs
the K=50 baseline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from kjoint_hungarian import kjoint_hungarian_step

CACHE_DIR = _ROOT / "experiments/E69_sequence_pair_search/results/placements"
N_STEPS = 50
K = 20      # smaller cluster
N_SLOTS = 100
WINNING_LANE = {"ibm01": "e25", "ibm04": "e41", "ibm09": "e41", "ibm12": "e41"}


def load_cached(bench_name: str, lane: str):
    cache_path = CACHE_DIR / f"{lane}_{bench_name}.pt"
    cache_obj = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if isinstance(cache_obj, dict):
        return cache_obj["placement"], cache_obj.get("proxy", None)
    return cache_obj, None


def run_loop(bench_name, lane, n_steps=N_STEPS):
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    placement_t, _ = load_cached(bench_name, lane)
    placement = placement_t.detach().clone().to(torch.float64)
    proxy_in = compute_proxy_cost(placement, bench, plc)["proxy_cost"]
    ov_in = compute_overlap_metrics(placement, bench)["overlap_count"]
    print(f"[init] {bench_name}/{lane} proxy_in={proxy_in:.5f} overlaps={ov_in}",
          flush=True)
    if ov_in != 0:
        return None

    evaluator = IncrementalProxyEvaluator(bench, plc, placement)
    accept_count = 0
    move_count = 0
    skip_illegal = 0
    rejection_streak = 0
    t_total0 = time.perf_counter()
    last_accept_step = -1

    for step in range(n_steps):
        t_step0 = time.perf_counter()
        new_placement, info = kjoint_hungarian_step(
            benchmark=bench, placement=placement, plc=plc, evaluator=evaluator,
            k=K, n_slots=N_SLOTS, mode="adjacency",
            commit_mode="sequential", cluster_seed=step,
        )
        step_wall = time.perf_counter() - t_step0
        accepted = info["accepted"]
        if accepted:
            accept_count += 1
            move_count += info.get("commit_n_moved", 0)
            placement = new_placement
            rejection_streak = 0
            last_accept_step = step
        else:
            rejection_streak += 1
        skip_illegal += info.get("commit_n_skipped_illegal", 0)
        if (step + 1) % 5 == 0 or accepted:
            cur_proxy = compute_proxy_cost(placement, bench, plc)["proxy_cost"]
            print(f"  step {step+1:>3d}: acc={accepted} "
                  f"mv={info.get('commit_n_moved', 0):>2d}/"
                  f"{info.get('commit_n_pending', 0):>2d} "
                  f"proxy={cur_proxy:.5f} (Δ={cur_proxy - proxy_in:+.6f}) "
                  f"w={step_wall:.2f}s", flush=True)
        if rejection_streak >= 15:
            print(f"  early stop at step {step+1}: 15 rejections", flush=True)
            break

    proxy_out = compute_proxy_cost(placement, bench, plc)["proxy_cost"]
    delta = proxy_out - proxy_in
    print(f"[done] {bench_name}/{lane}: {proxy_in:.5f}->{proxy_out:.5f} "
          f"Δ={delta:+.6f} ({delta*100:+.3f}%); acc={accept_count} "
          f"mv={move_count} skip={skip_illegal} wall={time.perf_counter()-t_total0:.0f}s",
          flush=True)
    return {"bench": bench_name, "lane": lane, "proxy_in": proxy_in,
            "proxy_out": proxy_out, "delta": delta, "accepts": accept_count,
            "moves": move_count}


def main():
    print(f"=== E67 K=20 multi-step plateau-lift (winning lanes) ===", flush=True)
    t0 = time.perf_counter()
    results = []
    for bench, lane in WINNING_LANE.items():
        print(f"\n--- {bench} (winning lane: {lane.upper()}) ---", flush=True)
        r = run_loop(bench, lane)
        if r is not None:
            results.append(r)

    print("\n=== SUMMARY (K=20) ===", flush=True)
    print(f"{'bench':<6s} {'lane':<5s} {'proxy_in':>9s} {'proxy_out':>10s} "
          f"{'Δ':>10s} {'Δ%':>7s} {'acc':>4s} {'mv':>4s}", flush=True)
    for r in results:
        print(f"{r['bench']:<6s} {r['lane']:<5s} {r['proxy_in']:9.5f} "
              f"{r['proxy_out']:10.5f} {r['delta']:+10.6f} "
              f"{r['delta']*100:+6.2f}% {r['accepts']:>4d} {r['moves']:>4d}",
              flush=True)
    avg = sum(r["delta"] for r in results) / max(len(results), 1)
    print(f"\n[verdict K=20] avg Δ = {avg:+.6f} ({avg*100:+.3f}%)", flush=True)
    print(f"[total wall] {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
