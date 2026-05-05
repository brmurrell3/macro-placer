"""E67 multi-step plateau-lift: K=50 Hungarian loop with cluster diversification.

The single-step plateau test (`smoke_plateau.py`) showed lift on the E25
lane (E48-irrelevant) but zero on the E41 lane (E48's winning lane on
3 of 4 fast benches). Hypothesis: the deterministic top-50-by-adjacency
cluster is locally optimal at the K-joint K=3 frontier; with random
sampling from the top-3K, different clusters expose different slack.

This script runs N steps with `cluster_seed = 0..N-1` on the *winning
lane* for each bench (E41 wins ibm04, ibm09, ibm12; E25 wins ibm01).
Each accepted move shifts the placement; subsequent steps work on the
new state. Per-step wall ~5–13 s; budget N=50 steps × 4 benches ≈
1000–2600 s total (~30 min, reasonable for an overnight check).

Verdict rule (multi-step):
  Final Δ <= -0.005 (>= 0.5 % cumulative): integrate as 5th phase.
  -0.005 < Δ <= -0.001: marginal; try larger N or smaller K=20.
  Δ > -0.001:           K=50 Hungarian saturates on the polished plateau;
                         consider different mechanism.
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
K = 50
N_SLOTS = 100

# Per-bench WINNING lane (from docs/results.md per-benchmark table; E48 hybrid
# picks min(E25, E41)). Matches the cached placements:
#   ibm01: E25 wins (0.892 < 0.921)
#   ibm04: E41 wins (1.015 < 1.034)
#   ibm09: E41 wins (0.833 < 0.867)
#   ibm12: E41 wins (1.209 < 1.213)
WINNING_LANE = {"ibm01": "e25", "ibm04": "e41", "ibm09": "e41", "ibm12": "e41"}


def load_cached(bench_name: str, lane: str):
    cache_path = CACHE_DIR / f"{lane}_{bench_name}.pt"
    cache_obj = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if isinstance(cache_obj, dict):
        return cache_obj["placement"], cache_obj.get("proxy", None)
    return cache_obj, None


def run_loop(bench_name: str, lane: str, n_steps: int = N_STEPS):
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))

    placement_t, cached_proxy = load_cached(bench_name, lane)
    placement = placement_t.detach().clone().to(torch.float64)
    proxy_in = compute_proxy_cost(placement, bench, plc)["proxy_cost"]
    ov_in = compute_overlap_metrics(placement, bench)["overlap_count"]
    print(
        f"[init] {bench_name}/{lane} proxy_in={proxy_in:.5f} "
        f"(cache={cached_proxy if cached_proxy else '?'}) overlaps={ov_in}",
        flush=True,
    )
    if ov_in != 0:
        print(f"  skip: cached has {ov_in} overlaps", flush=True)
        return None

    evaluator = IncrementalProxyEvaluator(bench, plc, placement)

    accept_count = 0
    move_count = 0
    skip_illegal = 0
    cumul_delta = 0.0
    rejection_streak = 0
    t_total0 = time.perf_counter()
    last_log_t = t_total0
    log_every = 5  # print every 5 steps

    for step in range(n_steps):
        t_step0 = time.perf_counter()
        new_placement, info = kjoint_hungarian_step(
            benchmark=bench,
            placement=placement,
            plc=plc,
            evaluator=evaluator,
            k=K,
            n_slots=N_SLOTS,
            mode="adjacency",
            commit_mode="sequential",
            cluster_seed=step,
        )
        step_wall = time.perf_counter() - t_step0
        accepted = info["accepted"]
        if accepted:
            accept_count += 1
            move_count += info.get("commit_n_moved", 0)
            placement = new_placement
            cumul_delta += info.get("commit_proxy_after", 0) - info.get(
                "proxy_baseline", 0
            )
            rejection_streak = 0
        else:
            rejection_streak += 1
        skip_illegal += info.get("commit_n_skipped_illegal", 0)

        if (step + 1) % log_every == 0 or accepted:
            cur_proxy = compute_proxy_cost(placement, bench, plc)["proxy_cost"]
            print(
                f"  step {step+1:>3d}: accepted={accepted} "
                f"moved={info.get('commit_n_moved', 0):>2d}/"
                f"{info.get('commit_n_pending', 0):>2d} "
                f"proxy={cur_proxy:.5f} (Δ_cumul={cur_proxy - proxy_in:+.6f}) "
                f"wall={step_wall:.2f}s",
                flush=True,
            )

        # Early-stop if rejected for 15 consecutive steps (saturation signal).
        if rejection_streak >= 15:
            print(
                f"  early stop at step {step+1}: 15 consecutive rejections",
                flush=True,
            )
            break

    proxy_out = compute_proxy_cost(placement, bench, plc)["proxy_cost"]
    ov_out = compute_overlap_metrics(placement, bench)["overlap_count"]
    delta = proxy_out - proxy_in
    total_wall = time.perf_counter() - t_total0
    print(
        f"[done] {bench_name}/{lane}: proxy {proxy_in:.5f} -> {proxy_out:.5f} "
        f"(Δ={delta:+.6f}, {delta*100:+.3f}%); accepts={accept_count}, "
        f"moves={move_count}, skipped_illegal={skip_illegal}; "
        f"wall={total_wall:.1f}s",
        flush=True,
    )
    return {
        "bench": bench_name,
        "lane": lane,
        "proxy_in": proxy_in,
        "proxy_out": proxy_out,
        "delta": delta,
        "delta_pct": delta * 100.0,
        "accepts": accept_count,
        "moves": move_count,
        "skipped_illegal": skip_illegal,
        "ov_out": ov_out,
        "wall_s": total_wall,
        "n_steps": step + 1,
    }


def main():
    print(
        f"=== E67 multi-step plateau-lift: N={N_STEPS} K={K} n_slots={N_SLOTS} "
        f"(winning lane only) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    results = []
    for bench, lane in WINNING_LANE.items():
        print(f"\n--- {bench} (winning lane: {lane.upper()}) ---", flush=True)
        r = run_loop(bench, lane)
        if r is not None:
            results.append(r)

    print("\n=== SUMMARY (winning-lane multi-step lift) ===", flush=True)
    print(
        f"{'bench':<6s} {'lane':<5s} {'proxy_in':>9s} {'proxy_out':>10s} "
        f"{'Δ':>10s} {'Δ%':>7s} {'acc':>5s} {'mv':>4s} {'wall':>6s}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['bench']:<6s} {r['lane']:<5s} {r['proxy_in']:9.5f} "
            f"{r['proxy_out']:10.5f} {r['delta']:+10.6f} "
            f"{r['delta_pct']:+6.2f}% {r['accepts']:>5d} {r['moves']:>4d} "
            f"{r['wall_s']:>5.0f}s",
            flush=True,
        )

    avg_delta = sum(r["delta"] for r in results) / max(len(results), 1)
    print(
        f"\n[verdict] avg Δ across {len(results)} winning-lane runs = "
        f"{avg_delta:+.6f} ({avg_delta*100:+.3f}%)",
        flush=True,
    )
    if avg_delta <= -0.005:
        print("[VERDICT: STRONG LIFT] integrate K=50 Hungarian as 5th phase.",
              flush=True)
    elif avg_delta <= -0.001:
        print("[VERDICT: MARGINAL] try larger N or K=20.", flush=True)
    else:
        print("[VERDICT: SATURATED] plateau too tight for K=50 even with "
              "diversification; consider different mechanism (NEB, smaller K).",
              flush=True)
    print(f"[total wall] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
