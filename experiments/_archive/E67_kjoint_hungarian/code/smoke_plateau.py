"""E67 plateau-lift test: K=50 Hungarian step on cached E25/E41 placements.

The E69 sequence-pair experiment cached lane outputs at
`experiments/E69_sequence_pair_search/results/placements/` —
post-CD-LNS-SA(-K-joint) placements at the plateau, with zero overlaps.
This script applies one V2 (sequential commit) K=50 Hungarian step to
each cached placement and reports the proxy delta + accept rate.

The previous V2 smoke (`smoke_v2.py`) verified the *mechanism* on a
non-converged SDF init (proxy 1.195). This script is the more
informative test: does V2 produce lift on the *plateau* where it would
actually integrate as a fifth phase post-K-joint K=3?

Decision rule for next-session integration:
  Δ <= -0.005 (>= 0.5 % improvement): K=50 Hungarian is alive on the
                                       plateau → integrate.
  -0.005 < Δ <= -0.0005: marginal lift; try cluster diversification.
  Δ > -0.0005:           plateau too tight; K=50 likely too coarse,
                         try K=20 or different cluster heuristic.

Wall budget per bench: ~10 s (cost-matrix dominated, similar to V1/V2
smoke). 4 benches × 2 lanes × ~10 s = ~80 s total.
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
BENCHES = ["ibm01", "ibm04", "ibm09", "ibm12"]
LANES = ["e25", "e41"]


def run_one(bench_name: str, lane: str):
    cache_path = CACHE_DIR / f"{lane}_{bench_name}.pt"
    if not cache_path.exists():
        return None, f"missing cache {cache_path.name}"

    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))

    cache_obj = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if isinstance(cache_obj, dict):
        placement_t = cache_obj["placement"]
    else:
        placement_t = cache_obj
    placement = placement_t.detach().clone().to(torch.float64)

    ov_in = compute_overlap_metrics(placement, bench)
    proxy_in = compute_proxy_cost(placement, bench, plc)
    if ov_in["overlap_count"] != 0:
        return None, (
            f"cached placement has {ov_in['overlap_count']} overlaps; skip "
            f"(K=50 Hungarian's accept rule requires zero pre-step overlaps)"
        )

    evaluator = IncrementalProxyEvaluator(bench, plc, placement)
    eval_proxy = evaluator.current_cost()["proxy"]

    t0 = time.perf_counter()
    new_placement, info = kjoint_hungarian_step(
        benchmark=bench,
        placement=placement,
        plc=plc,
        evaluator=evaluator,
        k=50,
        n_slots=100,
        mode="adjacency",
        commit_mode="sequential",
    )
    wall = time.perf_counter() - t0

    proxy_after_full = compute_proxy_cost(new_placement, bench, plc)
    ov_after = compute_overlap_metrics(new_placement, bench)
    delta_eval = info.get("commit_proxy_after", eval_proxy) - eval_proxy
    delta_full = proxy_after_full["proxy_cost"] - proxy_in["proxy_cost"]
    return {
        "bench": bench_name,
        "lane": lane,
        "n_macros": int(bench.num_macros),
        "n_hard": int(bench.num_hard_macros),
        "proxy_in_full": float(proxy_in["proxy_cost"]),
        "proxy_in_eval": float(eval_proxy),
        "proxy_out_full": float(proxy_after_full["proxy_cost"]),
        "proxy_out_eval": float(info.get("commit_proxy_after", float("nan"))),
        "delta_eval": float(delta_eval),
        "delta_full": float(delta_full),
        "ov_in": int(ov_in["overlap_count"]),
        "ov_out": int(ov_after["overlap_count"]),
        "n_pending": int(info.get("commit_n_pending", -1)),
        "n_moved": int(info.get("commit_n_moved", -1)),
        "n_skipped_illegal": int(info.get("commit_n_skipped_illegal", -1)),
        "n_no_op": int(info.get("commit_n_no_op", -1)),
        "accepted": bool(info["accepted"]),
        "reason": info.get("commit_reason", "?"),
        "wall_total": float(wall),
        "t_cost_matrix": float(info.get("t_cost_matrix_s", float("nan"))),
        "t_commit": float(info.get("t_commit_s", float("nan"))),
    }, None


def main():
    print(f"=== E67 plateau-lift: K=50 Hungarian on cached E25/E41 placements ===",
          flush=True)
    t0 = time.perf_counter()

    rows = []
    for lane in LANES:
        for b in BENCHES:
            print(f"\n--- {lane.upper()} lane / {b} ---", flush=True)
            row, err = run_one(b, lane)
            if row is None:
                print(f"  skip: {err}", flush=True)
                continue
            rows.append(row)
            print(
                f"  proxy_in={row['proxy_in_full']:.5f} -> "
                f"proxy_out={row['proxy_out_full']:.5f}  "
                f"Δ_full={row['delta_full']:+.6f} ({row['delta_full']*100:+.3f}%)",
                flush=True,
            )
            print(
                f"  pending={row['n_pending']} moved={row['n_moved']} "
                f"skipped_illegal={row['n_skipped_illegal']} no_op={row['n_no_op']}  "
                f"ov_in={row['ov_in']} ov_out={row['ov_out']}  "
                f"accepted={row['accepted']} reason={row['reason']}  "
                f"wall={row['wall_total']:.2f}s",
                flush=True,
            )

    print("\n=== SUMMARY ===", flush=True)
    hdr = (
        f"{'bench':<6s} {'lane':<5s} {'proxy_in':>9s} {'proxy_out':>10s} "
        f"{'Δ_full':>10s} {'Δ%':>7s} {'mv':>3s}/{'pn':<3s} {'acc':>4s} "
        f"{'reason':<20s}"
    )
    print(hdr, flush=True)
    for r in rows:
        print(
            f"{r['bench']:<6s} {r['lane']:<5s} {r['proxy_in_full']:9.5f} "
            f"{r['proxy_out_full']:10.5f} {r['delta_full']:+10.6f} "
            f"{r['delta_full']*100:+6.2f}% {r['n_moved']:>3d}/{r['n_pending']:<3d} "
            f"{str(r['accepted']):<4s} {r['reason']:<20s}",
            flush=True,
        )

    n_accept = sum(1 for r in rows if r["accepted"])
    avg_delta = (
        sum(r["delta_full"] for r in rows if r["accepted"]) / max(n_accept, 1)
    )
    print(
        f"\n[verdict] {n_accept}/{len(rows)} accepted; "
        f"avg Δ_full on accepts = {avg_delta:+.6f} ({avg_delta*100:+.3f}%)",
        flush=True,
    )
    if n_accept >= 1 and avg_delta <= -0.005:
        print("[VERDICT: STRONG LIFT] integrate K=50 Hungarian as 5th phase.",
              flush=True)
    elif n_accept >= 1 and avg_delta <= -0.0005:
        print("[VERDICT: MARGINAL] try cluster diversification + multi-step loop.",
              flush=True)
    else:
        print("[VERDICT: NO LIFT] plateau too tight for K=50; try K=20 or "
              "different cluster heuristic.", flush=True)
    print(f"[total wall] {time.perf_counter() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
