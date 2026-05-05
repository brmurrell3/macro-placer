"""E67 V2 smoke: compare joint vs sequential commit on ibm01 SDF init.

V1 (`smoke.py`) found that K=50 Hungarian's joint commit collides 8 of 50
moves and triggers full revert (Risk b in the manifest predicted this).
V2 fix: apply moves one at a time in priority order with a per-move
legality check (`_is_legal_2d_excluded(..., excluded=[])`) that rejects
collisions against already-committed cluster siblings.

This smoke runs ONE step in each commit mode, on identical input state,
and prints the accept/skip/move counts + proxy delta side-by-side.

Wall budget: < 30 s (~12 s for cost-matrix build × 2 if we re-run; we
build evaluator twice but reuse SDF init).
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
from macro_place.cd_core import sdf_init, project_overlaps
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from kjoint_hungarian import kjoint_hungarian_step


def run_one(bench, plc, placement_f64_template, mode: str):
    placement = placement_f64_template.detach().clone()
    evaluator = IncrementalProxyEvaluator(bench, plc, placement)
    t0 = time.perf_counter()
    new_placement, info = kjoint_hungarian_step(
        benchmark=bench,
        placement=placement,
        plc=plc,
        evaluator=evaluator,
        k=50,
        n_slots=100,
        mode="adjacency",
        commit_mode=mode,
    )
    info["wall_total_s"] = time.perf_counter() - t0
    return new_placement, info


def fmt_info(info):
    return {
        "accepted": info["accepted"],
        "reason": info.get("commit_reason", "?"),
        "n_moved": info.get("commit_n_moved", -1),
        "n_pending": info.get("commit_n_pending", -1),
        "n_skipped_illegal": info.get("commit_n_skipped_illegal", -1),
        "n_no_op": info.get("commit_n_no_op", -1),
        "overlap_count": info.get("commit_overlap_count", -1),
        "proxy_before": info.get("proxy_baseline", float("nan")),
        "proxy_after": info.get("commit_proxy_after", float("nan")),
        "t_total": info.get("t_total_s", float("nan")),
    }


def main():
    print("=== E67 V2 smoke: joint vs sequential commit (ibm01 SDF init) ===",
          flush=True)
    t0 = time.perf_counter()

    bench_dir = find_benchmark_dir("ibm01")
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"[load] ibm01: num_macros={bench.num_macros}, "
        f"n_hard={bench.num_hard_macros}, "
        f"canvas={bench.canvas_width:.1f}x{bench.canvas_height:.1f}",
        flush=True,
    )

    placement = sdf_init(bench)
    placement, _ = project_overlaps(placement, bench)
    overlaps_init = compute_overlap_metrics(placement, bench)
    placement_f64 = placement.detach().clone().to(torch.float64)
    print(
        f"[sdf_init] residual_overlaps={overlaps_init['overlap_count']}, "
        f"placement dtype={placement_f64.dtype}",
        flush=True,
    )

    print("--- run 1: commit_mode='joint' (V1 baseline) ---", flush=True)
    _, info_joint = run_one(bench, plc, placement_f64, "joint")
    j = fmt_info(info_joint)

    print("--- run 2: commit_mode='sequential' (V2 fix) ---", flush=True)
    _, info_seq = run_one(bench, plc, placement_f64, "sequential")
    s = fmt_info(info_seq)

    print("\n=== side-by-side ===", flush=True)
    cols = ("accepted", "reason", "n_pending", "n_moved",
            "n_skipped_illegal", "n_no_op", "overlap_count",
            "proxy_before", "proxy_after", "t_total")
    print(f"{'metric':<22s} {'joint (V1)':<22s} {'sequential (V2)':<22s}",
          flush=True)
    for c in cols:
        jv = j[c]
        sv = s[c]
        if isinstance(jv, float):
            print(f"{c:<22s} {jv:<22.6f} {sv:<22.6f}", flush=True)
        else:
            print(f"{c:<22s} {str(jv):<22s} {str(sv):<22s}", flush=True)

    delta = s["proxy_after"] - s["proxy_before"]
    print(
        f"\n[V2 outcome] {s['n_moved']}/{s['n_pending']} moves committed; "
        f"proxy delta = {delta:+.6f} ({'improvement' if delta < 0 else 'no change'}); "
        f"accepted = {s['accepted']}",
        flush=True,
    )
    print(f"[smoke] total wall = {time.perf_counter() - t0:.2f} s — OK",
          flush=True)


if __name__ == "__main__":
    main()
