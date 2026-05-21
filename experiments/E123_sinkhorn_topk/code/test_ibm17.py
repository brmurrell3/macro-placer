"""E123 — ibm17 single-bench smoke comparing Sinkhorn vs torch.topk + CD polish.

Pipeline (matches the "V3Min ovl10 720s" reference baseline):
  1) V3 placer (Sinkhorn or topk) with overlap_lambda_end=10, 500 Adam steps
  2) greedy_macro_legalize + project_overlaps
  3) CD adaptive polish (here: 600s, vs 720s "V3Min" baseline — gap from
     Adam descent timing is left for the CD budget so we don't overshoot)

Baseline target (from V3Min ovl10 720s on ibm17): ~1.20.
Goal: Sinkhorn variant ≤ 1.18.

Run:
  uv run python experiments/E123_sinkhorn_topk/code/test_ibm17.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3_sinkhorn import SmoothGlobalPlacerV3Sinkhorn


# Use V3Min "ovl10 720s" baseline budget split: ~120s smooth + ~600s CD = 720s.
BENCH = os.environ.get("E123_BENCH", "ibm17")
CD_BUDGET_S = float(os.environ.get("E123_CD_BUDGET", "600.0"))


def cd_polish(pos: torch.Tensor, benchmark, plc, budget_s: float):
    t0 = time.time()
    ovl_before = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl_before > 0:
        print(f"  WARN: input has {ovl_before} overlaps; CD requires legal init", flush=True)
        return pos, float("inf"), 0.0, 0

    placement_f64 = pos.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    stats = run_cd_adaptive(
        evaluator=evaluator,
        benchmark=benchmark,
        plc=plc,
        movable=movable,
        min_time_s=0.0,
        hard_cap_s=budget_s,
        patience=3,
        plateau_threshold=1e-4,
        log_fn=None,
    )
    polished = evaluator.placement.clone().to(torch.float32)
    final_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    wall = time.time() - t0
    return polished, final_proxy, wall, stats.get("sweeps", 0)


def run_one(label: str, placer, benchmark, plc):
    print(f"\n=== {label} on {BENCH} ===", flush=True)
    t0 = time.time()
    pos = placer.place(benchmark)
    wall_smooth = time.time() - t0
    proxy_raw = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    ovl_raw = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(f"  RAW (smooth+legalize): proxy={proxy_raw:.5f} ovl={ovl_raw} wall={wall_smooth:.1f}s", flush=True)

    polished, proxy_polish, wall_polish, sweeps = cd_polish(pos, benchmark, plc, CD_BUDGET_S)
    print(f"  POLISHED (CD {CD_BUDGET_S}s, {sweeps} sweeps): proxy={proxy_polish:.5f} wall={wall_polish:.1f}s", flush=True)

    return {
        "label": label,
        "raw_proxy": proxy_raw,
        "polish_proxy": proxy_polish,
        "raw_wall_s": wall_smooth,
        "polish_wall_s": wall_polish,
        "raw_overlaps": ovl_raw,
        "cd_sweeps": sweeps,
    }


def main():
    bench_dir = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"Loaded {BENCH}: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}, "
        f"canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
        flush=True,
    )

    # Use V3Min ovl10 baseline config.
    base_cfg = dict(
        num_steps=500,
        lr_frac=0.005,
        init="sdf",
        overlap_lambda_end=10.0,
        verbose=True,
        log_every=100,
    )

    results = []

    # 1) E111 V3 with torch.topk (ablation reference, exactly equivalent to E111)
    placer_topk = SmoothGlobalPlacerV3Sinkhorn(
        **base_cfg, use_sinkhorn=False,
    )
    results.append(run_one("E123 V3-Sinkhorn (use_sinkhorn=False, == E111 V3)", placer_topk, benchmark, plc))
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) Sinkhorn ε=0.1 (default)
    placer_sk = SmoothGlobalPlacerV3Sinkhorn(
        **base_cfg, use_sinkhorn=True,
        sinkhorn_eps=0.1, sinkhorn_iters=50,
    )
    results.append(run_one("E123 V3-Sinkhorn (eps=0.1, iters=50)", placer_sk, benchmark, plc))
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 3) Sinkhorn ε=0.05 (closer to true topk, less gradient flow on non-top)
    placer_sk_sharp = SmoothGlobalPlacerV3Sinkhorn(
        **base_cfg, use_sinkhorn=True,
        sinkhorn_eps=0.05, sinkhorn_iters=80,
    )
    results.append(run_one("E123 V3-Sinkhorn (eps=0.05, iters=80, sharp)", placer_sk_sharp, benchmark, plc))
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 4) Sinkhorn ε=0.3 (more smoothing, more gradient flow)
    placer_sk_smooth = SmoothGlobalPlacerV3Sinkhorn(
        **base_cfg, use_sinkhorn=True,
        sinkhorn_eps=0.3, sinkhorn_iters=50,
    )
    results.append(run_one("E123 V3-Sinkhorn (eps=0.3, iters=50, smooth)", placer_sk_smooth, benchmark, plc))
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 5) Sinkhorn ε anneal 0.3 → 0.05 (curriculum: smooth gradient early, sharp late)
    placer_sk_anneal = SmoothGlobalPlacerV3Sinkhorn(
        **base_cfg, use_sinkhorn=True,
        sinkhorn_eps=0.3, sinkhorn_eps_end=0.05, sinkhorn_iters=60,
    )
    results.append(run_one("E123 V3-Sinkhorn (eps anneal 0.3→0.05, iters=60)", placer_sk_anneal, benchmark, plc))

    print("\n\n=== SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['label']:55s} raw={r['raw_proxy']:.5f}  polish={r['polish_proxy']:.5f}  "
              f"raw_wall={r['raw_wall_s']:.0f}s  polish_wall={r['polish_wall_s']:.0f}s",
              flush=True)

    print(f"\nReference: V3Min ovl10 720s on ibm17 ≈ 1.20", flush=True)
    print(f"Goal:      polish_proxy < 1.18", flush=True)

    out_path = _HERE.parent / "results" / f"{BENCH}_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
