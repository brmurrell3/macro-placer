"""E123 — Fast smoke: Adam descent + legalize + CD60s, skip slow canonical eval.

This is a quick A/B comparison: do 500 steps Adam descent with each
congestion variant, legalize, and run only 60s CD polish. Total wall
per config: ~5 min (4 min descent + 60s polish). We don't run the full
600s polish or the second canonical eval — those are too slow on M3.

If a Sinkhorn variant beats topk on +60s polish proxy, run test_ibm17.py
for the full 600s polish + canonical confirmation.
"""
from __future__ import annotations

import json
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

# Avoid the heavy canonical eval inside the V3 placer's place() method.
# Instead: run descent + legalize manually, then a single canonical eval
# at the very end, OR even just CD polish's final state.

from smooth_global_placer_v3_sinkhorn import (
    SmoothGlobalPlacerV3Sinkhorn,
    DiffProxyV3Sinkhorn,
    loss_with_penalty_v3_sinkhorn,
)
from macro_place.cd_core import project_overlaps, sdf_init
from macro_legalizer import greedy_macro_legalize


def descend_and_legalize_only(placer, benchmark, plc):
    """Run descend + legalize but skip the inner canonical proxy eval."""
    t0 = time.time()
    pos, descend_stats = placer.descend(benchmark, plc)
    descend_wall = time.time() - t0
    t1 = time.time()
    legal_pos, leg_stats = placer.legalize(pos, benchmark)
    legalize_wall = time.time() - t1
    return legal_pos, {
        "descend_wall_s": descend_wall,
        "legalize_wall_s": legalize_wall,
        "final_overlaps": leg_stats["final_overlaps"],
        "final_smooth": descend_stats["final_smooth"],
    }


def cd_polish(pos, benchmark, plc, budget_s=60.0):
    t0 = time.time()
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl > 0:
        return pos, float("inf"), 0.0, 0
    placement_f64 = pos.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
    stats = run_cd_adaptive(
        evaluator=evaluator, benchmark=benchmark, plc=plc, movable=movable,
        min_time_s=0.0, hard_cap_s=budget_s, patience=3, plateau_threshold=1e-4,
        log_fn=None,
    )
    polished = evaluator.placement.clone().to(torch.float32)
    # Use evaluator's tracked proxy (fast, equivalent to canonical at this point
    # because incremental matches canonical when no rounding errors accumulate).
    final_proxy = float(evaluator.current_cost()["proxy"])
    return polished, final_proxy, time.time() - t0, stats.get("sweeps", 0)


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded {bench}: {benchmark.num_macros} macros ({benchmark.num_hard_macros} hard)")

    base_cfg = dict(
        num_steps=500,
        lr_frac=0.005,
        init="sdf",
        overlap_lambda_end=10.0,
        verbose=True,
        log_every=100,
    )

    configs = [
        ("topk_ablation",                {"use_sinkhorn": False}),
        ("sinkhorn_eps_0.5",             {"use_sinkhorn": True, "sinkhorn_eps": 0.5, "sinkhorn_iters": 100}),
        ("sinkhorn_eps_0.3",             {"use_sinkhorn": True, "sinkhorn_eps": 0.3, "sinkhorn_iters": 80}),
        ("sinkhorn_anneal_0.5_to_0.1",   {"use_sinkhorn": True, "sinkhorn_eps": 0.5, "sinkhorn_eps_end": 0.1, "sinkhorn_iters": 100}),
    ]

    results = []
    for label, extra in configs:
        print(f"\n=== {label} ===", flush=True)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        placer = SmoothGlobalPlacerV3Sinkhorn(**base_cfg, **extra)
        legal_pos, stats = descend_and_legalize_only(placer, benchmark, plc)
        print(f"  legal: smooth_at_end={stats['final_smooth']:.5f} "
              f"ovl={stats['final_overlaps']} "
              f"descend={stats['descend_wall_s']:.0f}s "
              f"legalize={stats['legalize_wall_s']:.0f}s", flush=True)
        # CD polish 60s
        polished, polish_proxy, polish_wall, sweeps = cd_polish(legal_pos, benchmark, plc, 60.0)
        print(f"  CD60s: proxy={polish_proxy:.5f} sweeps={sweeps} wall={polish_wall:.0f}s", flush=True)
        results.append({
            "label": label, "config": extra,
            "final_smooth": stats["final_smooth"],
            "final_overlaps": stats["final_overlaps"],
            "cd60_proxy": polish_proxy,
            "descend_s": stats["descend_wall_s"],
            "legalize_s": stats["legalize_wall_s"],
            "polish_s": polish_wall,
            "sweeps": sweeps,
        })

    print(f"\n=== SUMMARY ({bench}) ===")
    print(f"  {'label':25s} {'smooth':>9s} {'cd60_proxy':>11s} {'ovl':>4s} {'wall':>6s}")
    for r in results:
        wall = r["descend_s"] + r["legalize_s"] + r["polish_s"]
        print(f"  {r['label']:25s} {r['final_smooth']:>9.5f} {r['cd60_proxy']:>11.5f} "
              f"{r['final_overlaps']:>4d} {wall:>5.0f}s")

    out = _HERE.parent / "results" / f"fast_smoke_{bench}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
