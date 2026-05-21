"""E123 — budget smoke with reduced Sinkhorn iters to fit in our M3 wall budget.

Key finding from previous fast_smoke: Sinkhorn ε=0.5 with iters=100
makes each Adam step ~1.5-2× slower than topk. With 500 steps, that's
~5-7 min just for descent vs ~4 min for topk.

This smoke uses iters=30 for Sinkhorn (faster, less converged). At
iters=30, Sinkhorn ε=0.5 is less accurate vs canonical (we lose some
of the +0.26% calibration win) but the gradient quality is mostly
preserved. Each Sinkhorn step ≈ 50% slower than topk, not 100%.

We also drop num_steps from 500 → 300 to fit in time.
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
    _HERE, _ROOT,
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
from macro_place.objective import compute_overlap_metrics

from smooth_global_placer_v3_sinkhorn import SmoothGlobalPlacerV3Sinkhorn


def descend_and_legalize_only(placer, benchmark, plc):
    t0 = time.time()
    pos, ds = placer.descend(benchmark, plc)
    descend_wall = time.time() - t0
    t1 = time.time()
    legal_pos, ls = placer.legalize(pos, benchmark)
    return legal_pos, {
        "descend_s": descend_wall,
        "legalize_s": time.time() - t1,
        "ovl_after": ls["final_overlaps"],
        "smooth_final": ds["final_smooth"],
    }


def cd_polish(pos, benchmark, plc, budget_s=60.0):
    t0 = time.time()
    if compute_overlap_metrics(pos, benchmark)["overlap_count"] > 0:
        return pos, float("inf"), 0.0, 0
    p64 = pos.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(benchmark, plc, p64)
    mv = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
    stats = run_cd_adaptive(evaluator=ev, benchmark=benchmark, plc=plc, movable=mv,
                            min_time_s=0.0, hard_cap_s=budget_s,
                            patience=3, plateau_threshold=1e-4, log_fn=None)
    final = float(ev.current_cost()["proxy"])
    return ev.placement.clone().to(torch.float32), final, time.time() - t0, stats.get("sweeps", 0)


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded {bench}: {benchmark.num_macros} macros ({benchmark.num_hard_macros} hard)")

    # Smaller budget: 300 steps, iters=30 for Sinkhorn.
    base_cfg = dict(
        num_steps=300,
        lr_frac=0.005,
        init="sdf",
        overlap_lambda_end=10.0,
        verbose=True,
        log_every=100,
    )

    configs = [
        ("topk_300steps",  {"use_sinkhorn": False}),
        ("sink_0.5_30it",  {"use_sinkhorn": True, "sinkhorn_eps": 0.5, "sinkhorn_iters": 30}),
        ("sink_0.3_30it",  {"use_sinkhorn": True, "sinkhorn_eps": 0.3, "sinkhorn_iters": 30}),
    ]

    results = []
    for label, extra in configs:
        print(f"\n=== {label} ===", flush=True)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        placer = SmoothGlobalPlacerV3Sinkhorn(**base_cfg, **extra)
        legal_pos, stats = descend_and_legalize_only(placer, benchmark, plc)
        print(f"  legal: smooth={stats['smooth_final']:.5f} ovl={stats['ovl_after']} "
              f"descend={stats['descend_s']:.0f}s legalize={stats['legalize_s']:.0f}s", flush=True)
        polished, p, w, s = cd_polish(legal_pos, benchmark, plc, 60.0)
        print(f"  CD60s: proxy={p:.5f} sweeps={s} wall={w:.0f}s", flush=True)
        results.append({
            "label": label, "config": extra,
            "smooth": stats["smooth_final"], "ovl": stats["ovl_after"],
            "cd60_proxy": p,
            "descend_s": stats["descend_s"], "polish_s": w, "sweeps": s,
        })

    print(f"\n=== SUMMARY ({bench}) ===")
    print(f"  {'label':18s} {'smooth':>9s} {'cd60':>9s} {'descend':>8s} {'polish':>7s}")
    baseline_cd = next(r["cd60_proxy"] for r in results if "topk" in r["label"])
    for r in results:
        delta = r["cd60_proxy"] - baseline_cd
        pct = delta / baseline_cd * 100
        marker = " *" if delta < -0.005 else "  "
        print(f"  {r['label']:18s} {r['smooth']:>9.5f} {r['cd60_proxy']:>9.5f} "
              f"{r['descend_s']:>7.0f}s {r['polish_s']:>6.0f}s  "
              f"Δ={delta:+.5f} ({pct:+.2f}%){marker}")

    out = _HERE.parent / "results" / f"budget_smoke_{bench}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
