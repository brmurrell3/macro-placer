"""E119 — Fast ibm17 head-to-head: V3 descent + 60s CD polish.

Compare default trace vs best trace on ibm17 with a SHORT budget.
Goal: get a directional answer in ~5 min wall (not 12).

If best beats default by >2% → suggests structural lift. Investigate further.
If best ties or loses → tuning is within noise.
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
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3  # noqa: E402

BEST_TRACE = dict(
    sigma_cell_frac=0.2,
    beta_minmax=16.0,
    beta_range_per_cell=10.0,
)


def run_pipeline(benchmark, plc, trace_kwargs, label, cd_budget_s=60.0):
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    descender = SmoothGlobalPlacerV3(
        num_steps=500,
        lr_frac=0.005,
        gamma_start_frac=5e-3,
        gamma_end_frac=5e-5,
        overlap_lambda_end=10.0,
        overlap_ramp_pct=0.7,
        init="sdf",
        rng_seed=42,
        verbose=False,
        trace_kwargs=trace_kwargs,
    )
    pos = descender.place(benchmark)
    ovl0 = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl0 > 0:
        pos, _ = project_overlaps(pos, benchmark)
        ovl0 = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    descent_wall = time.time() - t0
    descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    print(f"  V3+legalize: proxy={descent_proxy:.5f} ovl={ovl0} wall={descent_wall:.0f}s", flush=True)

    t1 = time.time()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=cd_budget_s * 0.5,
        hard_cap_s=cd_budget_s,
        patience=3,
        plateau_threshold=0.001,
        log_fn=None,
    )
    final = evaluator.placement.detach().clone().to(torch.float32)
    final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
    final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
    polish_wall = time.time() - t1
    print(f"  +CD{cd_budget_s:.0f}s: proxy={final_proxy:.5f} ovl={final_ovl} wall={polish_wall:.0f}s", flush=True)

    total = time.time() - t0
    return {
        "label": label,
        "trace_kwargs": trace_kwargs,
        "descent_proxy": descent_proxy,
        "descent_wall_s": descent_wall,
        "final_proxy": final_proxy,
        "final_ovl": final_ovl,
        "polish_wall_s": polish_wall,
        "total_wall_s": total,
    }


def main():
    bench = "ibm17"
    print(f"=== E119 fast ibm17 head-to-head (V3 + CD60s) ===", flush=True)
    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    results = []
    # 1) Default
    results.append(run_pipeline(benchmark, plc, trace_kwargs=None, label="DEFAULT (sr=2, σ=0.5, β_mm=6, β_rg=4)"))

    # Re-load plc to reset state
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) Best
    results.append(run_pipeline(benchmark, plc, trace_kwargs=BEST_TRACE, label="BEST (sr=2, σ=0.2, β_mm=16, β_rg=10)"))

    print(f"\n=== SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['label']:55s}: descent={r['descent_proxy']:.5f}  "
              f"+CD60s={r['final_proxy']:.5f}  wall={r['total_wall_s']:.0f}s", flush=True)
    d, b = results[0]["final_proxy"], results[1]["final_proxy"]
    delta_pct = (b - d) / d * 100
    print(f"\nBest vs Default: {delta_pct:+.2f}%", flush=True)
    if delta_pct < -2.0:
        print("VERDICT: structural lift; promote to V3Min ovl10 720s test", flush=True)
    elif delta_pct < 0.0:
        print("VERDICT: marginal lift (<2%); within noise. Run --all to confirm.", flush=True)
    else:
        print("VERDICT: no lift; defaults are at local optimum for the smooth proxy.", flush=True)

    out = _HERE.parent / "results" / "ibm17_fast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
