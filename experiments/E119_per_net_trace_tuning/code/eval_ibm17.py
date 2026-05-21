"""E119 — Test V3Min ovl10 720s with best per-net-trace hparams on ibm17.

Sweep winner (from `results/sweep_summary.md`):
  sr=2, σ=0.2, β_mm=16, β_rg=10
  - ibm10 +16.2%, ibm12 +17.3%, ibm17 +12.2%
  - max abs gap = 17.3% (default 23.7%)

Baseline (default sr=2/σ=0.5/β_mm=6/β_rg=4) ibm17 + V3Min ovl10 720s = 1.20.

This script:
  1. Construct V3Min ovl10 720s placer with `trace_kwargs={...best...}`
  2. Run on ibm17 only
  3. Compare to baseline 1.20

We pass `trace_kwargs` through to SmoothGlobalPlacerV3 (which forwards
to DiffProxyV3 → PerNetTraceCongestion). We also override
`plc.smooth_range` before constructing the placer so the proxy reads
the new sr (which is the only hparam not in trace_kwargs).

Run:
  uv run python experiments/E119_per_net_trace_tuning/code/eval_ibm17.py
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
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3  # noqa: E402


# Sweep winner from results/sweep_summary.md
BEST_TRACE = dict(
    sigma_cell_frac=0.2,
    beta_minmax=16.0,
    beta_range_per_cell=10.0,
    # NOTE: smooth_range is read from plc.smooth_range inside PerNetTraceCongestion;
    # we set it on plc before instantiation. The best smooth_range is 2 = default.
)
BEST_SMOOTH_RANGE = 2  # same as default; mainly the other hparams differ


def v3min_ovl10_720s_place(benchmark, plc, trace_kwargs=None, smooth_range=2):
    """Run V3Min ovl10 12-min pipeline with optional trace_kwargs.

    Equivalent to the production thinkorplace-v2 placer except with
    explicit `trace_kwargs` passing.
    """
    t0 = time.time()
    deadline = t0 + 720.0
    # Override plc.smooth_range so PerNetTraceCongestion uses our value
    prev_sr = getattr(plc, "smooth_range", 2)
    plc.smooth_range = int(smooth_range)
    try:
        # Phase 1: V3 smooth descent + greedy legalize
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=10.0),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", 500)
                ovl_end = cfg["overlap_lambda_end"]
                print(f"  V3 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}", flush=True)
                descender = SmoothGlobalPlacerV3(
                    num_steps=num_steps,
                    lr_frac=0.005,
                    gamma_start_frac=5e-3,
                    gamma_end_frac=5e-5,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=0.7,
                    init="sdf",
                    rng_seed=42 + attempt * 100,
                    verbose=False,
                    trace_kwargs=trace_kwargs,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos_try2, _ = project_overlaps(pos_try, benchmark)
                    ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                    if ovl2 == 0:
                        pos = pos_try2
                        print(f"  attempt {attempt+1}: ovl=0 (post-project)", flush=True)
                        break
                else:
                    pos_try2, _ = project_overlaps(pos_try, benchmark)
                    ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                    if ovl2 == 0:
                        pos = pos_try2
                        print(f"  attempt {attempt+1}: ovl=0 after project", flush=True)
                        break
                    print(f"  attempt {attempt+1}: still {ovl_try}/{ovl2} overlaps", flush=True)
            except Exception as exc:
                print(f"  attempt {attempt+1} EXCEPTION: {exc}", flush=True)
                continue
            if time.time() > deadline - 60:
                break

        if pos is None:
            print("  fallback: SDF + project_overlaps", flush=True)
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        print(f"  basin: proxy={descent_proxy:.5f} ovl={descent_ovl} wall={descent_wall:.0f}s", flush=True)

        # Phase 2: CD polish budget
        remaining = deadline - time.time() - 10.0
        cd_budget = max(30.0, min(remaining, 600.0))
        print(f"  CD polish budget={cd_budget:.0f}s", flush=True)

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5,
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=0.001,
            log_fn=None,
        )
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        print(f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} total_wall={time.time()-t0:.0f}s", flush=True)
        return final, final_proxy, final_ovl
    finally:
        plc.smooth_range = prev_sr


def main():
    bench = "ibm17"
    print(f"\n=== E119 eval V3Min ovl10 720s on {bench} with best trace_kwargs ===", flush=True)
    print(f"  best: sr={BEST_SMOOTH_RANGE}, "
          f"σ={BEST_TRACE['sigma_cell_frac']}, "
          f"β_mm={BEST_TRACE['beta_minmax']}, "
          f"β_rg={BEST_TRACE['beta_range_per_cell']}", flush=True)
    print(f"  baseline: 1.20 (default trace) | sweep predicts ibm17 gap: 12.2% (vs 14.4% default)", flush=True)
    t0 = time.time()

    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    final, proxy, ovl = v3min_ovl10_720s_place(
        benchmark, plc,
        trace_kwargs=BEST_TRACE,
        smooth_range=BEST_SMOOTH_RANGE,
    )

    result = {
        "bench": bench,
        "trace_kwargs": BEST_TRACE,
        "smooth_range": BEST_SMOOTH_RANGE,
        "proxy": proxy,
        "ovl": ovl,
        "wall_s": time.time() - t0,
        "baseline_default": 1.20,  # M3 V3Min ovl10 720s default trace
    }
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ibm17_best.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}", flush=True)

    delta_pct = (proxy - 1.20) / 1.20 * 100
    verdict = "WIN" if proxy < 1.18 else ("MARGINAL" if proxy < 1.20 else "NO LIFT")
    print(f"\nibm17 + V3Min ovl10 720s: proxy={proxy:.5f}  (baseline 1.20, Δ={delta_pct:+.2f}%)  → {verdict}", flush=True)


if __name__ == "__main__":
    main()
