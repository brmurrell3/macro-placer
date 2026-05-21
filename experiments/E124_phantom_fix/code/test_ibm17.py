"""E124 — Phantom-stripe fix head-to-head on ibm17.

Mirrors the V3Min ovl10 720s config (descent ovl_lam_end=10 →
greedy_legalize → CD polish 600s). Runs two placers back-to-back:

  1. V3 baseline (E111 buggy soft_min_max) — proxy 1.20032 per spec
  2. V3 phantom-fixed (E124 exact min/max)

Decision rules:
  fixed < 1.195   → WIN, run --fast next
  fixed in [1.195, 1.205] → NEUTRAL, ship if zero overlaps
  fixed > 1.205   → FALSIFY (fidelity-trap pattern even for bug fixes)

Wall budget per placer: ~720s (~360s descent + ~360s CD polish; in
practice descent is ~30-60s and CD takes the rest).
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

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from smooth_global_placer_v3_fixed import SmoothGlobalPlacerV3Fixed


BENCH = "ibm17"

# V3Min ovl10 720s spec:
#   descent ovl_lam_end=10, num_steps=500 (default)
#   CD polish 600s after legalize
DESCENT_KWARGS = dict(
    num_steps=500,
    lr_frac=0.005,
    gamma_start_frac=5e-3,
    gamma_end_frac=5e-5,
    overlap_lambda_end=10.0,
    overlap_ramp_pct=0.7,
    init="sdf",
    rng_seed=42,
    verbose=False,
)
CD_POLISH_S = 600.0


def cd_polish(pos: torch.Tensor, benchmark, plc, budget_s: float):
    t0 = time.time()
    ovl_before = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl_before > 0:
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
        min_time_s=budget_s * 0.5,
        hard_cap_s=budget_s,
        patience=5,
        plateau_threshold=1e-3,
        log_fn=None,
    )
    polished = evaluator.placement.clone().to(torch.float32)
    final_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    wall = time.time() - t0
    return polished, final_proxy, wall, stats.get("sweeps", 0)


def run_one(label: str, placer_cls, benchmark, plc):
    print(f"\n=== {label} on {BENCH} ===", flush=True)
    t0 = time.time()
    placer = placer_cls(**DESCENT_KWARGS)
    pos = placer.place(benchmark)
    wall_smooth = time.time() - t0
    proxy_raw = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    ovl_raw = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(f"  RAW after smooth+legalize: proxy={proxy_raw:.5f} ovl={ovl_raw} "
          f"wall={wall_smooth:.1f}s", flush=True)

    polished, proxy_polish, wall_polish, sweeps = cd_polish(pos, benchmark, plc, CD_POLISH_S)
    print(f"  POLISHED (CD {CD_POLISH_S:.0f}s, {sweeps} sweeps): "
          f"proxy={proxy_polish:.5f} wall={wall_polish:.1f}s", flush=True)
    ovl_pol = compute_overlap_metrics(polished, benchmark)["overlap_count"]

    return {
        "label": label,
        "raw_proxy": proxy_raw,
        "polish_proxy": proxy_polish,
        "raw_wall_s": wall_smooth,
        "polish_wall_s": wall_polish,
        "raw_overlaps": ovl_raw,
        "polish_overlaps": ovl_pol,
        "cd_sweeps": sweeps,
    }


def main():
    bench_dir = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded {BENCH}: {benchmark.num_macros} macros "
          f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}, "
          f"canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
          flush=True)

    results = []

    # 1) V3 BASELINE (buggy soft_min_max — E111 production)
    results.append(run_one("V3 baseline (E111 buggy soft_min_max)",
                           SmoothGlobalPlacerV3, benchmark, plc))

    # Reload PLC to reset incremental state
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) V3 FIXED (E124 phantom-stripe fix)
    results.append(run_one("V3 fixed (E124 exact min/max)",
                           SmoothGlobalPlacerV3Fixed, benchmark, plc))

    print("\n\n=== SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['label']:55s} raw={r['raw_proxy']:.5f}  "
              f"polish={r['polish_proxy']:.5f}  "
              f"raw_wall={r['raw_wall_s']:.0f}s  "
              f"polish_wall={r['polish_wall_s']:.0f}s  "
              f"ovl_pol={r['polish_overlaps']}", flush=True)

    baseline_pol = results[0]["polish_proxy"]
    fixed_pol = results[1]["polish_proxy"]
    delta_abs = fixed_pol - baseline_pol
    delta_pct = (fixed_pol - baseline_pol) / baseline_pol * 100.0
    print(f"\nDelta (fixed - baseline): {delta_abs:+.5f} ({delta_pct:+.2f}%)", flush=True)
    print(f"V3Min ovl10 720s spec: 1.20032", flush=True)

    if fixed_pol < 1.195:
        print("DECISION: WIN — run --fast next", flush=True)
    elif fixed_pol <= 1.205:
        print("DECISION: NEUTRAL — ship if zero overlaps everywhere", flush=True)
    else:
        print("DECISION: FALSIFY — fidelity-trap pattern even for bug fixes", flush=True)

    out_path = _HERE.parent / "results" / "ibm17_phantom_fix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
