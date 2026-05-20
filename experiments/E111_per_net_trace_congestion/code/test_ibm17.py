"""E111 — Smoke test on ibm17 (the worst-case bench).

Run:
  uv run python experiments/E111_per_net_trace_congestion/code/test_ibm17.py

Compares:
  1. E110 SmoothGlobalPlacer (bbox-uniform)         — baseline
  2. E111 SmoothGlobalPlacerV3 (per-net trace)      — variant
And then 60s CD polish on each via IncrementalProxyEvaluator + run_cd_adaptive.

Success: V3 raw < 1.50 (vs E110 ~1.74), V3 + CD60s < 1.40 (vs E110 ~1.46).
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

from smooth_global_placer import SmoothGlobalPlacer
from smooth_global_placer_v3 import SmoothGlobalPlacerV3


BENCH = "ibm17"
CD_BUDGET_S = 60.0


def cd_polish(pos: torch.Tensor, benchmark, plc, budget_s: float):
    """Polish a legal placement with the CD adaptive driver.

    Returns (polished_pos, final_proxy, wall_s, sweeps).
    """
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
    print(f"  RAW after smooth+legalize: proxy={proxy_raw:.5f} ovl={ovl_raw} wall={wall_smooth:.1f}s", flush=True)

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
    print(f"Loaded {BENCH}: {benchmark.num_macros} macros "
          f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}, "
          f"canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
          flush=True)

    # Use same config as E110 default for a fair head-to-head.
    cfg = dict(num_steps=500, lr_frac=0.005, init="sdf",
               overlap_lambda_end=50.0, verbose=True, log_every=100)

    results = []

    # 1) E110 baseline (bbox-uniform)
    placer_e110 = SmoothGlobalPlacer(**cfg)
    results.append(run_one("E110 SmoothGlobalPlacer (bbox-uniform)", placer_e110, benchmark, plc))

    # Reload PLC to reset incremental state (each placer mutates the plc).
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) E111 V3 (per-net trace)
    placer_e111 = SmoothGlobalPlacerV3(**cfg)
    results.append(run_one("E111 SmoothGlobalPlacerV3 (per-net trace)", placer_e111, benchmark, plc))

    # Summary
    print("\n\n=== SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['label']:50s} raw={r['raw_proxy']:.5f}  polish={r['polish_proxy']:.5f}  "
              f"raw_wall={r['raw_wall_s']:.0f}s  polish_wall={r['polish_wall_s']:.0f}s",
              flush=True)

    print(f"\nReference: lane-4 default --all ibm17 = 1.32360 (target to beat)", flush=True)

    out_path = _HERE.parent / "results" / "ibm17_smoke.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
