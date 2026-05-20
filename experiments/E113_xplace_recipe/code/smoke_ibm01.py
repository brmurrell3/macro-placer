"""E113 — Smoke test on ibm01: V3 (Adam) vs V4 (Xplace recipe).

Runs:
  1. V3 default       (Adam, linear gamma anneal, linear λ ramp)
  2. V4 default       (Nesterov-BB, overflow gamma, HPWL-feedback λ)
  3. V4 simple-mom    (Heavy-ball, same adaptive λ/γ) — fallback if BB unstable

Then 60s CD polish on each. Targets:
  - V4 raw < 0.95 (V3 raw on ibm01 ~ 0.895)
  - V4 + CD60s < 0.84 (V3 + CD120s = 0.846; if V4 reaches ≤0.84 we win)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

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
from smooth_global_placer_v4 import SmoothGlobalPlacerV4


BENCH = "ibm01"
CD_BUDGET_S = 60.0


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
    print(f"\n=== {label} ({BENCH}) ===", flush=True)
    t0 = time.time()
    pos = placer.place(benchmark)
    wall_smooth = time.time() - t0
    proxy_raw = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    ovl_raw = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(f"  RAW: proxy={proxy_raw:.5f} ovl={ovl_raw} wall={wall_smooth:.1f}s",
          flush=True)

    polished, proxy_polish, wall_polish, sweeps = cd_polish(
        pos, benchmark, plc, CD_BUDGET_S
    )
    print(f"  +CD60s: proxy={proxy_polish:.5f} sweeps={sweeps} wall={wall_polish:.1f}s",
          flush=True)
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

    cfg_common = dict(num_steps=500, init="sdf", verbose=True, log_every=100)

    results = []

    # 1) V3 baseline (Adam, linear schedules)
    placer_v3 = SmoothGlobalPlacerV3(lr_frac=0.005, **cfg_common)
    results.append(run_one("V3 (Adam + linear γ/λ)", placer_v3, benchmark, plc))

    # Reload to reset incremental state.
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) V4 with Nesterov-BB + overflow gamma + HPWL-feedback density weight
    placer_v4_bb = SmoothGlobalPlacerV4(
        base_lr_frac=0.005,
        overlap_lambda_init=0.5, overlap_lambda_max=100.0,
        gamma_start_frac=5e-3, gamma_end_frac=5e-5,
        gamma_base_frac=1e-3,
        use_overflow_gamma=True,
        nesterov_use_bb=True,
        **cfg_common,
    )
    results.append(
        run_one("V4 (Nesterov-BB + overflow γ + HPWL-fb λ)", placer_v4_bb, benchmark, plc)
    )

    # Reload to reset incremental state.
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 3) V4 with heavy-ball momentum (sanity check — should be ~Adam quality)
    placer_v4_hb = SmoothGlobalPlacerV4(
        base_lr_frac=0.005,
        overlap_lambda_init=0.5, overlap_lambda_max=100.0,
        gamma_start_frac=5e-3, gamma_end_frac=5e-5,
        use_overflow_gamma=True,
        nesterov_use_bb=False,
        **cfg_common,
    )
    results.append(
        run_one("V4 (Heavy-ball + overflow γ + HPWL-fb λ)", placer_v4_hb, benchmark, plc)
    )

    # Summary
    print("\n\n=== SUMMARY (ibm01) ===", flush=True)
    for r in results:
        print(
            f"  {r['label']:60s}  raw={r['raw_proxy']:.5f}  "
            f"+CD60s={r['polish_proxy']:.5f}  "
            f"wall={r['raw_wall_s']:.0f}+{r['polish_wall_s']:.0f}s",
            flush=True,
        )

    print(
        f"\nTargets:\n"
        f"  V4 raw   < 0.95  (V3 raw = {results[0]['raw_proxy']:.5f})\n"
        f"  V4 +CD60 < 0.84  (V3 +CD60 = {results[0]['polish_proxy']:.5f})",
        flush=True,
    )

    out_path = _HERE.parent / "results" / "smoke_ibm01.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
