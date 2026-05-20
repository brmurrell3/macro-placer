"""E113 — Test V5 (multi-stage Adam + margin) on ibm17.

Compares:
  1. V3 (Adam + linear γ/λ)                 — baseline
  2. V5 default (multi-stage, no margin)    — does multi-stage help?
  3. V5 + margin=0.002, λ_C=200             — does margin + high λ_C help?
  4. V5 + margin=0.005, λ_C=200             — slightly larger margin
  5. V5 + adaptive λ + margin=0.005         — adaptive density weight

All followed by 60s CD polish. Targets:
  - V5+margin raw < 1.30 (V3 raw on ibm17 = 1.304)
  - V5+margin +CD60s < 1.24 (V3 +CD60s on ibm17 = 1.246; if ≤1.20 great)
"""
from __future__ import annotations

import io
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
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from smooth_global_placer_v5 import SmoothGlobalPlacerV5


class quiet:
    def __enter__(self): self.old = sys.stdout; sys.stdout = io.StringIO(); return self
    def __exit__(self, *a): sys.stdout = self.old


BENCH = "ibm17"
CD_BUDGET_S = 60.0


def cd_polish(pos, benchmark, plc, budget_s):
    t0 = time.time()
    if compute_overlap_metrics(pos, benchmark)["overlap_count"] > 0:
        return pos, float("inf"), 0.0, 0
    ev = IncrementalProxyEvaluator(benchmark, plc, pos.detach().clone().to(torch.float64))
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    stats = run_cd_adaptive(
        ev, benchmark, plc, movable,
        min_time_s=0.0, hard_cap_s=budget_s,
        patience=3, plateau_threshold=1e-4, log_fn=None,
    )
    polished = ev.placement.clone().to(torch.float32)
    return polished, float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"]), \
           time.time() - t0, stats.get("sweeps", 0)


def run_one(label, placer, benchmark, plc):
    print(f"\n=== {label} ({BENCH}) ===", flush=True)
    t0 = time.time()
    with quiet():
        pos = placer.place(benchmark)
    wall_smooth = time.time() - t0
    proxy_raw = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    ovl_raw = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(f"  RAW: proxy={proxy_raw:.5f} ovl={ovl_raw} wall={wall_smooth:.1f}s", flush=True)

    polished, proxy_polish, wall_polish, sweeps = cd_polish(pos, benchmark, plc, CD_BUDGET_S)
    print(f"  +CD60s: proxy={proxy_polish:.5f} sweeps={sweeps} wall={wall_polish:.1f}s", flush=True)
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
        f"Loaded {BENCH}: {benchmark.num_macros} macros ({benchmark.num_hard_macros} hard), "
        f"nets={benchmark.num_nets}, canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
        flush=True,
    )

    results = []

    placer_v3 = SmoothGlobalPlacerV3(num_steps=500, lr_frac=0.005, init="sdf", verbose=False)
    results.append(run_one("V3 baseline (Adam + linear γ/λ)", placer_v3, benchmark, plc))

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer_v5 = SmoothGlobalPlacerV5(
        stage_steps=(250, 200, 150),
        base_lr_frac=0.005, verbose=False,
    )
    results.append(run_one("V5 multi-stage (no margin)", placer_v5, benchmark, plc))

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer_v5_m002_λ200 = SmoothGlobalPlacerV5(
        stage_steps=(250, 200, 150),
        base_lr_frac=0.005, verbose=False,
        overlap_lambda_stage_C=(50.0, 200.0),
        overlap_margin_frac=0.002,
    )
    results.append(run_one("V5 margin=0.002 λ_C=200", placer_v5_m002_λ200, benchmark, plc))

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer_v5_m005_λ200 = SmoothGlobalPlacerV5(
        stage_steps=(250, 200, 150),
        base_lr_frac=0.005, verbose=False,
        overlap_lambda_stage_C=(50.0, 200.0),
        overlap_margin_frac=0.005,
    )
    results.append(run_one("V5 margin=0.005 λ_C=200", placer_v5_m005_λ200, benchmark, plc))

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer_v5_adaptive = SmoothGlobalPlacerV5(
        stage_steps=(250, 200, 150),
        base_lr_frac=0.005, verbose=False,
        use_adaptive_lambda=True,
        adaptive_lambda_init=0.5, adaptive_lambda_max=200.0,
        overlap_margin_frac=0.005,
    )
    results.append(run_one("V5 adaptive λ + margin=0.005", placer_v5_adaptive, benchmark, plc))

    # Summary
    print("\n\n=== SUMMARY (ibm17) ===", flush=True)
    for r in results:
        print(
            f"  {r['label']:40s}  raw={r['raw_proxy']:.5f}  "
            f"+CD60s={r['polish_proxy']:.5f}  "
            f"wall={r['raw_wall_s']:.0f}+{r['polish_wall_s']:.0f}s",
            flush=True,
        )
    print(
        f"\nTargets vs V3 baseline:\n"
        f"  Raw   < 1.30  (V3 raw = {results[0]['raw_proxy']:.5f})\n"
        f"  +CD60 < 1.24  (V3 +CD60 = {results[0]['polish_proxy']:.5f})",
        flush=True,
    )
    out_path = _HERE.parent / "results" / "test_ibm17.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
