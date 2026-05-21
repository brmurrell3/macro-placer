"""E120 — head-to-head on ibm17 (the hardest IBM bench).

Compares:
  1. V3Min ovl10 720s baseline (E111MinimalPlacer with overlap_lambda_end=10)
  2. E120 ContinuousHessianSaddlePlacer at matched 720s budget

Success criterion (from manifest): E120 ibm17 + CD600s < 1.16, vs
V3Min baseline ~1.20.

Run:
  uv run python experiments/E120_continuous_saddle/code/test_ibm17.py
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
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "submissions" / "_archive" / "e111_minimal",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from continuous_saddle_placer import ContinuousHessianSaddlePlacer


BENCH = "ibm17"
BUDGET = 720.0


def main():
    bench_dir = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"Loaded {BENCH}: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}, "
        f"canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
        flush=True,
    )

    results = []

    # 1) E120 ContinuousHessianSaddlePlacer
    print(f"\n\n=== E120 ContinuousHessianSaddlePlacer on {BENCH} ===\n", flush=True)
    torch.manual_seed(42)
    placer_e120 = ContinuousHessianSaddlePlacer(
        budget_seconds=BUDGET,
        num_steps_phaseA=300,
        num_steps_resume=120,
        max_saddle_stages=2,
        overlap_lambda_end=10.0,
        cd_polish_s=300.0,
        eigsh_maxiter=300,
        verbose=True,
        log_every=100,
    )
    t0 = time.time()
    final_e120 = placer_e120.place(benchmark)
    proxy_e120 = float(compute_proxy_cost(final_e120, benchmark, plc)["proxy_cost"])
    wall_e120 = time.time() - t0
    print(f"\nE120 final: proxy={proxy_e120:.5f} wall={wall_e120:.0f}s", flush=True)
    results.append({
        "label": "E120 ContinuousHessianSaddle",
        "proxy": proxy_e120,
        "wall_s": wall_e120,
        "run_stats": placer_e120.last_run_stats,
    })

    # Reload plc state for the next run.
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) V3Min ovl10 720s baseline
    print(f"\n\n=== V3Min ovl10 720s baseline on {BENCH} ===\n", flush=True)
    from placer import E111MinimalPlacer
    torch.manual_seed(42)
    placer_v3min = E111MinimalPlacer(
        budget_seconds=BUDGET,
        num_steps=500,
        overlap_lambda_end=10.0,
        cd_polish_s=600.0,
        rng_seed=42,
        verbose=True,
    )
    t0 = time.time()
    final_v3 = placer_v3min.place(benchmark)
    proxy_v3 = float(compute_proxy_cost(final_v3, benchmark, plc)["proxy_cost"])
    wall_v3 = time.time() - t0
    print(f"\nV3Min final: proxy={proxy_v3:.5f} wall={wall_v3:.0f}s", flush=True)
    results.append({
        "label": "V3Min ovl10 720s baseline",
        "proxy": proxy_v3,
        "wall_s": wall_v3,
    })

    # Summary
    print(f"\n\n=== SUMMARY on {BENCH} ===", flush=True)
    for r in results:
        print(f"  {r['label']:40s}  proxy={r['proxy']:.5f}  wall={r['wall_s']:.0f}s",
              flush=True)
    delta = results[0]["proxy"] - results[1]["proxy"]
    pct = 100.0 * delta / results[1]["proxy"]
    print(f"\nE120 - V3Min = {delta:+.5f} ({pct:+.2f}%)", flush=True)

    out_path = _HERE.parent / "results" / "ibm17_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
