"""E117 — ibm17 +CD polish test.

Loads descent-only positions saved by test_ibm17_descent_only.py and
runs CD polish for 600s on each. Reports the polished canonical proxy.

Run AFTER test_ibm17_descent_only.py:
  uv run experiments/E117_gaussian_density/code/test_ibm17_cd_polish.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (_HERE, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def polish_one(label, pos_file, benchmark, plc, cd_budget=600.0):
    data = torch.load(pos_file)
    pos = data["positions"]
    print(f"\n=== {label} (loaded from {pos_file.name}) ===", flush=True)
    print(f"  pre-polish proxy={data['proxy']:.5f}", flush=True)

    # CD polish
    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    t0 = time.time()
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=cd_budget * 0.5,
        hard_cap_s=cd_budget,
        patience=5,
        plateau_threshold=0.001,
        log_fn=None,
    )
    wall = time.time() - t0

    final = evaluator.placement.detach().clone().to(torch.float32)
    canon = compute_proxy_cost(final, benchmark, plc)
    ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
    print(
        f"  POLISHED {label}: proxy={canon['proxy_cost']:.5f} "
        f"wl={canon['wirelength_cost']:.4f} d={canon['density_cost']:.4f} "
        f"c={canon['congestion_cost']:.4f} ovl={ovl} cd_wall={wall:.0f}s",
        flush=True,
    )
    return {
        "label": label, "proxy_pre": data['proxy'], "proxy": canon['proxy_cost'],
        "wl": canon['wirelength_cost'], "d": canon['density_cost'],
        "c": canon['congestion_cost'], "ovl": ovl, "cd_wall": wall,
    }


def main():
    bench_name = "ibm17"
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"bench={benchmark.name} num_macros={benchmark.num_macros}", flush=True)

    results_dir = _HERE.parent / "results"
    files = sorted(results_dir.glob("ibm17_descent_*.pt"))
    if not files:
        print(f"No descent results found in {results_dir}. Run test_ibm17_descent_only.py first.", flush=True)
        return

    out = []
    for f in files:
        label = f.stem.replace("ibm17_descent_", "")
        out.append(polish_one(label, f, benchmark, plc, cd_budget=600.0))

    print("\n\n=== ibm17 +CD600s POLISHED SUMMARY ===", flush=True)
    print(f"{'config':<25} {'pre':>9} {'post':>9} {'wl':>7} {'d':>7} {'c':>7} {'ovl':>4} {'cd_wall':>7}",
          flush=True)
    for r in out:
        print(f"{r['label']:<25} {r['proxy_pre']:>9.5f} {r['proxy']:>9.5f} "
              f"{r['wl']:>7.4f} {r['d']:>7.4f} {r['c']:>7.4f} "
              f"{r['ovl']:>4d} {r['cd_wall']:>6.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
