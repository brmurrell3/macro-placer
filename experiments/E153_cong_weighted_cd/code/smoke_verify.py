"""E153 smoke verify — focused test on x3.0 cong-weight to verify lift.

Validates:
  1. Final placement has zero overlaps
  2. Canonical proxy improves with proper CD C polish
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    bench_name = "ibm04"
    print(f"=== E153 verify x3.0 on {bench_name} ===", flush=True)

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    t0 = time.time()
    pos = sdf_init(benchmark)
    pos, _ = project_overlaps(pos, benchmark)
    print(f"  init proxy={float(compute_proxy_cost(pos, benchmark, plc)['proxy_cost']):.5f} wall={time.time()-t0:.0f}s", flush=True)

    movable = [i for i in range(benchmark.num_macros)
               if not bool(benchmark.macro_fixed[i])]

    # CD A canonical
    evA = IncrementalProxyEvaluator(benchmark, plc, pos)
    run_cd_adaptive(evA, benchmark, plc, movable,
                    min_time_s=120.0, hard_cap_s=200.0,
                    patience=5, plateau_threshold=0.001, log_fn=None)
    posA = evA.placement.detach().clone().to(torch.float32)
    detA = compute_proxy_cost(posA, benchmark, plc)
    ovlA = compute_overlap_metrics(posA, benchmark)["overlap_count"]
    print(f"  CD A: proxy={detA['proxy_cost']:.5f} c={detA['congestion_cost']:.4f} ovl={ovlA} wall={time.time()-t0:.0f}s", flush=True)

    # CD B: cong weight x3.0
    evB = IncrementalProxyEvaluator(benchmark, plc, posA.clone())
    evB.weights = {"wirelength": 1.0, "density": 0.5, "congestion": 1.5}  # 0.5 * 3.0
    t1 = time.time()
    run_cd_adaptive(evB, benchmark, plc, movable,
                    min_time_s=60.0, hard_cap_s=150.0,
                    patience=5, plateau_threshold=0.001, log_fn=None)
    posB = evB.placement.detach().clone().to(torch.float32)
    detB = compute_proxy_cost(posB, benchmark, plc)
    ovlB = compute_overlap_metrics(posB, benchmark)["overlap_count"]
    print(f"  CD B x3.0 (canonical recomputed): proxy={detB['proxy_cost']:.5f} c={detB['congestion_cost']:.4f} ovl={ovlB} Δ_vs_A={(detB['proxy_cost']-detA['proxy_cost'])/detA['proxy_cost']*100:+.2f}% wall_B={time.time()-t1:.0f}s", flush=True)

    # CD C: re-polish with canonical weights
    evC = IncrementalProxyEvaluator(benchmark, plc, posB.clone())
    t2 = time.time()
    run_cd_adaptive(evC, benchmark, plc, movable,
                    min_time_s=60.0, hard_cap_s=150.0,
                    patience=5, plateau_threshold=0.001, log_fn=None)
    posC = evC.placement.detach().clone().to(torch.float32)
    detC = compute_proxy_cost(posC, benchmark, plc)
    ovlC = compute_overlap_metrics(posC, benchmark)["overlap_count"]
    delta = (detC['proxy_cost'] - detA['proxy_cost']) / detA['proxy_cost'] * 100
    print(f"  CD C (canonical re-polish): proxy={detC['proxy_cost']:.5f} c={detC['congestion_cost']:.4f} ovl={ovlC} Δ_vs_A={delta:+.2f}% wall_C={time.time()-t2:.0f}s", flush=True)

    print(f"  TOTAL wall={time.time()-t0:.0f}s — proxyA={detA['proxy_cost']:.5f} proxyC={detC['proxy_cost']:.5f} lift={delta:+.2f}%", flush=True)
    if ovlC != 0:
        print(f"  WARNING: final placement has {ovlC} overlaps", flush=True)


if __name__ == "__main__":
    main()
