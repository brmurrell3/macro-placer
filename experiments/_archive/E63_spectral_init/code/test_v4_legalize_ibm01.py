"""Quick test: V4 fixed-aware row-pack on ibm01.

Goal: verify residual overlaps drop from 111 (V3) to ≤ a project_overlaps-
repairable level (typically <20).
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import torch

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from experiments.E63_spectral_init.code.cd_lns_sa_spectral_kjoint import (
    _spectral_init,
)

bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
bench_dir = find_benchmark_dir(bench_name)
benchmark, plc = load_benchmark_from_dir(str(bench_dir))

print(f"=== Testing V4 spectral legalization on {bench_name} ===")
print(f"  num_macros={benchmark.num_macros} n_hard={benchmark.num_hard_macros} "
      f"n_fixed={int(benchmark.macro_fixed.sum())}")
print(f"  canvas={benchmark.canvas_width:.1f} x {benchmark.canvas_height:.1f}")

t0 = time.perf_counter()
placement = _spectral_init(benchmark, plc, seed=42, log_fn=lambda s: print(s, flush=True))
print(f"  spectral_init wall = {time.perf_counter() - t0:.1f}s")

# Project overlaps to clean up any residuals.
placement, proj_iters = project_overlaps(placement, benchmark)
ovl = compute_overlap_metrics(placement, benchmark)
print(f"  After project_overlaps: residuals={ovl['overlap_count']} (iters={proj_iters})")
print(f"  total_overlap_area={ovl['total_overlap_area']:.4f}")

if ovl["overlap_count"] == 0:
    print("OK — V4 legalization succeeded.")
    sys.exit(0)
else:
    print(f"FAIL — V4 left {ovl['overlap_count']} residual overlaps.")
    sys.exit(1)
