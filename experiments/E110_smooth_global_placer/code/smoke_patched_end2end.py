"""End-to-end smoke test for cd_lns_sa_cascade_dp_lane_patched on ibm01.

Tests the full placer pipeline (all 5 lanes + cascade saddle) with a
realistic-but-shorter budget than production (15 min vs 55 min).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path('/Users/brendan/Developer/macro-place-challenge-2026')
sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from submissions.cd_lns_sa_cascade_dp_lane_patched.placer import (
    CDLNSSACascadeDPLanePatchedPlacer,
)


def main():
    bench_dir = find_benchmark_dir('ibm01')
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f'ibm01: {benchmark.num_macros} macros', flush=True)

    placer = CDLNSSACascadeDPLanePatchedPlacer(
        budget_seconds=900.0,  # 15 min
        verbose=True,
    )
    t0 = time.time()
    pos = placer.place(benchmark)
    final_proxy = float(compute_proxy_cost(pos, benchmark, plc)['proxy_cost'])
    final_ovl = compute_overlap_metrics(pos, benchmark)['overlap_count']
    print(
        f'\n=== END-TO-END ibm01 (900s budget): '
        f'proxy={final_proxy:.5f} ovl={final_ovl} '
        f'wall={time.time()-t0:.1f}s ===',
        flush=True,
    )


if __name__ == '__main__':
    main()
