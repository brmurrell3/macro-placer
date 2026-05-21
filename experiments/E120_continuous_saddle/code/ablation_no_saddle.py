"""E120 — ablation: same config as E120 but max_saddle_stages=0.

This isolates whether the lift comes from the saddle escape vs from
the same code path running pure Adam descent + CD polish. Useful for
ruling out (a) we accidentally tuned a hyperparameter that matters
more than saddle escape itself.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from continuous_saddle_placer import ContinuousHessianSaddlePlacer


def main():
    BENCH = "ibm01"
    bench_dir = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # Saddle disabled.
    torch.manual_seed(42)
    placer = ContinuousHessianSaddlePlacer(
        budget_seconds=120.0,
        num_steps_phaseA=200,
        num_steps_resume=80,
        max_saddle_stages=0,  # ABLATION
        overlap_lambda_end=10.0,
        cd_polish_s=60.0,
        verbose=True,
        log_every=200,
    )
    t0 = time.time()
    pos = placer.place(benchmark)
    p = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    print(f"\n[E120 no-saddle] proxy={p:.5f} wall={time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
