"""Polish saved descent positions on ibm17 with CD600s."""
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


def main(file_arg=None):
    if file_arg is None:
        file_arg = str(_HERE.parent / "results" / "ibm17_descent_gaussian_s1_0.pt")
    print(f"Loading positions from {file_arg}", flush=True)
    data = torch.load(file_arg)
    pos = data["positions"]

    bench_name = "ibm17"
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"bench={benchmark.name} num_macros={benchmark.num_macros}", flush=True)

    # Pre-polish: report current canonical proxy
    canon_pre = compute_proxy_cost(pos, benchmark, plc)
    ovl_pre = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    print(
        f"  PRE-POLISH: proxy={canon_pre['proxy_cost']:.5f} "
        f"wl={canon_pre['wirelength_cost']:.4f} d={canon_pre['density_cost']:.4f} "
        f"c={canon_pre['congestion_cost']:.4f} ovl={ovl_pre}",
        flush=True,
    )

    # CD polish
    cd_budget = 600.0
    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
    movable = [i for i in range(benchmark.num_macros)
               if not bool(benchmark.macro_fixed[i])]
    t0 = time.time()
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=cd_budget * 0.5,
        hard_cap_s=cd_budget,
        patience=5,
        plateau_threshold=0.001,
        log_fn=None,
    )
    cd_wall = time.time() - t0

    final = evaluator.placement.detach().clone().to(torch.float32)
    canon_post = compute_proxy_cost(final, benchmark, plc)
    ovl_post = compute_overlap_metrics(final, benchmark)["overlap_count"]
    print(
        f"  POST-POLISH: proxy={canon_post['proxy_cost']:.5f} "
        f"wl={canon_post['wirelength_cost']:.4f} d={canon_post['density_cost']:.4f} "
        f"c={canon_post['congestion_cost']:.4f} ovl={ovl_post} cd_wall={cd_wall:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
