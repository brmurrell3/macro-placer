"""E96: End-to-end multi-DP + full-polish driver.

Pipeline:
  1. Run K DP configs in sequence (multi_dp_basin).
  2. Pick best by legalize_proxy (overlap-penalty selection).
  3. Run full B-R0' polish on winner: CD + LNS + SA + cascade saddle.
  4. Save final placement + metrics.

Total wall: K × ~30s DP + 5s legalize + budget_s for polish.
Default K=4, budget_s=3300 → ~3 + 50 min wall per bench.

Comparison targets:
  - B-R0' single-DP polished_proxy (the baseline this should beat)
  - Cascade-capped proxy (the gold standard)

Usage:
  cd ~/macro-place-challenge-2026
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \\
    DP_DOCKER_IMAGE=dreamplace:custom DP_USE_GPU=0 \\
    python3 experiments/E96_multi_config_dp/code/multi_dp_polish.py ibm10 4 3000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
_E91 = _ROOT / "experiments" / "E91_dp_full_polish" / "code"
sys.path.insert(0, str(_E91))
sys.path.insert(0, str(_HERE))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from dp_full_polish import run_full_polish_on_init
from multi_dp_basin import multi_dp_basin


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    budget_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3300.0

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    print(f"\n=== E96 multi-DP polish: {bench_name} (K={K}, budget={budget_s:.0f}s) ===", flush=True)
    print(f"  n_macros={bench.num_macros}, n_hard={bench.num_hard_macros}", flush=True)

    t_total0 = time.time()

    # Phase 1: Multi-DP basin selection
    deadline = t_total0 + budget_s
    raw, legal, basin_stats = multi_dp_basin(
        bench, plc, K=K, deadline=deadline,
        log=lambda s: print(s, flush=True),
    )
    basin_wall = time.time() - t_total0
    if legal is None:
        print(f"  ABORT: all DP configs failed", flush=True)
        return

    legal_proxy = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    legal_ovl = int(compute_overlap_metrics(legal, bench)["overlap_count"])
    print(f"\n  multi-DP done in {basin_wall:.0f}s; winner={basin_stats['winner_label']} "
          f"legal_proxy={legal_proxy:.5f} ovl={legal_ovl}", flush=True)

    # Phase 2-5: Full polish on winner
    remaining = budget_s - basin_wall
    cd_cap = remaining * 0.22  # slightly more CD since we used less time on basin
    lns_cap = remaining * 0.07
    sa_cap = remaining * 0.07
    cascade_cap = remaining * 0.62
    print(f"\n  polish budgets: CD={cd_cap:.0f}s LNS={lns_cap:.0f}s SA={sa_cap:.0f}s "
          f"cascade={cascade_cap:.0f}s (remaining={remaining:.0f}s)", flush=True)

    polish_result = run_full_polish_on_init(
        legal, bench, plc,
        cd_hard_cap_s=cd_cap,
        lns_budget_s=lns_cap,
        sa_budget_s=sa_cap,
        cascade_budget_s=cascade_cap,
        log=lambda s: print(s, flush=True),
    )
    total_wall = time.time() - t_total0

    final_state = polish_result.pop("final_state")
    out = {
        **polish_result,
        "bench": bench_name,
        "K": K,
        "budget_s": budget_s,
        "total_wall": total_wall,
        "basin_wall": basin_wall,
        "legal_proxy": legal_proxy,
        "legal_ovl": legal_ovl,
        "basin_stats": basin_stats,
    }

    out_path = _HERE.parent / "results" / f"multi_dp_polish_{bench_name}_K{K}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    pt_out = _HERE.parent / "results" / f"multi_dp_polish_{bench_name}_K{K}.pt"
    torch.save({"placement": final_state, "out": out}, pt_out)

    print(f"\n=== {bench_name} MULTI-DP POLISH RESULT (K={K}) ===")
    print(f"  winner config  : {basin_stats['winner_label']}")
    print(f"  legal          : {legal_proxy:.5f}")
    print(f"  + CD adaptive  : {polish_result['cd_proxy']:.5f}")
    print(f"  + LNS gridbin  : {polish_result['lns_proxy']:.5f}")
    print(f"  + SA-v2        : {polish_result['sa_proxy']:.5f}")
    print(f"  + cascade      : {polish_result['final_proxy']:.5f} (ovl={polish_result['overlap_count']})")
    print(f"  basin_wall     : {basin_wall:.0f}s")
    print(f"  total_wall     : {total_wall:.0f}s")

    # Comparisons
    cascade_capped = {
        "ibm10": 1.0775, "ibm12": 1.3031, "ibm14": 1.2919, "ibm17": 1.4546,
    }
    b_r0_single = {
        "ibm01": 0.862, "ibm09": 0.789,
        "ibm10": 1.095, "ibm12": 1.129, "ibm14": 1.243, "ibm17": 1.307,
    }
    fp = polish_result["final_proxy"]
    if bench_name in cascade_capped:
        ref = cascade_capped[bench_name]
        delta = (fp - ref) / ref * 100
        print(f"  vs cascade-capped ({ref:.4f}): {delta:+.2f}%  "
              f"({'WIN' if delta < 0 else 'lose'})")
    if bench_name in b_r0_single:
        ref = b_r0_single[bench_name]
        delta = (fp - ref) / ref * 100
        print(f"  vs B-R0' single-DP ({ref:.4f}): {delta:+.2f}%  "
              f"({'WIN' if delta < 0 else 'lose'})")
    print(f"  results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
