"""E96: Polish a saved DP legal basin and compare to B-R0' single-DP baseline.

Loads a saved (raw, legal) DP basin from `multi_dp_basin.py`, runs full B-R0'
polish (CD-adaptive + LNS-gridbin + SA-v2 + cascading saddle), and writes the
result to JSON for comparison.

Usage:
  python3 experiments/E96_multi_config_dp/code/polish_basin.py \\
    experiments/E96_multi_config_dp/results/multi_dp_ibm10_K8.pt ibm10
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

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from dp_full_polish import run_full_polish_on_init


def main():
    pt_path = sys.argv[1]
    bench_name = sys.argv[2]
    budget_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3300.0

    data = torch.load(pt_path, weights_only=False)
    legal = data["legal_placement"]
    label = data.get("stats", {}).get("winner_label", "unknown")
    K = data.get("K", "?")

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    print(f"\n=== E96 polish: {bench_name} (label={label}, K={K}, budget={budget_s:.0f}s) ===", flush=True)
    print(f"  legal proxy: {float(compute_proxy_cost(legal, bench, plc)['proxy_cost']):.5f}", flush=True)
    print(f"  legal ovl: {int(compute_overlap_metrics(legal, bench)['overlap_count'])}", flush=True)

    t_total0 = time.time()
    cd_cap = budget_s * 0.20
    lns_cap = budget_s * 0.06
    sa_cap = budget_s * 0.06
    cascade_cap = budget_s * 0.60
    print(f"  budgets: CD={cd_cap:.0f}s LNS={lns_cap:.0f}s SA={sa_cap:.0f}s cascade={cascade_cap:.0f}s")

    polish_result = run_full_polish_on_init(
        legal, bench, plc,
        cd_hard_cap_s=cd_cap,
        lns_budget_s=lns_cap,
        sa_budget_s=sa_cap,
        cascade_budget_s=cascade_cap,
        log=print,
    )
    total_wall = time.time() - t_total0
    final_state = polish_result.pop("final_state")

    out_path = _HERE.parent / "results" / f"polish_{bench_name}_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    polish_result["bench"] = bench_name
    polish_result["winner_label"] = label
    polish_result["K"] = K
    polish_result["budget_s"] = budget_s
    polish_result["total_wall"] = total_wall
    polish_result["legal_proxy"] = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    polish_result["legal_ovl"] = int(compute_overlap_metrics(legal, bench)["overlap_count"])
    out_path.write_text(json.dumps(polish_result, indent=2))

    pt_out = _HERE.parent / "results" / f"polish_{bench_name}_{label}.pt"
    torch.save({"placement": final_state, "stats": polish_result}, pt_out)

    print(f"\n=== {bench_name} {label} POLISH RESULT ===")
    print(f"  legal           : {polish_result['legal_proxy']:.5f}")
    print(f"  + CD adaptive   : {polish_result['cd_proxy']:.5f}")
    print(f"  + LNS gridbin   : {polish_result['lns_proxy']:.5f}")
    print(f"  + SA-v2         : {polish_result['sa_proxy']:.5f}")
    print(f"  + cascade saddle: {polish_result['final_proxy']:.5f} (ovl={polish_result['overlap_count']})")
    print(f"  total wall: {total_wall:.0f}s")
    # Reference comparisons
    refs = {
        "ibm01": 0.85,
        "ibm10": 1.0775, "ibm12": 1.3031, "ibm14": 1.2919, "ibm17": 1.4546,
    }
    if bench_name in refs:
        ref = refs[bench_name]
        delta = (polish_result["final_proxy"] - ref) / ref * 100
        print(f"  vs cascade-capped ref ({ref}): {delta:+.2f}%")
    # B-R0' single-DP refs
    b_r0_refs = {
        "ibm01": 0.862, "ibm09": 0.789,
        "ibm10": 1.095, "ibm12": 1.129, "ibm14": 1.243, "ibm17": 1.307,
    }
    if bench_name in b_r0_refs:
        ref = b_r0_refs[bench_name]
        delta = (polish_result["final_proxy"] - ref) / ref * 100
        print(f"  vs B-R0' single-DP ref ({ref}): {delta:+.2f}%")
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
