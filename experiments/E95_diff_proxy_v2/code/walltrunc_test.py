"""E95 — test C1 on wall-truncated cascade output.

The cached cascade outputs in experiments/E84/results/ are from UNCAPPED
runs (1.0612 quality). C1 from those shows basin preservation but no
real lift, because the inputs are already canonical local minima.

The submission floor (1.137) is set by WALL-TRUNCATED cascade runs that
hit the 55 min cap. Those outputs have headroom — they're not at the
canonical local min yet. C1 might lift them.

This driver:
  1. Loads SDF init for a bench
  2. Runs E25 (CDLNSSAPlacer) for SHORT budget → simulates wall-cap
  3. Runs C1 (DiffProxyV2 + L-BFGS) from that output
  4. Compares C1-post vs (E25-only, E25 + cascade-uncapped)

If C1 lifts E25's truncated output by > 0.3 %, C1 has value as a
**post-processor on wall-capped cascade**. That's a deployable wedge
even if standalone C1 doesn't beat full cascade.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_E88_DIR = _ROOT / "experiments" / "E88_diff_proxy" / "code"
_E76_DIR = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
for p in (_ROOT, _E88_DIR, _E76_DIR, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from diff_proxy_v2 import DiffProxyV2
from spike_v2 import descend_lbfgs, evaluate_with_legalize

# Import E25 dynamically (it's at submissions/cd_lns_sa/placer.py).
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer


def run_walltrunc(bench_name: str, short_budget_s: int = 300):
    bench_dir = Path(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    print(f"\n========== {bench_name} (short budget {short_budget_s}s) ==========")

    # E25 with short budget to simulate wall-truncated cascade.
    # Allocate per-phase: ~70 % CD, ~15 % LNS, ~15 % SA.
    placer = CDLNSSAPlacer(
        cd_hard_cap_s=short_budget_s * 0.70,
        cd_min_time_s=min(60.0, short_budget_s * 0.20),
        lns_budget_s=short_budget_s * 0.15,
        sa_budget_s=short_budget_s * 0.15,
        verbose=False,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    e25_wall = time.time() - t0
    e25_canon = compute_proxy_cost(placement, benchmark, plc)
    print(f"[{bench_name}] E25 (budget={short_budget_s}s actual={e25_wall:.0f}s) "
          f"canon={e25_canon['proxy_cost']:.4f} ovl={int(e25_canon['overlap_count'])}")

    if int(e25_canon["overlap_count"]) > 0:
        print(f"[{bench_name}] E25 left overlaps; skipping C1 descent")
        return {
            "bench": bench_name,
            "short_budget": short_budget_s,
            "e25_canon": float(e25_canon["proxy_cost"]),
            "e25_overlaps": int(e25_canon["overlap_count"]),
            "e25_wall": e25_wall,
            "c1_skipped": "e25_has_overlaps",
        }

    # C1 descent.
    proxy = DiffProxyV2(benchmark, plc, device="cpu", gamma_frac=0.005)
    t0 = time.time()
    try:
        pos_final, log, c1_wall, best_pos, best_canon = descend_lbfgs(
            proxy, benchmark, plc, placement,
            num_stages=4, steps_per_stage=25,
            label=f"{bench_name}_c1",
        )
    except Exception as exc:
        return {
            "bench": bench_name,
            "short_budget": short_budget_s,
            "e25_canon": float(e25_canon["proxy_cost"]),
            "e25_overlaps": int(e25_canon["overlap_count"]),
            "c1_error": repr(exc),
            "c1_tb": traceback.format_exc(),
        }

    ev = evaluate_with_legalize(best_pos, benchmark, plc, label=f"{bench_name}_c1")
    c1_post = ev["post_legalize"]["proxy_cost"]
    lift = (float(e25_canon["proxy_cost"]) - c1_post) / float(e25_canon["proxy_cost"]) * 100.0
    print(f"[{bench_name}] C1 post-leg {c1_post:.4f} ovl={ev['post_legalize']['overlap_count']} "
          f"lift_vs_e25={lift:+.2f}%")

    return {
        "bench": bench_name,
        "short_budget": short_budget_s,
        "e25_canon": float(e25_canon["proxy_cost"]),
        "e25_overlaps": int(e25_canon["overlap_count"]),
        "e25_wall": e25_wall,
        "c1_post_leg_canon": c1_post,
        "c1_post_leg_overlaps": int(ev["post_legalize"]["overlap_count"]),
        "c1_wall": c1_wall,
        "lift_vs_e25_pct": lift,
        "c1_descent_log": log,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benches", nargs="*",
                    default=["ibm10", "ibm12", "ibm14", "ibm17"])
    ap.add_argument("--budget", type=int, default=300,
                    help="Short budget for E25 (seconds)")
    args = ap.parse_args()

    out_path = Path("experiments/E95_diff_proxy_v2/results/walltrunc_summary.json")
    all_results = {}
    for b in args.benches:
        all_results[b] = run_walltrunc(b, short_budget_s=args.budget)
        out_path.write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\n{'bench':<8} {'e25':>10} {'c1':>10} {'lift_%':>8}")
    for b, r in all_results.items():
        if "c1_post_leg_canon" not in r:
            print(f"{b:<8} (skipped/errored)")
            continue
        print(f"{b:<8} {r['e25_canon']:>10.4f} {r['c1_post_leg_canon']:>10.4f} "
              f"{r['lift_vs_e25_pct']:>7.2f}%")


if __name__ == "__main__":
    main()
