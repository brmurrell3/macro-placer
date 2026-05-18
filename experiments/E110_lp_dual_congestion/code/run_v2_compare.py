"""Compare Option C (baseline) vs v2-H1 vs v2-H2 vs v2-both on a chosen
benchmark, with a controllable budget.

Usage:
    uv run python experiments/E110_lp_dual_congestion/code/run_v2_compare.py \
        --bench ibm01 --budget-s 300 --variants base h1 h2 h1h2

Reports per-variant proxy + overlap + wall, plus pairwise deltas vs base.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics


def _run_variant(variant: str, bench: str, budget_s: float, verbose: bool) -> dict:
    """variant in {'base', 'h1', 'h2', 'h1h2'}."""
    os.environ["MPC_V2_H1"] = "1" if "h1" in variant else "0"
    os.environ["MPC_V2_H2"] = "1" if "h2" in variant else "0"
    # Force re-import so monkey-patches refresh.
    for mod_name in list(sys.modules):
        if mod_name.startswith("submissions") or mod_name.startswith("_cd_lns") \
           or mod_name in ("pa_core", "pa_replica", "pa_resample",
                            "lp_destroy_rank", "mcf_lp"):
            del sys.modules[mod_name]

    if variant == "base":
        # Run Option C directly (no patches).
        path = _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
        spec = importlib.util.spec_from_file_location(
            "_base_cdlns_csp", str(path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        placer = mod.CDLNSSACascadeStackedPeripheryPlacer(
            budget_seconds=budget_s, verbose=verbose,
        )
    else:
        path = _ROOT / "submissions" / "cd_lns_sa_cascade_v2" / "placer.py"
        spec = importlib.util.spec_from_file_location(
            "_v2_placer", str(path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        placer = mod.CDLNSSACascadeV2Placer(
            budget_seconds=budget_s, verbose=verbose,
        )

    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    t0 = time.time()
    placement = placer.place(benchmark)
    wall = time.time() - t0
    proxy = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(placement, benchmark)
    return {
        "variant": variant,
        "proxy": proxy,
        "overlaps": int(ovl["overlap_count"]),
        "overlap_area": float(ovl["total_overlap_area"]),
        "wall_s": wall,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="ibm01")
    p.add_argument("--budget-s", type=float, default=300.0)
    p.add_argument("--variants", nargs="+", default=["base", "h1", "h2", "h1h2"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    print(f"=" * 72)
    print(f"v2 variant comparison — bench={args.bench} budget={args.budget_s}s")
    print(f"=" * 72)

    results = []
    for v in args.variants:
        print(f"\n>>> variant: {v}")
        try:
            r = _run_variant(v, args.bench, args.budget_s, not args.quiet)
        except Exception as exc:
            print(f"  variant {v} FAILED: {exc!r}")
            results.append({
                "variant": v, "proxy": float("nan"), "overlaps": -1,
                "overlap_area": float("nan"), "wall_s": float("nan"),
            })
            continue
        results.append(r)
        print(f"  proxy={r['proxy']:.5f} overlaps={r['overlaps']} wall={r['wall_s']:.1f}s")

    print("\n" + "=" * 72)
    print(f"{'variant':<10} {'proxy':>10} {'overlaps':>10} {'wall_s':>10} {'delta_base':>12}")
    base_proxy = float("nan")
    for r in results:
        if r["variant"] == "base":
            base_proxy = r["proxy"]
            break
    for r in results:
        delta = float("nan")
        if not (base_proxy != base_proxy or r["proxy"] != r["proxy"]):  # both not nan
            delta = (r["proxy"] - base_proxy) / base_proxy * 100.0
        print(f"{r['variant']:<10} {r['proxy']:>10.5f} {r['overlaps']:>10d} "
              f"{r['wall_s']:>10.1f} {delta:>11.2f}%")

    print("\nKill gate: best variant must beat base by ≥ 0.5% (=delta_base ≤ -0.5%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
