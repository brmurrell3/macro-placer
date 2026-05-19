"""Head-to-head: WireMask polish vs continued SA-v2 polish on same plateau.

Reuses the plateau-pinning harness from E111. After CD-init to a
plateau, apply each polish method with same budget; compare final proxy.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from wiremask_polish import wiremask_polish


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ibm01")
    ap.add_argument("--cd-budget-s", type=float, default=120.0)
    ap.add_argument("--polish-budget-s", type=float, default=180.0)
    ap.add_argument("--n-per-axis", type=int, default=9)
    ap.add_argument("--local-radius-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+", default=["sa_v2", "wiremask"])
    args = ap.parse_args()

    torch.set_num_threads(1)
    print(f"=" * 72)
    print(f"E113 head-to-head — {args.bench} polish_budget={args.polish_budget_s}s")
    print(f"=" * 72)
    t0 = time.perf_counter()
    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement = sdf_init(benchmark)
    setup_eval = IncrementalProxyEvaluator(benchmark, plc, placement)
    init_proxy = float(setup_eval.current_cost()["proxy"])
    hard_movable = [i for i in range(benchmark.num_hard_macros)
                    if not bool(benchmark.macro_fixed[i])]
    movable_all = hard_movable + list(
        range(benchmark.num_hard_macros, placement.shape[0]))

    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish ({args.cd_budget_s}s) "
          f"|H|={len(hard_movable)} |M|={len(movable_all)}...")
    run_cd(setup_eval, benchmark, plc, movable_all, args.cd_budget_s, log_fn=None)
    plateau_proxy = float(setup_eval.current_cost()["proxy"])
    plateau_placement = setup_eval.placement.detach().clone()
    print(f"[{time.perf_counter()-t0:5.1f}s] plateau proxy = {plateau_proxy:.5f}")

    sa_spec = importlib.util.spec_from_file_location(
        "_e25", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py"),
    )
    sa_mod = importlib.util.module_from_spec(sa_spec)
    sa_spec.loader.exec_module(sa_mod)
    sa_v2 = sa_mod.run_sa_polish_v2

    results = []
    for method in args.methods:
        ev = IncrementalProxyEvaluator(benchmark, plc, plateau_placement.clone())
        pre = float(ev.current_cost()["proxy"])
        t_method0 = time.perf_counter()
        print(f"\n--- {method} ---")

        def log_fn(s):
            print(f"  {s}", flush=True)

        if method == "sa_v2":
            stats = sa_v2(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s,
                seed=args.seed, log_fn=log_fn,
            )
        elif method == "wiremask":
            stats = wiremask_polish(
                ev, benchmark, movable_all,
                n_per_axis=args.n_per_axis,
                local_radius_frac=args.local_radius_frac,
                max_passes=5,
                time_budget_s=args.polish_budget_s,
                log=log_fn,
            )
        elif method == "wiremask_then_sa":
            stats = wiremask_polish(
                ev, benchmark, movable_all,
                n_per_axis=args.n_per_axis,
                local_radius_frac=args.local_radius_frac,
                max_passes=3,
                time_budget_s=args.polish_budget_s * 0.5,
                log=log_fn,
            )
            stats2 = sa_v2(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s * 0.5,
                seed=args.seed, log_fn=log_fn,
            )
            stats = {"wiremask": stats, "sa_v2": stats2}
        elif method == "sa_then_wiremask":
            stats = sa_v2(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s * 0.5,
                seed=args.seed, log_fn=log_fn,
            )
            stats2 = wiremask_polish(
                ev, benchmark, movable_all,
                n_per_axis=args.n_per_axis,
                local_radius_frac=args.local_radius_frac,
                max_passes=3,
                time_budget_s=args.polish_budget_s * 0.5,
                log=log_fn,
            )
            stats = {"sa_v2": stats, "wiremask": stats2}
        else:
            print(f"  unknown method {method}; skipping")
            continue

        post = float(ev.current_cost()["proxy"])
        ovl = int(compute_overlap_metrics(
            ev.placement.detach().clone().to(torch.float32),
            benchmark)["overlap_count"])
        wall = time.perf_counter() - t_method0
        delta_pct = (post - pre) / pre * 100.0
        results.append({"method": method, "pre": pre, "post": post,
                        "delta_pct": delta_pct, "overlaps": ovl, "wall": wall})
        print(f"  result: pre={pre:.5f} post={post:.5f} Δ={delta_pct:+.3f}% "
              f"ovl={ovl} wall={wall:.0f}s")

    print("\n" + "=" * 72)
    print(f"Polish comparison — bench={args.bench} plateau={plateau_proxy:.5f}")
    print(f"{'method':<22} {'post':>10} {'delta':>10} {'ovl':>5} {'wall':>8}")
    valid = [r for r in results if r["overlaps"] == 0]
    if valid:
        best = min(valid, key=lambda x: x["post"])
        for r in sorted(results, key=lambda x: x["post"]):
            mark = " ← BEST" if r is best else ""
            print(f"{r['method']:<22} {r['post']:>10.5f} {r['delta_pct']:>+9.3f}% "
                  f"{r['overlaps']:>5d} {r['wall']:>7.0f}s{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
