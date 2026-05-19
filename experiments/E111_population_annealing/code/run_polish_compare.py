"""Polish-phase head-to-head — identical plateau, run multiple polish
methods, report final proxy for each. Eliminates plateau-variance noise
that confounds standalone smokes.

Pipeline:
  1. SDF init + CD polish to plateau (one canonical state).
  2. For each method in {sa_v2, lsmc, multistart, pa}:
     - Build a FRESH evaluator on a clone of the plateau placement.
     - Run that method's polish for the configured budget.
     - Report final proxy.

Same budget, same starting state, side-by-side numbers.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="ibm01")
    p.add_argument("--cd-budget-s", type=float, default=30.0)
    p.add_argument("--polish-budget-s", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+",
                   default=["sa_v2", "lsmc", "lsmc_random"],
                   choices=["sa_v2", "lsmc", "lsmc_random", "multistart", "pa"])
    p.add_argument("--k-cycle", type=int, default=4)
    p.add_argument("--inner-budget-s", type=float, default=25.0)
    args = p.parse_args()

    print(f"=" * 72)
    print(f"E111 polish head-to-head — {args.bench} polish_budget={args.polish_budget_s}s")
    print(f"=" * 72)

    t0 = time.perf_counter()
    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement = sdf_init(benchmark)
    setup_eval = IncrementalProxyEvaluator(benchmark, plc, placement)
    init_proxy = float(setup_eval.current_cost()["proxy"])

    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish ({args.cd_budget_s}s)...")
    hard_movable = [
        i for i in range(benchmark.num_hard_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    movable_all = hard_movable + list(
        range(benchmark.num_hard_macros, placement.shape[0])
    )
    run_cd(setup_eval, benchmark, plc, movable_all, args.cd_budget_s, log_fn=None)
    plateau_proxy = float(setup_eval.current_cost()["proxy"])
    plateau_placement = setup_eval.placement.detach().clone()
    print(f"[{time.perf_counter()-t0:5.1f}s] plateau proxy = {plateau_proxy:.5f}")

    # Lazy-load methods.
    sa_spec = importlib.util.spec_from_file_location(
        "_orig_e25", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py"),
    )
    sa_mod = importlib.util.module_from_spec(sa_spec)
    sa_spec.loader.exec_module(sa_mod)
    sa_v2 = sa_mod.run_sa_polish_v2

    from lsmc_polish import run_lsmc_polish
    from multistart_sa import run_multistart_sa_polish
    from pa_core import run_pa_polish

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
        elif method == "lsmc":
            stats = run_lsmc_polish(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s,
                seed=args.seed, log_fn=log_fn,
                K_cycle=args.k_cycle,
                inner_budget_s=args.inner_budget_s,
                spatial_kick=True,
            )
        elif method == "lsmc_random":
            stats = run_lsmc_polish(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s,
                seed=args.seed, log_fn=log_fn,
                K_cycle=args.k_cycle,
                inner_budget_s=args.inner_budget_s,
                spatial_kick=False,
            )
        elif method == "multistart":
            stats = run_multistart_sa_polish(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s,
                seed=args.seed, log_fn=log_fn,
                n_chains=4,
            )
        elif method == "pa":
            stats = run_pa_polish(
                ev, benchmark, plc, hard_movable,
                time_budget_s=args.polish_budget_s,
                seed=args.seed, log_fn=log_fn,
            )
        else:
            print(f"  unknown method {method}; skipping")
            continue

        post = float(ev.current_cost()["proxy"])
        ovl = int(compute_overlap_metrics(
            ev.placement.detach().clone().to(torch.float32),
            benchmark)["overlap_count"])
        wall = time.perf_counter() - t_method0
        delta_pct = (post - pre) / pre * 100.0
        results.append({
            "method": method, "pre": pre, "post": post,
            "delta_pct": delta_pct, "overlaps": ovl, "wall": wall,
        })
        print(f"  result: pre={pre:.5f} post={post:.5f} Δ={delta_pct:+.3f}% "
              f"ovl={ovl} wall={wall:.0f}s")

    print("\n" + "=" * 72)
    print(f"Polish comparison — bench={args.bench} plateau={plateau_proxy:.5f}")
    print(f"{'method':<15} {'post':>10} {'delta':>10} {'ovl':>5} {'wall':>8}")
    for r in sorted(results, key=lambda x: x["post"]):
        winner = " ← BEST" if r == sorted(results, key=lambda x: x["post"])[0] else ""
        print(f"{r['method']:<15} {r['post']:>10.5f} {r['delta_pct']:>+9.3f}% "
              f"{r['overlaps']:>5d} {r['wall']:>7.0f}s{winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
