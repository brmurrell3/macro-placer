"""Standalone PA smoke — bypasses the cascade pipeline and tests PA
directly with a dedicated budget.

Pipeline:
  1. SDF init → IncrementalProxyEvaluator
  2. CD polish (config'd budget) → reach LNS-representative plateau
  3. PA polish (config'd budget, full N=12+)
  4. Compare init / plateau / PA-final proxy + overlap

Used to validate H2 mechanism independent of cascade budget allocation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from pa_core import run_pa_polish


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="ibm01")
    p.add_argument("--cd-budget-s", type=float, default=30.0)
    p.add_argument("--pa-budget-s", type=float, default=200.0)
    p.add_argument("--baseline", action="store_true",
                   help="Run SA-v2 instead of PA for baseline comparison")
    p.add_argument("--multistart", action="store_true",
                   help="Run multi-start SA-v2 instead of PA")
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--n-replicas", type=int, default=12)
    p.add_argument("--n-ladder", type=int, default=20)
    p.add_argument("--sweeps-per-step", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=" * 72)
    if args.baseline:
        label = "BASELINE-SA-v2"
    elif args.multistart:
        label = f"MULTI-START-SA-v2 (N={args.n_chains})"
    else:
        label = "PA"
    print(f"E111 standalone {label} smoke")
    print(f"=" * 72)
    print(f"bench={args.bench} cd_budget={args.cd_budget_s}s pa_budget={args.pa_budget_s}s")
    if not args.baseline:
        print(f"N={args.n_replicas} ladder={args.n_ladder} sweeps={args.sweeps_per_step}")
    print(f"seed={args.seed}")
    print()

    # Load + init.
    t0 = time.perf_counter()
    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement = sdf_init(benchmark)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    init_proxy = float(evaluator.current_cost()["proxy"])
    print(f"[{time.perf_counter()-t0:5.1f}s] init proxy={init_proxy:.5f}")

    # CD polish to plateau.
    hard_movable = [
        i for i in range(benchmark.num_hard_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    movable_all = hard_movable + list(
        range(benchmark.num_hard_macros, placement.shape[0])
    )
    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish ({args.cd_budget_s}s)...")
    run_cd(evaluator, benchmark, plc, movable_all, args.cd_budget_s, log_fn=None)
    plateau_proxy = float(evaluator.current_cost()["proxy"])
    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish done: proxy={plateau_proxy:.5f}")

    # Polish phase (PA or SA-v2).
    if args.baseline:
        # Use the original SA-v2 from cd_lns_sa/placer.py.
        import importlib.util as _il
        _spec = _il.spec_from_file_location(
            "_orig_e25", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py")
        )
        _mod = _il.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        polish_fn = _mod.run_sa_polish_v2

        def log_fn(s):
            print(s, flush=True)

        print(f"[{time.perf_counter()-t0:5.1f}s] SA-v2 polish ({args.pa_budget_s}s)...")
        stats = polish_fn(
            evaluator, benchmark, plc, hard_movable,
            time_budget_s=args.pa_budget_s,
            seed=args.seed, log_fn=log_fn,
        )
    elif args.multistart:
        from multistart_sa import run_multistart_sa_polish

        def log_fn(s):
            print(s, flush=True)

        print(f"[{time.perf_counter()-t0:5.1f}s] multi-start SA-v2 "
              f"(N={args.n_chains}, budget {args.pa_budget_s}s)...")
        stats = run_multistart_sa_polish(
            evaluator, benchmark, plc, hard_movable,
            time_budget_s=args.pa_budget_s,
            seed=args.seed, log_fn=log_fn,
            n_chains=args.n_chains,
        )
    else:
        def log_fn(s):
            print(s, flush=True)

        print(f"[{time.perf_counter()-t0:5.1f}s] PA polish ({args.pa_budget_s}s)...")
        stats = run_pa_polish(
            evaluator, benchmark, plc, hard_movable,
            time_budget_s=args.pa_budget_s,
            seed=args.seed, log_fn=log_fn,
            n_replicas=args.n_replicas,
            n_ladder=args.n_ladder,
            sweeps_per_step=args.sweeps_per_step,
        )

    final_proxy = float(evaluator.current_cost()["proxy"])
    placement = evaluator.placement.detach().clone().to(
        evaluator.placement.dtype if False else __import__("torch").float32
    )
    proxy_cost = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
    overlaps = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])

    print()
    print("=" * 72)
    print(f"Init    proxy: {init_proxy:.5f}")
    print(f"Plateau proxy: {plateau_proxy:.5f}  (Δ from init: {plateau_proxy-init_proxy:+.5f})")
    print(f"Polish  proxy: {final_proxy:.5f}  (Δ from plateau: {final_proxy-plateau_proxy:+.5f})")
    print(f"Verified proxy via compute_proxy_cost: {proxy_cost:.5f}")
    print(f"Overlap count: {overlaps}")
    print(f"Stats: {stats}")
    print(f"Total wall: {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
