"""Isolated H1 smoke — compare LNS with LP-dual destroy vs cost-aware
destroy at the same budget, on the same starting plateau.

Skips the full cascade and isolates H1's effect on the LNS phase alone.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
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

from lp_destroy_rank import lp_dual_destroy


def _run_lns_with_destroy(evaluator, benchmark, plc, hard_movable, destroy_fn,
                          budget_s, label):
    """Mirror the LNS phase in cd_lns_sa.placer.run_lns_gridbin, but call
    the supplied destroy_fn instead of the module's _cost_aware_destroy.
    """
    # Lazy-load the original cd_lns_sa to reuse its run_lns_gridbin and
    # _gridbin_reinsert primitives — we'll monkey-patch the local destroy.
    _spec = importlib.util.spec_from_file_location(
        f"_orig_e25_{label}", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py")
    )
    mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mod)
    mod._cost_aware_destroy = destroy_fn

    pre = float(evaluator.current_cost()["proxy"])
    t0 = time.perf_counter()
    stats = mod.run_lns_gridbin(
        evaluator, benchmark, plc, hard_movable,
        time_budget_s=budget_s,
        destroy_frac=0.05,
        destroy_cap=30,
        log_fn=lambda s: print(f"    [{label}] {s}", flush=True),
    )
    wall = time.perf_counter() - t0
    post = float(evaluator.current_cost()["proxy"])
    overlaps = int(compute_overlap_metrics(
        evaluator.placement.detach().clone().to(__import__("torch").float32),
        benchmark)["overlap_count"])
    return {
        "label": label, "pre": pre, "post": post, "delta": post - pre,
        "delta_pct": (post - pre) / pre * 100.0,
        "wall_s": wall, "overlaps": overlaps,
        "stats": stats,
    }


def _cost_aware_destroy_orig(evaluator, hard_movable, K):
    """Reference cost-aware destroy (copy of the original)."""
    cw = evaluator.width / 2.0
    ch = evaluator.height / 2.0
    baseline_p = evaluator.current_cost()["proxy"]
    scores = []
    for idx in hard_movable:
        try:
            p = evaluator.delta_cost(idx, (cw, ch))["proxy"]
            scores.append((idx, p - baseline_p))
        except Exception:
            scores.append((idx, 0.0))
    scores.sort(key=lambda s: s[1])
    return [s[0] for s in scores[:K]]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="ibm01")
    p.add_argument("--cd-budget-s", type=float, default=30.0)
    p.add_argument("--lns-budget-s", type=float, default=120.0)
    p.add_argument("--mix-mode", default="mix",
                   choices=["pure_lp", "mix", "cost_aware"])
    args = p.parse_args()

    # Configure LP-dual mode via env so module reads it.
    os.environ["MPC_V2_H1_MIX"] = args.mix_mode

    print(f"=" * 72)
    print(f"E110 H1 isolated smoke — LP-dual vs cost-aware LNS destroy")
    print(f"=" * 72)
    print(f"bench={args.bench} cd_budget={args.cd_budget_s}s "
          f"lns_budget={args.lns_budget_s}s mix_mode={args.mix_mode}")
    print()

    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement0 = sdf_init(benchmark)
    print(f"loaded {args.bench}: {benchmark.num_hard_macros} hard macros, "
          f"grid {plc.grid_row}×{plc.grid_col}")

    # Build plateau once via CD.
    eval_setup = IncrementalProxyEvaluator(benchmark, plc, placement0)
    hard_movable = [
        i for i in range(benchmark.num_hard_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    movable_all = hard_movable + list(
        range(benchmark.num_hard_macros, placement0.shape[0])
    )
    t0 = time.perf_counter()
    print(f"\nCD polish ({args.cd_budget_s}s) to plateau...")
    run_cd(eval_setup, benchmark, plc, movable_all, args.cd_budget_s, log_fn=None)
    plateau_placement = eval_setup.placement.detach().clone()
    plateau_proxy = float(eval_setup.current_cost()["proxy"])
    print(f"plateau proxy = {plateau_proxy:.5f} (wall {time.perf_counter()-t0:.1f}s)")

    # ── Variant A: cost-aware destroy ──
    eval_a = IncrementalProxyEvaluator(benchmark, plc, plateau_placement.clone())
    print(f"\n--- Variant A: cost-aware destroy ---")
    res_a = _run_lns_with_destroy(
        eval_a, benchmark, plc, hard_movable,
        _cost_aware_destroy_orig, args.lns_budget_s, "cost_aware",
    )
    print(f"  pre={res_a['pre']:.5f} → post={res_a['post']:.5f} "
          f"Δ={res_a['delta']:+.5f} ({res_a['delta_pct']:+.2f}%) "
          f"overlaps={res_a['overlaps']} wall={res_a['wall_s']:.1f}s")

    # ── Variant B: LP-dual destroy ──
    eval_b = IncrementalProxyEvaluator(benchmark, plc, plateau_placement.clone())
    print(f"\n--- Variant B: LP-dual destroy ({args.mix_mode}) ---")
    res_b = _run_lns_with_destroy(
        eval_b, benchmark, plc, hard_movable,
        lp_dual_destroy, args.lns_budget_s, f"lp_dual_{args.mix_mode}",
    )
    print(f"  pre={res_b['pre']:.5f} → post={res_b['post']:.5f} "
          f"Δ={res_b['delta']:+.5f} ({res_b['delta_pct']:+.2f}%) "
          f"overlaps={res_b['overlaps']} wall={res_b['wall_s']:.1f}s")

    # ── Verdict ──
    print("\n" + "=" * 72)
    print(f"{'variant':<20} {'pre':>10} {'post':>10} {'delta':>10} {'pct':>8}")
    print(f"{'cost-aware':<20} {res_a['pre']:>10.5f} {res_a['post']:>10.5f} "
          f"{res_a['delta']:>+10.5f} {res_a['delta_pct']:>+7.2f}%")
    print(f"{'lp-dual-'+args.mix_mode:<20} {res_b['pre']:>10.5f} {res_b['post']:>10.5f} "
          f"{res_b['delta']:>+10.5f} {res_b['delta_pct']:>+7.2f}%")
    h1_extra_lift = (res_a['post'] - res_b['post']) / res_a['post'] * 100.0
    print(f"\nLP-dual extra lift vs cost-aware: {h1_extra_lift:+.3f}%")
    if h1_extra_lift > 0.5:
        print("  ✔ H1 passes 0.5% extra-lift gate.")
    elif h1_extra_lift > 0.0:
        print("  ◦ H1 marginally lifts; below 0.5% threshold.")
    else:
        print("  ✘ H1 does NOT lift vs cost-aware.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
