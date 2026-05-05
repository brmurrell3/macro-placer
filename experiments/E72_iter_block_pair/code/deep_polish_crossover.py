"""Deep-polished crossover: E61-V2-style with cached placements.

Hypothesis: E71 V3 found 50% feasible crossovers on ibm01 with extended
legalize, but their proxy was worse than starting E25 because the polish
budget (90s) was too short. E61 V2 historical used ~30 min polish (CD +
LNS + SA-v2) and reached the third basin (-0.55% on ibm12).

This script: try N crossover seeds, pick top-K feasible ones, deep-polish
each (full CD + LNS + SA-v2), return best.

Wall budget: ~30-60 min/bench. Tractable on 4 benches in parallel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "code"
_E71 = _ROOT / "experiments" / "E71_kblock_sp_swap" / "code"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_E69) not in sys.path:
    sys.path.insert(0, str(_E69))
if str(_E71) not in sys.path:
    sys.path.insert(0, str(_E71))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extended_legalize import extended_legalize
from sp_search_placer import sp_guided_swap_search

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse E25 polish primitives.
import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
spec = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_E25_MOD)
run_lns_gridbin = _E25_MOD.run_lns_gridbin
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2


def quadrant_crossover(state, other, benchmark, *, seed: int):
    """E61 V2 quadrant crossover with extended legalize."""
    rng = np.random.default_rng(seed=seed)
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    state_f64 = state.detach().to(torch.float64)
    state_np = state_f64.cpu().numpy()
    quadrant = (
        (state_np[:n_hard, 0] >= cw / 2.0).astype(int)
        + 2 * (state_np[:n_hard, 1] >= ch / 2.0).astype(int)
    )
    pick_state = rng.random(4) < 0.5
    if pick_state.all() or (~pick_state).all():
        pick_state[int(rng.integers(4))] = not pick_state[int(rng.integers(4))]

    crossover = other.detach().to(torch.float64).clone()
    n_from_state = 0
    for i in range(n_hard):
        if pick_state[quadrant[i]]:
            crossover[i] = state_f64[i]
            n_from_state += 1

    crossover_f32 = crossover.to(torch.float32)
    crossover_f32, _ = project_overlaps(crossover_f32, benchmark)
    overlap = compute_overlap_metrics(crossover_f32, benchmark)["overlap_count"]
    if overlap > 0:
        crossover_f32, _ = extended_legalize(
            crossover_f32, benchmark, max_passes=15, jitter_scale=0.3, seed=seed,
        )
        overlap = compute_overlap_metrics(crossover_f32, benchmark)["overlap_count"]
    return crossover_f32, {
        "feasible": overlap == 0,
        "residual_overlap": overlap,
        "n_from_state": n_from_state,
        "pick_state": pick_state.tolist(),
    }


def deep_polish(placement, benchmark, plc, *, cd_budget=1200.0, lns_budget=600.0, sa_budget=600.0, log=None):
    """Full polish: CD + LNS + SA-v2, like E25/E41 production pipeline."""
    if log is None:
        log = lambda s: print(s, flush=True)
    placement = placement.detach().clone()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]

    log(f"  [polish-CD] starting (cap={cd_budget:.0f}s)...")
    info = run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=300.0, hard_cap_s=cd_budget,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    log(f"  [polish-CD] {info['exit_reason']}, sweeps={info['sweeps']}, "
        f"proxy={evaluator.current_cost()['proxy']:.5f}")

    log(f"  [polish-LNS] starting (budget={lns_budget:.0f}s)...")
    lns_info = run_lns_gridbin(
        evaluator=evaluator, benchmark=benchmark, plc=plc,
        hard_movable=[i for i in range(n_hard) if not bool(fixed[i])],
        time_budget_s=lns_budget,
        destroy_frac=0.05, destroy_cap=30, seed=42, log_fn=None,
    )
    log(f"  [polish-LNS] samples={lns_info['samples']}, Δ={lns_info['total_improvement']:+.5f}, "
        f"proxy={evaluator.current_cost()['proxy']:.5f}")

    log(f"  [polish-SA] starting (budget={sa_budget:.0f}s)...")
    sa_info = run_sa_polish_v2(
        evaluator=evaluator, benchmark=benchmark, plc=plc,
        hard_movable=[i for i in range(n_hard) if not bool(fixed[i])],
        time_budget_s=sa_budget, T0=5e-4, Tf=1e-6, seed=42, log_fn=None,
    )
    log(f"  [polish-SA] best={sa_info['best_proxy']:.5f}, "
        f"final={evaluator.current_cost()['proxy']:.5f}")

    return evaluator.placement.detach().clone().to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3, help="top-K feasible to deep-polish")
    ap.add_argument("--cd-budget", type=float, default=1200.0)
    ap.add_argument("--lns-budget", type=float, default=600.0)
    ap.add_argument("--sa-budget", type=float, default=600.0)
    ap.add_argument("--quick-polish", type=float, default=120.0, help="brief polish for filtering")
    ap.add_argument("--iter-e69-rounds", type=int, default=2)
    args = ap.parse_args()

    bench_name = args.bench
    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25 = torch.load(e69 / f"e25_{bench_name}.pt", weights_only=False)
    e41 = torch.load(e69 / f"e41_{bench_name}.pt", weights_only=False)
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    e25_pl = e25["placement"]
    e41_pl = e41["placement"]
    e25_proxy = float(e25["proxy"])
    e41_proxy = float(e41["proxy"])
    if e25_proxy <= e41_proxy:
        state, other = e25_pl, e41_pl; sl, ol = "E25", "E41"
        state_proxy = e25_proxy
    else:
        state, other = e41_pl, e25_pl; sl, ol = "E41", "E25"
        state_proxy = e41_proxy

    print(f"[deep-cross] {bench_name}: start={sl} ({state_proxy:.5f}), other={ol}")
    print(f"[deep-cross] trying {args.n_seeds} seeds, picking top-{args.top_k} for deep polish")

    # Phase 1: try N crossover seeds, brief-polish each, rank by proxy.
    candidates = []
    t_phase1 = time.time()
    for s in range(args.n_seeds):
        cross, x_stats = quadrant_crossover(state, other, benchmark, seed=42 + s)
        if not x_stats["feasible"]:
            print(f"  seed {s}: infeasible (residual={x_stats['residual_overlap']})")
            continue

        # Brief polish for ranking.
        evaluator = IncrementalProxyEvaluator(benchmark, plc, cross.clone())
        n_hard = benchmark.num_hard_macros
        fixed = benchmark.macro_fixed.cpu().numpy()
        movable = [i for i in range(n_hard) if not bool(fixed[i])]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=30.0, hard_cap_s=args.quick_polish,
            patience=3, plateau_threshold=0.005, log_fn=None,
        )
        polished = evaluator.placement.detach().clone().to(torch.float32)
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        if polished_overlap > 0:
            print(f"  seed {s}: brief polish left overlaps={polished_overlap}")
            continue
        candidates.append({
            "seed": 42 + s, "polished": polished, "proxy": polished_proxy,
            "x_stats": x_stats,
        })
        print(f"  seed {s}: brief polish proxy={polished_proxy:.5f} "
              f"(state={x_stats['n_from_state']}, pick={x_stats['pick_state']})")

    t_phase1_wall = time.time() - t_phase1
    print(f"[deep-cross] phase 1 done in {t_phase1_wall:.0f}s: "
          f"{len(candidates)}/{args.n_seeds} feasible")

    if not candidates:
        print(f"[deep-cross] NO FEASIBLE CROSSOVERS — falling back to start ({state_proxy:.5f})")
        return

    # Sort by brief-polish proxy; take top K.
    candidates.sort(key=lambda c: c["proxy"])
    top_candidates = candidates[:args.top_k]
    print(f"[deep-cross] top {len(top_candidates)} for deep polish:")
    for c in top_candidates:
        print(f"  seed {c['seed']}: brief proxy={c['proxy']:.5f}")

    # Phase 2: deep polish each top candidate.
    deep_results = []
    for c in top_candidates:
        print(f"\n  deep-polishing seed {c['seed']} (brief={c['proxy']:.5f})...")
        polished = deep_polish(
            c["polished"], benchmark, plc,
            cd_budget=args.cd_budget, lns_budget=args.lns_budget, sa_budget=args.sa_budget,
        )
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        deep_results.append({
            "seed": c["seed"], "deep_polished": polished,
            "proxy": polished_proxy, "overlap": polished_overlap,
        })
        print(f"  seed {c['seed']}: deep proxy={polished_proxy:.5f} overlap={polished_overlap}")

    # Phase 3: pick best deep-polished.
    deep_results.sort(key=lambda r: r["proxy"])
    best = deep_results[0]
    print(f"\n[deep-cross] best after deep polish: seed {best['seed']} proxy={best['proxy']:.5f}")

    # Phase 4: iter-E69 SP-search on the best.
    final_state = best["deep_polished"].detach().clone()
    best_proxy_overall = best["proxy"]
    best_state_overall = final_state.detach().clone()
    iter_log = []
    for r in range(args.iter_e69_rounds):
        print(f"\n  iter-E69 round {r+1}/{args.iter_e69_rounds}")
        rotated, sp_stats = sp_guided_swap_search(
            final_state, other, benchmark, plc,
            max_attempts=300, budget_seconds=600.0,
            seed=100 + r, axis_preserving_only=True, log=lambda s: None,
        )
        rotated_overlap = compute_overlap_metrics(rotated, benchmark)["overlap_count"]
        rotated_proxy = float(sp_stats["final_proxy"])
        print(f"    swap: tried={sp_stats['swaps_tried']} accepted={sp_stats['swaps_accepted']} "
              f"proxy={rotated_proxy:.5f}")
        iter_log.append({"round": r, "tried": sp_stats["swaps_tried"],
                         "accepted": sp_stats["swaps_accepted"], "proxy": rotated_proxy})
        if rotated_overlap == 0 and rotated_proxy < best_proxy_overall - 1e-7:
            best_proxy_overall = rotated_proxy
            best_state_overall = rotated.detach().clone()
            print(f"    NEW BEST: {best_proxy_overall:.5f}")
        # Continue from updated state for next round.
        final_state = rotated

    out = {
        "bench_name": bench_name,
        "start_basin": sl,
        "start_proxy": state_proxy,
        "other_basin_proxy": e41_proxy if sl == "E25" else e25_proxy,
        "n_seeds_tried": args.n_seeds,
        "n_feasible": len(candidates),
        "deep_polish_results": [{"seed": r["seed"], "proxy": r["proxy"]} for r in deep_results],
        "best_deep_proxy": best["proxy"],
        "iter_e69_log": iter_log,
        "final_proxy": best_proxy_overall,
        "lift_vs_start": best_proxy_overall - state_proxy,
        "lift_frac": (best_proxy_overall - state_proxy) / state_proxy,
    }
    out_path = _HERE.parent / "results" / f"deep_cross_{bench_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    out_pt = _HERE.parent / "results" / f"deep_cross_{bench_name}.pt"
    torch.save({"placement": best_state_overall.cpu(), "stats": out, "bench_name": bench_name},
               out_pt)
    print(f"\n[deep-cross] saved -> {out_path}")
    print(f"  start: {state_proxy:.5f}")
    print(f"  best:  {best_proxy_overall:.5f}")
    print(f"  lift:  {out['lift_vs_start']:+.5f} ({100 * out['lift_frac']:+.3f}%)")


if __name__ == "__main__":
    main()
