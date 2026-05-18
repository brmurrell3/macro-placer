"""Calibration probe — do LP-dual macro scores correlate with measured
canonical Δproxy on real LNS destroy candidates?

The H1 hypothesis (E110 manifest) is that LP-dual prices π are valid
first-order subgradients of canonical congestion. If true, the ranking
of macros by π-weighted score should align with the ranking of macros
by their actual canonical Δproxy when moved.

Spike on ibm01: build evaluator with SDF init, run a quick CD polish to
reach an LNS-representative plateau, solve LP, score 64 random hard
movable macros by LP duals, compare to actual canonical Δproxy of a
move-to-canvas-center probe (matches `_cost_aware_destroy`'s probe).

Decision rule:
  Spearman ≥ 0.4  → continue H1 development on May 19.
  Spearman 0.2 – 0.4 → tune subset sizes and re-run.
  Spearman < 0.2 → abandon H1; reallocate to H2 hyperparameter sweep.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Same directory imports (the mcf_lp module).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir

from mcf_lp import build_and_solve_subset_mcf, score_macros_by_duals


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two arrays."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = ((rx - rx_mean) * (ry - ry_mean)).sum()
    den = np.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum())
    if den == 0:
        return float("nan")
    return float(num / den)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="ibm01")
    p.add_argument("--cd-budget-s", type=float, default=30.0)
    p.add_argument("--n-probes", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-edge-frac", type=float, default=0.20)
    p.add_argument("--top-k-nets", type=int, default=2000)
    p.add_argument("--lp-budget-s", type=float, default=10.0)
    p.add_argument(
        "--also-test-frac",
        nargs="*",
        type=float,
        default=[0.10, 0.30],
        help="Additional top_edge_frac values to test (kept short for runtime).",
    )
    args = p.parse_args()

    print(f"=" * 72)
    print(f"E110 calibration probe — LP-dual π vs canonical Δproxy")
    print(f"=" * 72)
    print(f"bench={args.bench} cd_budget={args.cd_budget_s}s n_probes={args.n_probes}")
    print(f"top_edge_frac={args.top_edge_frac} top_k_nets={args.top_k_nets}")
    print(f"lp_budget={args.lp_budget_s}s seed={args.seed}")
    print()

    # 1. Load bench + build evaluator.
    t0 = time.perf_counter()
    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement = sdf_init(benchmark)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    init_proxy = evaluator.current_cost()["proxy"]
    print(f"[{time.perf_counter()-t0:5.1f}s] loaded {args.bench}: "
          f"{benchmark.num_hard_macros} hard macros, {evaluator.num_nets} nets, "
          f"grid {evaluator.grid_row}×{evaluator.grid_col}, "
          f"init proxy={init_proxy:.5f}")

    # 2. Quick CD polish to reach LNS-representative plateau.
    hard_movable = [
        i for i in range(benchmark.num_hard_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    movable_all = hard_movable + list(
        range(benchmark.num_hard_macros, placement.shape[0])
    )
    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish for {args.cd_budget_s}s...")
    stats = run_cd(evaluator, benchmark, plc, movable_all, args.cd_budget_s, log_fn=None)
    polish_proxy = evaluator.current_cost()["proxy"]
    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish done: proxy={polish_proxy:.5f} "
          f"(Δ={polish_proxy-init_proxy:+.5f}) sweeps={stats['sweeps']}")

    # 3. Solve LP at current placement (try main config + extras).
    configs = [(args.top_edge_frac, args.top_k_nets)]
    for f in args.also_test_frac:
        configs.append((f, args.top_k_nets))

    results = []
    for (edge_frac, k_nets) in configs:
        print(f"\n[{time.perf_counter()-t0:5.1f}s] solving LP "
              f"(top_edge_frac={edge_frac}, top_k_nets={k_nets})...")
        duals = build_and_solve_subset_mcf(
            evaluator,
            top_edge_frac=edge_frac,
            top_k_nets=k_nets,
            wall_budget_s=args.lp_budget_s,
        )
        print(f"  status={duals.status} wall={duals.solver_wall_s:.2f}s "
              f"n_vars={duals.n_vars} n_constraints={duals.n_constraints} "
              f"n_subset_nets={duals.n_subset_nets}")
        print(f"  info: {duals.info}")
        if duals.status != "ok":
            print(f"  LP did not solve; skipping correlation.")
            results.append((edge_frac, k_nets, duals.status, float("nan"), float("nan")))
            continue

        nz_h = int((duals.pi_h > 0).sum())
        nz_v = int((duals.pi_v > 0).sum())
        max_pi = float(max(duals.pi_h.max(), duals.pi_v.max()))
        sum_pi = float(duals.pi_h.sum() + duals.pi_v.sum())
        print(f"  pi: nonzero H={nz_h} V={nz_v} max={max_pi:.4f} sum={sum_pi:.4f}")

        # 4. Score all hard movable macros by LP duals.
        scored = score_macros_by_duals(evaluator, hard_movable, duals)
        score_map = {m: s for (m, s) in scored}

        # 5. Sample probes uniformly from hard_movable, measure Δproxy of
        #    move-to-center (matches _cost_aware_destroy probe). For a
        #    Spearman that exercises the ranking, sample from the top half
        #    of the LP score distribution AND the bottom half evenly.
        rng = np.random.default_rng(args.seed)
        sorted_macros = [m for (m, _) in scored]
        n_top = args.n_probes // 2
        n_bot = args.n_probes - n_top
        top_pool = sorted_macros[: max(n_top * 2, n_top)]
        bot_pool = sorted_macros[-max(n_bot * 2, n_bot):]
        top_pick = rng.choice(top_pool, size=min(n_top, len(top_pool)), replace=False)
        bot_pick = rng.choice(bot_pool, size=min(n_bot, len(bot_pool)), replace=False)
        probe_macros = list(top_pick) + list(bot_pick)

        cw = evaluator.width / 2.0
        ch = evaluator.height / 2.0
        baseline_p = evaluator.current_cost()["proxy"]

        probe_scores = []
        probe_deltas = []
        for m in probe_macros:
            m = int(m)
            try:
                p_after = evaluator.delta_cost(m, (cw, ch))["proxy"]
                d = p_after - baseline_p
            except Exception:
                continue
            probe_scores.append(score_map.get(m, 0.0))
            probe_deltas.append(d)

        if len(probe_scores) < 8:
            print(f"  insufficient probes ({len(probe_scores)})")
            results.append((edge_frac, k_nets, duals.status, float("nan"), len(probe_scores)))
            continue

        ps = np.array(probe_scores)
        pd = np.array(probe_deltas)
        # Higher π-score = more congestion contribution = LARGER reduction
        # in proxy when destroyed (move-to-center samples a typical alt
        # location). So we expect Δproxy to be MORE NEGATIVE for higher score.
        # Spearman(score, -delta) should be positive if hypothesis holds.
        spearman_pos = _spearman(ps, -pd)
        spearman_raw = _spearman(ps, pd)
        print(f"  probes: n={len(probe_scores)} "
              f"score_range=[{ps.min():.3f}, {ps.max():.3f}] "
              f"delta_range=[{pd.min():+.5f}, {pd.max():+.5f}]")
        print(f"  Spearman(score, -delta) = {spearman_pos:+.3f}  "
              f"(raw Spearman(score, delta) = {spearman_raw:+.3f})")
        results.append((edge_frac, k_nets, duals.status, spearman_pos, len(probe_scores)))

    print("\n" + "=" * 72)
    print("Calibration summary:")
    print(f"{'edge_frac':>10} {'k_nets':>8} {'status':>10} {'spearman':>10} {'n_probes':>10}")
    best_spearman = float("-inf")
    for (ef, kn, st, sp, n) in results:
        print(f"{ef:>10.2f} {kn:>8d} {st:>10s} {sp:>10.3f} {n:>10}")
        if isinstance(sp, float) and not np.isnan(sp) and sp > best_spearman:
            best_spearman = sp

    print("\nDecision:")
    if best_spearman >= 0.4:
        print(f"  ✔ best Spearman={best_spearman:+.3f} ≥ 0.4 → CONTINUE H1 to May 19.")
        return 0
    elif best_spearman >= 0.2:
        print(f"  ◦ best Spearman={best_spearman:+.3f} ∈ [0.2, 0.4) → "
              f"borderline; sweep top_k_nets and re-test.")
        return 0
    else:
        print(f"  ✘ best Spearman={best_spearman:+.3f} < 0.2 → "
              f"ABANDON H1; reallocate to H2.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
