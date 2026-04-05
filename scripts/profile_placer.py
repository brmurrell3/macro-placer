"""
Phase 2 profiling: captures component breakdown, surrogate accuracy,
convergence curves, and phase timing for all 17 benchmarks.

Outputs:
  results/profiling/components.csv       -- WL/density/congestion per benchmark
  results/profiling/convergence.csv      -- proxy cost after each improvement
  results/profiling/surrogate_accuracy.csv -- surrogate vs real proxy for sampled candidates
  results/profiling/phase_timing.csv     -- time per phase per benchmark
  results/profiling/benchmark_structure.csv -- structural properties per benchmark
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from scipy import stats

import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.evaluate import IBM_BENCHMARKS, REPLACE_BASELINES

# Import placer modules
sys.path.insert(0, str(ROOT / "submissions" / "polyhedra"))
from placer import (
    PolyhedraNavigationPlacer, GridSurrogate, Navigator, NeighborGenerator,
    LPSolver, extract_assignment_vectorized
)

OUT_DIR = ROOT / "results" / "profiling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TESTCASE_ROOT = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def profile_benchmark(name):
    """Run full profiling on a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Profiling {name}")
    print(f"{'='*60}")

    # Load benchmark
    t_load = time.time()
    benchmark, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / name))
    t_load = time.time() - t_load

    sizes = benchmark.macro_sizes.numpy()
    hard_indices = np.arange(benchmark.num_hard_macros)
    movable_hard = hard_indices[~benchmark.macro_fixed[:benchmark.num_hard_macros].numpy()]

    phase_times = {"load": t_load}

    # Phase 1: Initial placement (SDF)
    placer = PolyhedraNavigationPlacer(navigate=False, verbose=False)
    t0 = time.time()
    init_pos = placer._initial_placement(benchmark)
    phase_times["sdf_init"] = time.time() - t0

    # Phase 2: Assignment extraction
    t0 = time.time()
    assignment = extract_assignment_vectorized(init_pos, sizes, movable_hard)
    phase_times["assignment"] = time.time() - t0

    # Phase 3: LP solver build + solve
    t0 = time.time()
    lp = LPSolver(benchmark, plc)
    phase_times["lp_build"] = time.time() - t0

    t0 = time.time()
    lp_result = lp.solve(assignment, time_limit=60.0)
    phase_times["lp_solve"] = time.time() - t0

    # Phase 4: Surrogate build
    t0 = time.time()
    surrogate = GridSurrogate(benchmark, plc)
    phase_times["surrogate_build"] = time.time() - t0

    # Component breakdown at init
    init_costs = compute_proxy_cost(
        torch.tensor(init_pos, dtype=torch.float32), benchmark, plc
    )

    # Navigation setup
    ng = NeighborGenerator(alpha=1.0)
    nav = Navigator(lp, ng, benchmark, plc, surrogate=surrogate)

    convergence_log = []
    convergence_log.append({
        "iteration": -1, "elapsed": 0.0,
        "proxy": init_costs["proxy_cost"],
        "wl": init_costs["wirelength_cost"],
        "density": init_costs["density_cost"],
        "congestion": init_costs["congestion_cost"],
    })

    # Collect candidate positions for post-hoc surrogate accuracy analysis
    sampled_positions = []

    # Navigation loop (NO compute_proxy_cost inside - matches production speed)
    t_nav_start = time.time()
    nav_time = 40.0

    surrogate.init_from_placement(init_pos)
    best_proxy_surr = surrogate.get_proxy_cost()
    current_positions = init_pos.copy()
    current_assignment = dict(assignment)
    hpwl_result = lp_result
    improvements = 0
    stale_iters = 0
    lp_resolves = 0
    surrogate_evals = 0
    n_candidates_total = 0
    improvement_timestamps = []

    for iteration in range(500):
        elapsed = time.time() - t_nav_start
        if elapsed > nav_time:
            break
        if stale_iters > 100:
            break

        candidates = ng.rank_candidates(hpwl_result["duals"], current_assignment, top_k=150)
        if not candidates:
            break

        cluster_candidates = ng.propose_cluster_flips(
            hpwl_result["duals"], current_assignment,
            surrogate.macro_to_nets, n_proposals=20, rng=nav.rng
        )

        projected = []
        for pair, new_dir, dual_mag in candidates:
            if time.time() - t_nav_start > nav_time:
                break
            old_dir = current_assignment[pair]
            current_assignment[pair] = new_dir
            new_pos = nav._project_flip(current_positions, pair, new_dir, current_assignment)
            current_assignment[pair] = old_dir

            if new_pos is None:
                continue

            i, k = pair
            moved = []
            if not np.allclose(new_pos[i], current_positions[i]):
                moved.append(i)
            if not np.allclose(new_pos[k], current_positions[k]):
                moved.append(k)
            for m in range(benchmark.num_hard_macros):
                if m != i and m != k and not np.allclose(new_pos[m], current_positions[m]):
                    moved.append(m)

            surr_proxy = surrogate.evaluate_move(moved, new_pos)
            surrogate_evals += 1
            projected.append((surr_proxy, new_pos, pair, new_dir, old_dir, False))

        for cluster in cluster_candidates:
            new_pos, new_assign = nav._project_cluster_flip(
                current_positions, cluster, current_assignment
            )
            if new_pos is None:
                continue
            moved = []
            for m in range(benchmark.num_hard_macros):
                if not np.allclose(new_pos[m], current_positions[m]):
                    moved.append(m)
            if moved:
                surr_proxy = surrogate.evaluate_move(moved, new_pos)
                surrogate_evals += 1
                projected.append((surr_proxy, new_pos, cluster, None, None, True))

        n_candidates_total += len(projected)

        if not projected:
            stale_iters += 1
            continue

        projected.sort(key=lambda x: x[0])

        # Save candidates for post-hoc accuracy (cheap: just copy positions)
        if iteration % 25 == 0 and len(sampled_positions) < 30:
            for cand in projected[:min(3, len(projected))]:
                surr_est, new_pos_c, _, _, _, _ = cand
                if not nav._check_overlaps_fast(new_pos_c):
                    sampled_positions.append((surr_est, new_pos_c.copy()))

        # Accept best
        best_cand = projected[0]
        surr_proxy, new_pos, pair_or_cluster, new_dir, old_dir, is_cluster = best_cand

        if nav._check_overlaps_fast(new_pos):
            stale_iters += 1
            continue

        if surr_proxy < best_proxy_surr:
            best_proxy_surr = surr_proxy

            if is_cluster:
                for (p, d) in pair_or_cluster:
                    current_assignment[p] = d
            else:
                current_assignment[pair_or_cluster] = new_dir

            current_positions = new_pos.copy()
            current_assignment = extract_assignment_vectorized(
                current_positions, sizes, movable_hard
            )
            nav._macro_to_pairs = None
            surrogate.init_from_placement(current_positions)

            improvements += 1
            stale_iters = 0
            improvement_timestamps.append((iteration, time.time() - t_nav_start, surr_proxy))

            remaining = nav_time - (time.time() - t_nav_start)
            if improvements % 5 == 0 and lp_resolves < 3 and remaining > 20:
                hpwl_result = lp.solve(current_assignment, time_limit=min(15.0, remaining - 10))
                lp_resolves += 1
        else:
            stale_iters += 1
            remaining = nav_time - (time.time() - t_nav_start)
            if stale_iters % 20 == 0 and lp_resolves < 20 and remaining > 20:
                hpwl_result = lp.solve(current_assignment, time_limit=min(15.0, remaining - 10))
                lp_resolves += 1

    phase_times["navigation"] = time.time() - t_nav_start

    # Post-hoc: surrogate accuracy (evaluate sampled positions with real proxy)
    t0 = time.time()
    surrogate_samples = []
    for surr_est, pos_copy in sampled_positions[:15]:
        real_costs = compute_proxy_cost(
            torch.tensor(pos_copy, dtype=torch.float32), benchmark, plc
        )
        surrogate_samples.append({
            "surrogate": surr_est,
            "real_proxy": real_costs["proxy_cost"],
            "real_wl": real_costs["wirelength_cost"],
            "real_den": real_costs["density_cost"],
            "real_cong": real_costs["congestion_cost"],
        })
    phase_times["accuracy_sampling"] = time.time() - t0

    # Post-hoc: convergence log from improvement timestamps + final eval
    final_costs = compute_proxy_cost(
        torch.tensor(current_positions, dtype=torch.float32), benchmark, plc
    )
    convergence_log.append({
        "iteration": 999,
        "elapsed": phase_times["navigation"],
        "proxy": final_costs["proxy_cost"],
        "wl": final_costs["wirelength_cost"],
        "density": final_costs["density_cost"],
        "congestion": final_costs["congestion_cost"],
    })
    # Also log surrogate-based improvement timestamps
    for it, elapsed, surr_p in improvement_timestamps:
        convergence_log.append({
            "iteration": it,
            "elapsed": elapsed,
            "proxy": surr_p,  # surrogate estimate (not real proxy)
            "wl": 0, "density": 0, "congestion": 0,  # not available w/o eval
        })

    total_time = sum(v for k, v in phase_times.items() if k != "accuracy_sampling")

    # Benchmark structural properties
    n_hard = benchmark.num_hard_macros
    canvas_area = benchmark.canvas_width * benchmark.canvas_height
    hard_area = float((sizes[:n_hard, 0] * sizes[:n_hard, 1]).sum())
    utilization = hard_area / canvas_area
    n_fixed = int(benchmark.macro_fixed.sum().item())
    avg_aspect = float(np.mean(
        np.maximum(sizes[:n_hard, 0], sizes[:n_hard, 1]) /
        np.maximum(np.minimum(sizes[:n_hard, 0], sizes[:n_hard, 1]), 1e-6)
    ))

    structure = {
        "name": name,
        "num_hard": n_hard,
        "num_soft": benchmark.num_soft_macros,
        "num_nets": benchmark.num_nets,
        "canvas_w": benchmark.canvas_width,
        "canvas_h": benchmark.canvas_height,
        "canvas_area": canvas_area,
        "hard_area": hard_area,
        "utilization": utilization,
        "num_fixed": n_fixed,
        "num_pairs": len(assignment),
        "avg_aspect_ratio": avg_aspect,
        "grid_rows": benchmark.grid_rows,
        "grid_cols": benchmark.grid_cols,
    }

    print(f"  Final proxy: {final_costs['proxy_cost']:.4f} "
          f"(wl={final_costs['wirelength_cost']:.4f}, "
          f"den={final_costs['density_cost']:.4f}, "
          f"cong={final_costs['congestion_cost']:.4f})")
    print(f"  Improvements: {improvements}, Surrogate evals: {surrogate_evals}, "
          f"LP resolves: {lp_resolves}")
    print(f"  Total time: {total_time:.1f}s "
          f"(init={phase_times['sdf_init']:.1f}s, "
          f"lp={phase_times['lp_solve']:.1f}s, "
          f"nav={phase_times['navigation']:.1f}s)")

    return {
        "name": name,
        "init_costs": init_costs,
        "final_costs": final_costs,
        "phase_times": phase_times,
        "convergence": convergence_log,
        "surrogate_samples": surrogate_samples,
        "structure": structure,
        "improvements": improvements,
        "surrogate_evals": surrogate_evals,
        "lp_resolves": lp_resolves,
        "n_candidates_total": n_candidates_total,
    }


def main():
    all_results = []

    for name in IBM_BENCHMARKS:
        result = profile_benchmark(name)
        all_results.append(result)

    # Write component breakdown CSV
    with open(OUT_DIR / "components.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "proxy", "wl", "density", "congestion",
            "wl_contrib", "den_contrib", "cong_contrib",
            "replace_proxy", "gap_pct",
            "init_proxy", "init_wl", "init_den", "init_cong",
            "nav_delta",
        ])
        for r in all_results:
            fc = r["final_costs"]
            ic = r["init_costs"]
            rep = REPLACE_BASELINES.get(r["name"], 0)
            gap = (fc["proxy_cost"] - rep) / rep * 100 if rep else 0
            w.writerow([
                r["name"],
                f"{fc['proxy_cost']:.6f}",
                f"{fc['wirelength_cost']:.6f}",
                f"{fc['density_cost']:.6f}",
                f"{fc['congestion_cost']:.6f}",
                f"{1.0 * fc['wirelength_cost']:.6f}",
                f"{0.5 * fc['density_cost']:.6f}",
                f"{0.5 * fc['congestion_cost']:.6f}",
                f"{rep:.4f}",
                f"{gap:.2f}",
                f"{ic['proxy_cost']:.6f}",
                f"{ic['wirelength_cost']:.6f}",
                f"{ic['density_cost']:.6f}",
                f"{ic['congestion_cost']:.6f}",
                f"{ic['proxy_cost'] - fc['proxy_cost']:.6f}",
            ])

    # Write convergence CSV
    with open(OUT_DIR / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "iteration", "elapsed", "proxy", "wl", "density", "congestion"])
        for r in all_results:
            for c in r["convergence"]:
                w.writerow([
                    r["name"], c["iteration"], f"{c['elapsed']:.3f}",
                    f"{c['proxy']:.6f}", f"{c['wl']:.6f}",
                    f"{c['density']:.6f}", f"{c['congestion']:.6f}",
                ])

    # Write surrogate accuracy CSV
    with open(OUT_DIR / "surrogate_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "surrogate_proxy", "real_proxy",
            "real_wl", "real_den", "real_cong",
        ])
        for r in all_results:
            for s in r["surrogate_samples"]:
                w.writerow([
                    r["name"],
                    f"{s['surrogate']:.6f}", f"{s['real_proxy']:.6f}",
                    f"{s['real_wl']:.6f}", f"{s['real_den']:.6f}",
                    f"{s['real_cong']:.6f}",
                ])

    # Write phase timing CSV
    with open(OUT_DIR / "phase_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "load", "sdf_init", "assignment", "lp_build",
            "lp_solve", "surrogate_build", "navigation", "total",
        ])
        for r in all_results:
            pt = r["phase_times"]
            total = sum(v for k, v in pt.items() if k != "accuracy_sampling")
            w.writerow([
                r["name"],
                f"{pt['load']:.3f}", f"{pt['sdf_init']:.3f}",
                f"{pt['assignment']:.3f}", f"{pt['lp_build']:.3f}",
                f"{pt['lp_solve']:.3f}", f"{pt['surrogate_build']:.3f}",
                f"{pt['navigation']:.3f}", f"{total:.3f}",
            ])

    # Write benchmark structure CSV
    with open(OUT_DIR / "benchmark_structure.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "num_hard", "num_soft", "num_nets",
            "canvas_w", "canvas_h", "utilization",
            "num_fixed", "num_pairs", "avg_aspect_ratio",
        ])
        for r in all_results:
            s = r["structure"]
            w.writerow([
                s["name"], s["num_hard"], s["num_soft"], s["num_nets"],
                f"{s['canvas_w']:.1f}", f"{s['canvas_h']:.1f}",
                f"{s['utilization']:.4f}",
                s["num_fixed"], s["num_pairs"], f"{s['avg_aspect_ratio']:.2f}",
            ])

    # Print summary analysis
    print("\n" + "=" * 80)
    print("PHASE 2 PROFILING RESULTS")
    print("=" * 80)

    # 1. Component contribution table
    print("\n## 1. Cost Component Contributions")
    print(f"{'Benchmark':>10} {'Proxy':>8} {'WL*1.0':>8} {'Den*0.5':>9} {'Cong*0.5':>10} "
          f"{'WL%':>6} {'Den%':>6} {'Cong%':>6}")
    print("-" * 75)
    for r in all_results:
        fc = r["final_costs"]
        p = fc["proxy_cost"]
        wl_c = 1.0 * fc["wirelength_cost"]
        den_c = 0.5 * fc["density_cost"]
        cong_c = 0.5 * fc["congestion_cost"]
        print(f"{r['name']:>10} {p:8.4f} {wl_c:8.4f} {den_c:9.4f} {cong_c:10.4f} "
              f"{wl_c/p*100:5.1f}% {den_c/p*100:5.1f}% {cong_c/p*100:5.1f}%")

    avg_proxy = np.mean([r["final_costs"]["proxy_cost"] for r in all_results])
    avg_wl = np.mean([r["final_costs"]["wirelength_cost"] for r in all_results])
    avg_den = np.mean([r["final_costs"]["density_cost"] for r in all_results])
    avg_cong = np.mean([r["final_costs"]["congestion_cost"] for r in all_results])
    print("-" * 75)
    print(f"{'AVG':>10} {avg_proxy:8.4f} {avg_wl:8.4f} {0.5*avg_den:9.4f} {0.5*avg_cong:10.4f} "
          f"{avg_wl/avg_proxy*100:5.1f}% {0.5*avg_den/avg_proxy*100:5.1f}% "
          f"{0.5*avg_cong/avg_proxy*100:5.1f}%")

    # 2. Benchmark triage
    print("\n## 2. Benchmark Triage vs RePlAce")
    gaps = []
    for r in all_results:
        rep = REPLACE_BASELINES.get(r["name"], 0)
        gap = (r["final_costs"]["proxy_cost"] - rep) / rep * 100
        gaps.append((r["name"], gap, r["final_costs"]["proxy_cost"], rep))
    gaps.sort(key=lambda x: x[1])

    winning = [(n, g, p, rp) for n, g, p, rp in gaps if g <= 0]
    close = [(n, g, p, rp) for n, g, p, rp in gaps if 0 < g <= 5]
    hard_loss = [(n, g, p, rp) for n, g, p, rp in gaps if g > 5]

    print(f"\n  WINNING ({len(winning)}): ", ", ".join(f"{n} ({g:+.1f}%)" for n, g, _, _ in winning))
    print(f"  CLOSE ({len(close)}):   ", ", ".join(f"{n} ({g:+.1f}%)" for n, g, _, _ in close))
    print(f"  HARD LOSS ({len(hard_loss)}): ", ", ".join(f"{n} ({g:+.1f}%)" for n, g, _, _ in hard_loss))

    # Structural correlation
    print("\n  Structural correlations with gap:")
    gap_vals = [g for _, g, _, _ in gaps]
    for prop in ["num_hard", "utilization", "num_nets", "num_pairs"]:
        prop_vals = [r["structure"][prop] for r in all_results]
        corr, pval = stats.spearmanr(prop_vals, gap_vals)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"    {prop:>15}: rho={corr:+.3f}, p={pval:.3f} {sig}")

    # 3. Surrogate accuracy
    print("\n## 3. Surrogate Accuracy")
    all_surr = []
    all_real = []
    per_bm_corr = {}
    for r in all_results:
        if r["surrogate_samples"]:
            surrs = [s["surrogate"] for s in r["surrogate_samples"]]
            reals = [s["real_proxy"] for s in r["surrogate_samples"]]
            all_surr.extend(surrs)
            all_real.extend(reals)
            if len(surrs) >= 3:
                rho, _ = stats.spearmanr(surrs, reals)
                per_bm_corr[r["name"]] = rho

    if all_surr:
        global_rho, global_p = stats.spearmanr(all_surr, all_real)
        print(f"  Global Spearman rho: {global_rho:.4f} (p={global_p:.2e}, n={len(all_surr)})")
        print(f"  Mean absolute error: {np.mean(np.abs(np.array(all_surr) - np.array(all_real))):.4f}")
        print(f"  Mean bias (surr - real): {np.mean(np.array(all_surr) - np.array(all_real)):+.4f}")

        if per_bm_corr:
            print(f"\n  Per-benchmark Spearman rho:")
            for bname, rho in sorted(per_bm_corr.items()):
                print(f"    {bname:>10}: {rho:+.4f}")
            print(f"    {'AVG':>10}: {np.mean(list(per_bm_corr.values())):+.4f}")
    else:
        print("  No surrogate samples collected!")

    # 4. Convergence summary
    print("\n## 4. Convergence Summary")
    print(f"{'Benchmark':>10} {'Init':>8} {'Final':>8} {'Delta':>8} {'Impr':>5} {'Surr Evals':>10}")
    print("-" * 55)
    for r in all_results:
        ic = r["init_costs"]["proxy_cost"]
        fc = r["final_costs"]["proxy_cost"]
        print(f"{r['name']:>10} {ic:8.4f} {fc:8.4f} {ic-fc:8.4f} "
              f"{r['improvements']:>5} {r['surrogate_evals']:>10}")

    # 5. Phase timing
    print("\n## 5. Phase Timing (seconds)")
    print(f"{'Benchmark':>10} {'Load':>6} {'SDF':>6} {'Assign':>7} {'LP Build':>8} "
          f"{'LP Solve':>8} {'Surr':>6} {'Nav':>7} {'Total':>7}")
    print("-" * 75)
    for r in all_results:
        pt = r["phase_times"]
        total = sum(v for k, v in pt.items() if k != "accuracy_sampling")
        print(f"{r['name']:>10} {pt['load']:6.2f} {pt['sdf_init']:6.2f} "
              f"{pt['assignment']:7.3f} {pt['lp_build']:8.3f} "
              f"{pt['lp_solve']:8.2f} {pt['surrogate_build']:6.3f} "
              f"{pt['navigation']:7.1f} {total:7.1f}")

    # Top 3 time sinks
    avg_phases = defaultdict(float)
    for r in all_results:
        for phase, t in r["phase_times"].items():
            if phase != "accuracy_sampling":
                avg_phases[phase] += t / len(all_results)
    top3 = sorted(avg_phases.items(), key=lambda x: -x[1])[:3]
    print(f"\n  Top 3 time sinks (avg per benchmark):")
    for phase, t in top3:
        print(f"    {phase:>20}: {t:.2f}s ({t/sum(avg_phases.values())*100:.1f}%)")

    print(f"\nAll CSVs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
