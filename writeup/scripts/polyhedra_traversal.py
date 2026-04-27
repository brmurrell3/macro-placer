#!/usr/bin/env python3
"""
Polyhedra traversal analysis: quantify how many pairwise L/R/A/B
assignments change between SDF initialization and DPO output.

Tests the "penalty-as-barrier-crossing" claim: if DPO changes many
pair assignments, it genuinely crosses between polyhedra in the
feasible region rather than staying in the SDF-initialized polyhedron.

Usage:
    uv run python scripts/polyhedra_traversal.py
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "polyhedra"))

from macro_place.loader import load_benchmark_from_dir
from assignment import extract_assignment, DIR_NAMES


def extract_assignment_with_overlap(positions, sizes, hard_indices):
    """
    Extract pairwise L/R/A/B assignment, flagging overlapping/ambiguous pairs.

    Returns:
        dict mapping (i, k) -> direction in {0=L, 1=R, 2=B, 3=A, 4=overlap}
    """
    n = len(hard_indices)
    pos = positions[hard_indices]
    sz = sizes[hard_indices]

    x, y = pos[:, 0], pos[:, 1]
    w, h = sz[:, 0], sz[:, 1]
    hw, hh = w / 2, h / 2

    right_x = x + hw
    left_x = x - hw
    top_y = y + hh
    bottom_y = y - hh

    ai, bi = np.triu_indices(n, k=1)

    # Gaps: positive means separated, negative means overlapping in that dim
    gap_l = left_x[bi] - right_x[ai]   # i left of k
    gap_r = left_x[ai] - right_x[bi]   # i right of k
    gap_b = bottom_y[bi] - top_y[ai]   # i below k
    gap_a = bottom_y[ai] - top_y[bi]   # i above k

    gaps = np.stack([gap_l, gap_r, gap_b, gap_a], axis=1)
    best_dirs = np.argmax(gaps, axis=1)
    best_gaps = np.max(gaps, axis=1)

    assignment = {}
    for idx in range(len(ai)):
        i = int(hard_indices[ai[idx]])
        k = int(hard_indices[bi[idx]])
        if best_gaps[idx] < -0.01:
            assignment[(i, k)] = 4  # overlapping
        else:
            assignment[(i, k)] = int(best_dirs[idx])

    return assignment


def run_sdf_init(benchmark):
    """Run SDF placer to get initial positions."""
    import importlib.util
    sdf_path = ROOT / "submissions" / "polyhedra" / "init" / "sdf.py"
    spec = importlib.util.spec_from_file_location("sdf", str(sdf_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SDFPlacer(seed=42).place(benchmark)


def run_dpo(benchmark):
    """Run DPO placer to get final positions."""
    import importlib.util
    dpo_path = ROOT / "submissions" / "dpo" / "placer.py"
    spec = importlib.util.spec_from_file_location("dpo", str(dpo_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DPOPlacer(seed=42, use_sdf_init=True).place(benchmark)


def analyze_benchmark(name):
    """Run full traversal analysis on one benchmark."""
    print(f"\n{'='*70}")
    print(f"  POLYHEDRA TRAVERSAL ANALYSIS: {name}")
    print(f"{'='*70}")

    # Load benchmark
    bench_dir = f"external/MacroPlacement/Testcases/ICCAD04/{name}"
    benchmark, plc = load_benchmark_from_dir(bench_dir)

    n_hard = benchmark.num_hard_macros
    hard_indices = np.arange(n_hard)
    movable_hard = hard_indices[~benchmark.macro_fixed[:n_hard].numpy()]
    sizes = benchmark.macro_sizes.numpy()

    print(f"  Hard macros: {n_hard}  (movable: {len(movable_hard)})")
    n_pairs = len(movable_hard) * (len(movable_hard) - 1) // 2
    print(f"  Movable pairs: {n_pairs}")

    # Step 1: SDF init
    print(f"\n  Running SDF init...")
    t0 = time.time()
    sdf_pos = run_sdf_init(benchmark)
    sdf_time = time.time() - t0
    sdf_pos_np = sdf_pos.numpy()
    print(f"  SDF init done in {sdf_time:.1f}s")

    # Step 2: DPO optimization
    print(f"\n  Running DPO placer...")
    t0 = time.time()
    dpo_pos = run_dpo(benchmark)
    dpo_time = time.time() - t0
    dpo_pos_np = dpo_pos.numpy()
    print(f"  DPO done in {dpo_time:.1f}s")

    # Step 3: Extract assignments
    sdf_assign = extract_assignment_with_overlap(sdf_pos_np, sizes, movable_hard)
    dpo_assign = extract_assignment_with_overlap(dpo_pos_np, sizes, movable_hard)

    # Step 4: Compare
    dir_labels = ["L", "R", "B", "A", "OVL"]
    changed = 0
    unchanged = 0
    transition_counts = Counter()
    sdf_dir_counts = Counter()
    dpo_dir_counts = Counter()

    for pair in sdf_assign:
        s = sdf_assign[pair]
        d = dpo_assign[pair]
        sdf_dir_counts[s] += 1
        dpo_dir_counts[d] += 1
        if s != d:
            changed += 1
            transition_counts[(s, d)] += 1
        else:
            unchanged += 1

    total = changed + unchanged
    pct = 100.0 * changed / total if total > 0 else 0.0

    print(f"\n  --- RESULTS ---")
    print(f"  Total movable pairs:  {total}")
    print(f"  Changed assignments:  {changed}")
    print(f"  Unchanged:            {unchanged}")
    print(f"  Change fraction:      {pct:.1f}%")

    # Direction distributions
    print(f"\n  Direction distribution:")
    print(f"  {'Dir':<6} {'SDF':<10} {'DPO':<10} {'Delta':<10}")
    for d_idx in range(5):
        s_count = sdf_dir_counts[d_idx]
        d_count = dpo_dir_counts[d_idx]
        delta = d_count - s_count
        s_pct = 100.0 * s_count / total if total > 0 else 0.0
        d_pct = 100.0 * d_count / total if total > 0 else 0.0
        print(f"  {dir_labels[d_idx]:<6} {s_count:>5} ({s_pct:4.1f}%)  "
              f"{d_count:>5} ({d_pct:4.1f}%)  {delta:+d}")

    # Top transition types
    if transition_counts:
        print(f"\n  Top 10 transition types (from -> to):")
        for (s, d), count in transition_counts.most_common(10):
            pct_t = 100.0 * count / changed if changed > 0 else 0.0
            print(f"    {dir_labels[s]} -> {dir_labels[d]}: {count:>5} ({pct_t:4.1f}% of changes)")

    # Displacement statistics
    disp = np.linalg.norm(dpo_pos_np[:n_hard] - sdf_pos_np[:n_hard], axis=1)
    movable_disp = disp[movable_hard]
    canvas_diag = np.sqrt(benchmark.canvas_width**2 + benchmark.canvas_height**2)

    print(f"\n  Macro displacement (SDF -> DPO):")
    print(f"    Mean:   {movable_disp.mean():.1f} um  ({100*movable_disp.mean()/canvas_diag:.1f}% of diagonal)")
    print(f"    Median: {np.median(movable_disp):.1f} um  ({100*np.median(movable_disp)/canvas_diag:.1f}% of diagonal)")
    print(f"    Max:    {movable_disp.max():.1f} um  ({100*movable_disp.max()/canvas_diag:.1f}% of diagonal)")
    print(f"    Min:    {movable_disp.min():.1f} um")
    print(f"    Canvas diagonal: {canvas_diag:.1f} um")

    return {
        "name": name,
        "n_hard": n_hard,
        "n_movable": len(movable_hard),
        "total_pairs": total,
        "changed": changed,
        "pct_changed": pct,
        "mean_disp": float(movable_disp.mean()),
        "mean_disp_frac": float(movable_disp.mean() / canvas_diag),
    }


if __name__ == "__main__":
    os.chdir(ROOT)

    benchmarks = ["ibm01", "ibm10"]
    results = []

    for name in benchmarks:
        r = analyze_benchmark(name)
        results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Bench':<8} {'Macros':<8} {'Pairs':<8} {'Changed':<8} {'%':>6}  {'Mean Disp':>12}")
    for r in results:
        print(f"  {r['name']:<8} {r['n_movable']:<8} {r['total_pairs']:<8} "
              f"{r['changed']:<8} {r['pct_changed']:5.1f}%  "
              f"{r['mean_disp']:.1f}um ({r['mean_disp_frac']*100:.1f}% diag)")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    for r in results:
        if r['pct_changed'] > 30:
            print(f"  {r['name']}: {r['pct_changed']:.0f}% pairs changed -- "
                  f"DPO aggressively crosses polyhedra boundaries.")
        elif r['pct_changed'] > 10:
            print(f"  {r['name']}: {r['pct_changed']:.0f}% pairs changed -- "
                  f"DPO moderately crosses polyhedra boundaries.")
        else:
            print(f"  {r['name']}: {r['pct_changed']:.0f}% pairs changed -- "
                  f"DPO mostly stays within the SDF polyhedron.")
