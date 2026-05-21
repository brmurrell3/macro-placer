"""Analyze structural attributes vs v2-vs-E138 winner for E157."""

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = os.path.join(HERE, "..", "attrs.json")

with open(ATTRS) as f:
    recs = json.load(f)

# Print the table
print("=" * 130)
print(f"{'bench':6} {'macros':>7} {'hard':>5} {'soft':>5} {'nets':>6} {'cells':>6} {'cw':>6} {'ch':>6} "
      f"{'hd%':>5} {'td%':>5} {'avgN':>5} {'v2':>7} {'e138':>7} {'win':>5} {'diff':>8}")
print("-" * 130)
for r in recs:
    print(f"{r['bench']:6} {r['num_macros']:7d} {r['num_hard']:5d} {r['num_soft']:5d} {r['num_nets']:6d} "
          f"{r['grid_cells']:6d} {r['canvas_w']:6.1f} {r['canvas_h']:6.1f} "
          f"{r['hard_density']*100:5.1f} {r['total_density']*100:5.1f} {r['avg_net_size']:5.2f} "
          f"{r['v2']:7.4f} {r['e138']:7.4f} {r['winner']:>5} {r['v2_minus_e138']:+8.4f}")
print("=" * 130)

# Print summary stats: v2-winners vs e138-winners
v2_recs = [r for r in recs if r["winner"] == "v2"]
e138_recs = [r for r in recs if r["winner"] == "e138"]
print(f"\nv2-winners ({len(v2_recs)}): {sorted(r['bench'] for r in v2_recs)}")
print(f"e138-winners ({len(e138_recs)}): {sorted(r['bench'] for r in e138_recs)}")

# Means per group for each attr
attrs_to_check = [
    "num_macros", "num_hard", "num_soft", "num_nets", "num_ports",
    "grid_rows", "grid_cols", "grid_cells",
    "canvas_area", "hard_area", "soft_area",
    "hard_density", "total_density", "avg_net_size",
]
print(f"\n{'attr':22} {'v2-mean':>12} {'e138-mean':>12} {'ratio':>8} {'v2-med':>10} {'e138-med':>10}")
print("-" * 90)
import statistics
for a in attrs_to_check:
    vm = statistics.mean(r[a] for r in v2_recs)
    em = statistics.mean(r[a] for r in e138_recs)
    vmd = statistics.median(r[a] for r in v2_recs)
    emd = statistics.median(r[a] for r in e138_recs)
    ratio = em / vm if vm else float("inf")
    print(f"{a:22} {vm:12.3f} {em:12.3f} {ratio:8.3f} {vmd:10.3f} {emd:10.3f}")

# Try a single-threshold rule on each attr: "pick v2 if attr < t, else e138"
# Score = sum of accuracy over the 17 benches given each threshold candidate.
print(f"\n=== single-threshold rule scoring ===")
print(f"{'attr':22} {'best_thr':>12} {'correct':>8} {'lift_vs_v2':>12} {'lift_vs_e138':>14}")
print("-" * 90)

# Sums
sum_v2 = sum(r["v2"] for r in recs)
sum_e138 = sum(r["e138"] for r in recs)
n = len(recs)
print(f"baseline avg v2 = {sum_v2/n:.5f}   E138 = {sum_e138/n:.5f}")

# Oracle: per-bench best
oracle = sum(min(r["v2"], r["e138"]) for r in recs) / n
print(f"oracle (per-bench best) = {oracle:.5f}\n")

for a in attrs_to_check:
    # Try every value in recs as the threshold split
    vals = sorted(set(r[a] for r in recs))
    best = None
    for t in vals + [vals[-1] + 1]:
        # Rule A: attr < t -> v2, else e138
        score_A = 0.0
        correct_A = 0
        for r in recs:
            if r[a] < t:
                score_A += r["v2"]
                if r["winner"] == "v2": correct_A += 1
            else:
                score_A += r["e138"]
                if r["winner"] == "e138": correct_A += 1
        score_A /= n
        # Rule B: attr < t -> e138, else v2
        score_B = 0.0
        correct_B = 0
        for r in recs:
            if r[a] < t:
                score_B += r["e138"]
                if r["winner"] == "e138": correct_B += 1
            else:
                score_B += r["v2"]
                if r["winner"] == "v2": correct_B += 1
        score_B /= n
        for direction, score, correct in [("A", score_A, correct_A), ("B", score_B, correct_B)]:
            if best is None or score < best[1]:
                best = (t, score, correct, direction)
    t, score, correct, direction = best
    print(f"{a:22} thr={t:10.4g} {correct:3d}/17 ({correct/n*100:.0f}%) "
          f"avg={score:.5f}  lift_v2={(sum_v2/n - score)*100/(sum_v2/n):+.3f}%  "
          f"lift_e138={(sum_e138/n - score)*100/(sum_e138/n):+.3f}%  dir={direction}")
