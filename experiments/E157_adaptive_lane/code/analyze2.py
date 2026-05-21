"""Deeper analysis: 2-attribute rules + check which benches are misclassified."""

import json
import os
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = os.path.join(HERE, "..", "attrs.json")

with open(ATTRS) as f:
    recs = json.load(f)

n = len(recs)
sum_v2 = sum(r["v2"] for r in recs)
sum_e138 = sum(r["e138"] for r in recs)
print(f"baseline avg v2 = {sum_v2/n:.5f}   E138 = {sum_e138/n:.5f}")
print(f"oracle (per-bench best) = {sum(min(r['v2'], r['e138']) for r in recs)/n:.5f}\n")

# ===== Best single rule: hard_density < 0.4997 -> v2 else e138 =====
# Check which benches it misclassifies
print("=== hard_density < 0.4997 -> v2, else -> e138 ===")
print(f"{'bench':6} {'hard_d':>7} {'win':>5} {'pred':>5} {'correct':>8} {'v2':>7} {'e138':>7}")
for r in recs:
    pred = "v2" if r["hard_density"] < 0.4997 else "e138"
    ok = "Y" if pred == r["winner"] else "MISS"
    print(f"{r['bench']:6} {r['hard_density']:7.4f} {r['winner']:>5} {pred:>5} {ok:>8} {r['v2']:7.4f} {r['e138']:7.4f}")

# Compute adaptive avg
adaptive = sum(r["v2"] if r["hard_density"] < 0.4997 else r["e138"] for r in recs) / n
print(f"\nadaptive avg = {adaptive:.5f}")
print(f"v2-only      = {sum_v2/n:.5f}")
print(f"e138-only    = {sum_e138/n:.5f}")
print(f"oracle       = {sum(min(r['v2'], r['e138']) for r in recs)/n:.5f}")

# ===== Try 2-attribute combo: split, then a secondary rule on misses =====
# Actually, let's just search all pairs for a hyperplane (sign(a*x + b*y - c) > 0 -> v2 else e138)
# Brute force over thresholds
print("\n=== best 2-attr linear split (a*A + b*B < c) ===")

attrs_to_check = [
    "num_macros", "num_hard", "num_soft", "num_nets", "num_ports",
    "grid_rows", "grid_cols", "grid_cells",
    "canvas_area", "hard_area", "soft_area",
    "hard_density", "avg_net_size",
]
best = None
# Try (attrA - kappa * attrB < t) -> v2 else e138
# Or simpler: hard_density and one other
# Try just paired thresholds: (A < tA) AND (B op tB) -> v2 else e138
results_pair = []
for a, b in itertools.combinations(attrs_to_check, 2):
    a_vals = sorted(set(r[a] for r in recs))
    b_vals = sorted(set(r[b] for r in recs))
    for ta in a_vals + [a_vals[-1] + 1]:
        for tb in b_vals + [b_vals[-1] + 1]:
            for op_a in ["<", ">="]:
                for op_b in ["<", ">="]:
                    for combo in ["AND", "OR"]:
                        for choose in ["v2", "e138"]:
                            total = 0.0
                            correct = 0
                            for r in recs:
                                a_match = (r[a] < ta) if op_a == "<" else (r[a] >= ta)
                                b_match = (r[b] < tb) if op_b == "<" else (r[b] >= tb)
                                hit = (a_match and b_match) if combo == "AND" else (a_match or b_match)
                                pred = choose if hit else ("v2" if choose == "e138" else "e138")
                                total += r[pred]
                                if pred == r["winner"]:
                                    correct += 1
                            avg = total / n
                            if best is None or avg < best[0]:
                                best = (avg, a, ta, op_a, b, tb, op_b, combo, choose, correct)

avg, a, ta, op_a, b, tb, op_b, combo, choose, correct = best
print(f"best: ({a} {op_a} {ta:.4g}) {combo} ({b} {op_b} {tb:.4g}) -> {choose} else other")
print(f"      correct={correct}/17 avg={avg:.5f}  lift_vs_v2={(sum_v2/n - avg)*100/(sum_v2/n):+.3f}%")

# Show which benches are picked
for r in recs:
    a_match = (r[a] < ta) if op_a == "<" else (r[a] >= ta)
    b_match = (r[b] < tb) if op_b == "<" else (r[b] >= tb)
    hit = (a_match and b_match) if combo == "AND" else (a_match or b_match)
    pred = choose if hit else ("v2" if choose == "e138" else "e138")
    ok = "Y" if pred == r["winner"] else "MISS"
    print(f"{r['bench']:6} winner={r['winner']:5} pred={pred:5} {ok:>5} ({a}={r[a]:.4g} {b}={r[b]:.4g})")

# What if we just consider: does the 14/17 single-attr rule miss small or large benches?
print("\n=== Bottom-line view ===")
print(f"v2-only-all-17 avg     : {sum_v2/n:.5f}")
print(f"E138-only-all-17 avg   : {sum_e138/n:.5f}")
print(f"hard_density 0.50 rule : {sum(r['v2'] if r['hard_density'] < 0.4997 else r['e138'] for r in recs)/n:.5f}")
print(f"per-bench oracle       : {sum(min(r['v2'], r['e138']) for r in recs)/n:.5f}")
