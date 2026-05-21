"""Leave-one-out cross-validation: does the rule generalize?"""

import json
import os
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = os.path.join(HERE, "..", "attrs.json")

with open(ATTRS) as f:
    recs = json.load(f)

n = len(recs)


def best_single_attr(train):
    """Returns (attr, threshold, direction) where direction is 'A' (attr<t->v2) or 'B' (attr<t->e138)."""
    attrs_to_check = [
        "num_macros", "num_hard", "num_soft", "num_nets", "num_ports",
        "grid_rows", "grid_cols", "grid_cells",
        "canvas_area", "hard_area", "soft_area",
        "hard_density", "avg_net_size",
    ]
    best = None
    nt = len(train)
    for a in attrs_to_check:
        vals = sorted(set(r[a] for r in train))
        for t in vals + [vals[-1] + 1]:
            for direction in ["A", "B"]:
                score = 0.0
                for r in train:
                    if direction == "A":
                        score += r["v2"] if r[a] < t else r["e138"]
                    else:
                        score += r["e138"] if r[a] < t else r["v2"]
                if best is None or score < best[0]:
                    best = (score, a, t, direction)
    return best[1], best[2], best[3]


def predict(r, attr, thr, direction):
    if direction == "A":
        return "v2" if r[attr] < thr else "e138"
    else:
        return "e138" if r[attr] < thr else "v2"


# LOO
print("=== LEAVE-ONE-OUT ===")
print(f"{'held':6} {'attr':16} {'thr':>10} {'dir':>3} {'pred':>5} {'true':>5} {'ok?':>5} {'used_score':>10}")
loo_avg = 0.0
v2_avg = 0.0
e138_avg = 0.0
for held in recs:
    train = [r for r in recs if r["bench"] != held["bench"]]
    attr, thr, direction = best_single_attr(train)
    pred = predict(held, attr, thr, direction)
    used = held["v2"] if pred == "v2" else held["e138"]
    ok = "Y" if pred == held["winner"] else "MISS"
    print(f"{held['bench']:6} {attr:16} {thr:10.4g} {direction:>3} {pred:>5} {held['winner']:>5} {ok:>5} {used:10.4f}")
    loo_avg += used
    v2_avg += held["v2"]
    e138_avg += held["e138"]

print(f"\nLOO avg using best train-single-attr rule = {loo_avg/n:.5f}")
print(f"v2 baseline = {v2_avg/n:.5f}, E138 baseline = {e138_avg/n:.5f}")
print(f"oracle = {sum(min(r['v2'], r['e138']) for r in recs)/n:.5f}")
