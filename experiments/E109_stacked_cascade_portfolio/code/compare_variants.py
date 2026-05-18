"""Compare all stacked variants from overnight runs.

Reads /tmp logs from concurrent --all runs:
  - stacked_periphery (original): 1.0575
  - stacked_periphery_k3 (K_eps=3)
  - stacked_periphery_aws-cpu (verification)
  - cascade_tabu_stacked (tabu eigvecs) — if available
  - cascade_no_e41_deep (skip E41) — if available

Computes per-bench winners, aggregate, identifies champion candidate.
"""
import json
import re
import sys
from pathlib import Path


def parse_log(path, name):
    if not Path(path).exists():
        return None
    text = Path(path).read_text()
    # Get per-bench results from summary table
    rows = re.findall(r"^\s+(ibm\d+|\w+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+", text, re.M)
    if not rows:
        return None
    benches = {b: float(p) for b, p in rows if b.startswith("ibm")}
    avg_match = re.search(r"^\s+AVG\s+([\d.]+)", text, re.M)
    avg = float(avg_match.group(1)) if avg_match else None
    return {"name": name, "benches": benches, "avg": avg}


sources = [
    ("/tmp/stacked_periphery_all.log", "champion-stacked_periphery"),
    ("/tmp/stacked_periphery_k3_all.log", "K_eps=3"),
    ("/tmp/awscpu_stacked_periphery.log", "aws-cpu verify"),
    ("/tmp/cascade_tabu_stacked_all.log", "tabu_stacked"),
    ("/tmp/cascade_no_e41_deep_all.log", "no_e41_deep"),
]

results = []
for path, name in sources:
    r = parse_log(path, name)
    if r:
        results.append(r)
        print(f"{name}: avg={r['avg']:.5f} ({len(r['benches'])}/17 benches)")
    else:
        print(f"{name}: NO DATA")

if not results:
    print("No results to compare")
    sys.exit(0)

# Per-bench table comparing all variants
all_benches = sorted({b for r in results for b in r['benches']})
print(f"\n{'bench':<10} " + " ".join(f"{r['name'][:14]:>15}" for r in results))
for b in all_benches:
    row = []
    for r in results:
        v = r['benches'].get(b)
        row.append(f"{v:>15.4f}" if v is not None else f"{'—':>15}")
    print(f"{b:<10} " + " ".join(row))

# Best per-bench (per-bench oracle — not directly usable but informative)
oracle = {}
for b in all_benches:
    best = min((r['benches'][b], r['name']) for r in results if b in r['benches'])
    oracle[b] = best
oracle_avg = sum(v[0] for v in oracle.values()) / len(oracle)
print(f"\nPer-bench oracle avg (NOT usable, per-bench tuning): {oracle_avg:.5f}")
print(f"Oracle wins:")
from collections import Counter
src_counts = Counter(v[1] for v in oracle.values())
for src, cnt in src_counts.most_common():
    print(f"  {src}: {cnt} benches")
