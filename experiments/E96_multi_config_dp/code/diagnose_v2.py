"""Diagnose the v2 selection bug.

Loads the v2 ibm10 polish JSON when written and inspects the candidates list.
If candidates[0] is dw_high but its score is higher than baseline's, the sort
was wrong. If candidates[0] is baseline_auto, the bug is in the printout
(elsewhere in the code).
"""
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    json_path = "/home/ubuntu/macro-place-challenge-2026/experiments/E96_multi_config_dp/results/multi_dp_polish_ibm10_K4.json"
else:
    json_path = sys.argv[1]

try:
    d = json.load(open(json_path))
except FileNotFoundError:
    print(f"JSON not found at {json_path}")
    sys.exit(1)

cands = d["basin_stats"]["candidates"]
print(f"basin_stats.winner_label: {d['basin_stats']['winner_label']}")
print(f"basin_stats.winner_legalize_proxy: {d['basin_stats']['winner_legalize_proxy']:.5f}")
print(f"final_proxy (polish result): {d['final_proxy']:.5f}")
print(f"legal_proxy (winner's basin): {d['legal_proxy']:.5f}")
print()
print(f"Stored candidates ({len(cands)} items, in sort order):")
for i, c in enumerate(cands):
    print(f"  [{i}] {c['label']:18s} legal={c['legalize_proxy']:.4f} ovl={c['legalize_ovl']} score={c['score']:.4f}")

# Verify sort is monotonic
scores = [c['score'] for c in cands]
print()
print(f"Scores in stored order: {scores}")
print(f"Sorted scores: {sorted(scores)}")
print(f"Stored == sorted? {scores == sorted(scores)}")

if scores != sorted(scores):
    print(f"\n⚠ BUG: candidates are not in sorted-by-score order!")
    # Find the right winner
    sorted_cands = sorted(cands, key=lambda c: c["score"])
    print(f"Expected winner: {sorted_cands[0]['label']} (score {sorted_cands[0]['score']:.4f})")
    print(f"Actual winner stored: {cands[0]['label']} (score {cands[0]['score']:.4f})")
elif cands[0]['label'] != d['basin_stats']['winner_label']:
    print(f"\n⚠ ANOMALY: candidates[0]['label'] = {cands[0]['label']} but winner_label = {d['basin_stats']['winner_label']}")
else:
    print(f"\n✓ Candidates sorted correctly and winner consistent.")
