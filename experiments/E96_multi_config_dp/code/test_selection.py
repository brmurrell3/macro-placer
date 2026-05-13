"""Tiny test to verify multi_dp_basin's candidate selection.

The v2 polish log on ibm10 shows winner=dw_high despite baseline_auto
having a lower score. This script tests the sort logic with fixtures.
"""
candidates = [
    {"label": "baseline_auto", "legalize_proxy": 1.4433, "legalize_ovl": 0,
     "score": 1.4433 + 10.0 * 0},
    {"label": "dw_low", "legalize_proxy": 2.2561, "legalize_ovl": 0,
     "score": 2.2561 + 10.0 * 0},
    {"label": "dw_high", "legalize_proxy": 1.5902, "legalize_ovl": 0,
     "score": 1.5902 + 10.0 * 0},
    {"label": "seed_2024", "legalize_proxy": 3.2612, "legalize_ovl": 0,
     "score": 3.2612 + 10.0 * 0},
]

# Same sort logic as multi_dp_basin
candidates.sort(key=lambda c: c["score"])
print("After sort:")
for c in candidates:
    print(f"  {c['label']:15s} score={c['score']:.4f}")

winner = candidates[0]
print(f"\nWinner: {winner['label']} (legal={winner['legalize_proxy']:.4f}, ovl={winner['legalize_ovl']})")
assert winner["label"] == "baseline_auto", f"BUG: expected baseline_auto, got {winner['label']}"
print("\n✓ Selection logic is correct in isolation.")
