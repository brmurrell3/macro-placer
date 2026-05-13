"""Test multi_dp_basin's exact selection logic with v2's actual log values.

Reproduces the candidates list with the values seen in the v2 polish log.
If sort+pick correctly selects baseline_auto, then the v2 bug is somewhere
else (race condition? state corruption?). If it picks dw_high, the bug is
in the logic.
"""

# From v2 polish log:
# baseline_auto: basin 1.3681/106 → legal 1.4433/0 score=1.4433
# dw_low:        basin 2.2351/598 → legal 2.2561/0 score=2.2561
# dw_high:       basin 1.3602/760 → legal 1.5902/0 score=1.5902
# seed_2024:     basin 3.3602/659 → legal 3.2612/0 score=3.2612

import random


def append_with_torch_tensors():
    """Simulate the dict construction with torch tensor placement values."""
    try:
        import torch
        placements = [torch.zeros(786, 2) for _ in range(4)]
    except ImportError:
        placements = [None for _ in range(4)]

    candidates = []
    for i, (label, legal_proxy, leg_ovl, placement) in enumerate([
        ("baseline_auto", 1.4433, 0, placements[0]),
        ("dw_low",        2.2561, 0, placements[1]),
        ("dw_high",       1.5902, 0, placements[2]),
        ("seed_2024",     3.2612, 0, placements[3]),
    ]):
        score = legal_proxy + 10.0 * leg_ovl
        candidates.append({
            "label": label,
            "raw_placement": placement,
            "legal_placement": placement,
            "basin_proxy": 1.0,
            "basin_ovl": 100,
            "legalize_proxy": legal_proxy,
            "legalize_ovl": leg_ovl,
            "score": score,
            "dp_wall": 100,
            "leg_wall": 5,
            "total_wall": 105,
        })

    print("Before sort:")
    for c in candidates:
        print(f"  {c['label']:18s} score={c['score']:.4f}")

    candidates.sort(key=lambda c: c["score"])
    print("\nAfter sort:")
    for c in candidates:
        print(f"  {c['label']:18s} score={c['score']:.4f}")

    winner = candidates[0]
    print(f"\nWinner: {winner['label']} (score={winner['score']:.4f})")
    assert winner["label"] == "baseline_auto", f"BUG: expected baseline_auto, got {winner['label']}"
    print("✓ Selection correct.")


if __name__ == "__main__":
    append_with_torch_tensors()
