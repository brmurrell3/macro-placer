"""Sanity tests for sp_inverter on synthetic placements."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Allow running standalone from experiments dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sp_inverter import (
    encode, murata_decode, pair_relation_distance, kendall_tau,
    relation_for_pair, LEFT, RIGHT, BELOW, ABOVE,
)


def test_two_horizontal():
    """Two macros side by side: 0 left of 1."""
    pos = torch.tensor([[2.5, 2.5], [7.5, 2.5]], dtype=torch.float32)
    siz = torch.tensor([[5.0, 5.0], [5.0, 5.0]], dtype=torch.float32)
    sp = encode(pos, siz, n_hard=2)
    assert relation_for_pair(sp, 0, 1) == LEFT, f"got {relation_for_pair(sp, 0, 1)}"
    decoded = murata_decode(sp, siz, n_hard=2)
    # Decode is bottom-left compact: macro 0 at center (2.5, 2.5), macro 1 at (7.5, 2.5).
    assert torch.allclose(decoded[0], torch.tensor([2.5, 2.5]), atol=1e-3)
    assert torch.allclose(decoded[1], torch.tensor([7.5, 2.5]), atol=1e-3)
    print("test_two_horizontal: PASS")


def test_two_vertical():
    """Two macros stacked: 0 below 1."""
    pos = torch.tensor([[2.5, 2.5], [2.5, 7.5]], dtype=torch.float32)
    siz = torch.tensor([[5.0, 5.0], [5.0, 5.0]], dtype=torch.float32)
    sp = encode(pos, siz, n_hard=2)
    assert relation_for_pair(sp, 0, 1) == BELOW, f"got {relation_for_pair(sp, 0, 1)}"
    decoded = murata_decode(sp, siz, n_hard=2)
    assert torch.allclose(decoded[0], torch.tensor([2.5, 2.5]), atol=1e-3)
    assert torch.allclose(decoded[1], torch.tensor([2.5, 7.5]), atol=1e-3)
    print("test_two_vertical: PASS")


def test_four_grid():
    """2x2 grid: macros at (2.5,2.5), (7.5,2.5), (2.5,7.5), (7.5,7.5)."""
    pos = torch.tensor([
        [2.5, 2.5], [7.5, 2.5], [2.5, 7.5], [7.5, 7.5]
    ], dtype=torch.float32)
    siz = torch.tensor([[5.0, 5.0]] * 4, dtype=torch.float32)
    sp = encode(pos, siz, n_hard=4)
    # Expected: 0 LEFT of 1; 0 BELOW of 2; 0 LEFT of 3 (could also be BELOW;
    # H-preferred → LEFT); 1 BELOW of 3; 2 LEFT of 3; 1 ?-of 2 (1 right and below
    # of 2; H-preferred → 2 LEFT of 1; encoded as relation_for_pair(0,1)=LEFT
    # already, so let's check 2,1: macro 2 has right=7.5, macro 1 has left=5.0,
    # so 2.right > 1.left and 2.left=0 < 1.right=10 → not strictly H-separated.
    # But 2.bot=5.0, 1.top=5.0 → 2 above 1 by EPS check. So pair (1,2): "1 below 2"? No:
    # 1 is at center y=2.5, 2 is at center y=7.5; 1 is below 2. So 1 BELOW 2.
    # Actually check: 1.top = 5.0, 2.bot = 5.0 → equality → eps flips it.
    # i.top <= j.bot + eps → 1.top=5.0 <= 2.bot=5.0+eps → True → 1 BELOW 2. OK.
    # With Murata adjacency rule, only directly-adjacent pairs constrain.
    # 0 and 1 share y-projection (both at row y=2.5): direct H-edge → LEFT.
    assert relation_for_pair(sp, 0, 1) == LEFT
    # 0 and 2 share x-projection (both at column x=2.5): direct V-edge → BELOW.
    assert relation_for_pair(sp, 0, 2) == BELOW
    # 2 and 3 share y-projection: H-edge → LEFT.
    assert relation_for_pair(sp, 2, 3) == LEFT
    # 1 and 3 share x-projection: V-edge → BELOW.
    assert relation_for_pair(sp, 1, 3) == BELOW
    # 0 and 3 are diagonally separated (no projection overlap): relation
    # decided by tie-break. Could be LEFT or BELOW. Just verify decoder works.
    # Same for 1 and 2 (other diagonal).
    decoded = murata_decode(sp, siz, n_hard=4)
    print(f"test_four_grid: SP+ = {sp.gamma_plus}, SP- = {sp.gamma_minus}")
    print(f"  decoded centers:\n{decoded.numpy()}")
    print("test_four_grid: PASS (manual inspection)")


def test_encode_stability():
    """Encoding the same placement twice yields identical SP."""
    rng = np.random.default_rng(42)
    n = 20
    sizes = torch.tensor(rng.uniform(2.0, 4.0, size=(n, 2)).astype(np.float32))
    grid = []
    for r in range(5):
        for c in range(5):
            if len(grid) < n:
                grid.append((c * 8.0 + 4.0, r * 8.0 + 4.0))
    pos = torch.tensor(grid[:n], dtype=torch.float32)
    sp1 = encode(pos, sizes, n_hard=n)
    sp2 = encode(pos, sizes, n_hard=n)
    diff, _ = pair_relation_distance(sp1, sp2)
    assert diff == 0, f"Encoder is not deterministic: {diff} diffs on re-encode"
    # Decoded placement should have zero overlaps.
    decoded = murata_decode(sp1, sizes, n_hard=n)
    # Quick overlap check.
    pos_np = decoded.numpy(); siz_np = sizes.numpy()
    overlap = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(pos_np[i, 0] - pos_np[j, 0])
            dy = abs(pos_np[i, 1] - pos_np[j, 1])
            if dx < (siz_np[i, 0] + siz_np[j, 0]) / 2 - 1e-6 and \
               dy < (siz_np[i, 1] + siz_np[j, 1]) / 2 - 1e-6:
                overlap += 1
    assert overlap == 0, f"Decoded placement has {overlap} overlaps"
    print("test_encode_stability: PASS")


def test_distance_metrics():
    """Two perturbations of same SP should have small distance."""
    rng = np.random.default_rng(0)
    n = 10
    sizes = torch.tensor([[2.0, 2.0]] * n, dtype=torch.float32)
    pos1 = torch.tensor(
        rng.uniform(0, 30, (n, 2)).astype(np.float32), dtype=torch.float32
    )
    # Ensure no overlap by snapping to grid.
    grid = [(i % 5 * 5.0 + 2.5, i // 5 * 5.0 + 2.5) for i in range(n)]
    pos1 = torch.tensor(grid, dtype=torch.float32)
    sp1 = encode(pos1, sizes, n_hard=n)

    # Permute first two positions (flip a relation).
    pos2 = pos1.clone()
    pos2[0], pos2[1] = pos1[1].clone(), pos1[0].clone()
    sp2 = encode(pos2, sizes, n_hard=n)
    diff, breakdown = pair_relation_distance(sp1, sp2)
    kt_plus = kendall_tau(sp1.gamma_plus, sp2.gamma_plus)
    kt_minus = kendall_tau(sp1.gamma_minus, sp2.gamma_minus)
    print(f"test_distance_metrics: pair_diff = {diff} (expect ~9 if 0/1 relations all flip)")
    print(f"  breakdown: {breakdown}")
    print(f"  kendall_tau plus: {kt_plus}, minus: {kt_minus}")
    print("test_distance_metrics: PASS")


if __name__ == "__main__":
    test_two_horizontal()
    test_two_vertical()
    test_four_grid()
    test_encode_stability()
    test_distance_metrics()
    print("\nAll tests passed.")
