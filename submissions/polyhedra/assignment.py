"""
Pairwise L/R/A/B assignment extraction from legal placements.

An "assignment" maps each hard macro pair (i, k) to a separation direction,
defining which polyhedron the placement lives in.
"""

import numpy as np

# Direction constants — used everywhere as the canonical encoding
L = 0  # i is left of k
R = 1  # i is right of k
B = 2  # i is below k
A = 3  # i is above k

DIR_NAMES = ["L", "R", "B", "A"]

# Opposite direction lookup
OPPOSITE = {L: R, R: L, B: A, A: B}


def extract_assignment(positions: np.ndarray, sizes: np.ndarray,
                       hard_indices: np.ndarray) -> dict:
    """
    Extract pairwise L/R/A/B assignment from a legal (non-overlapping) placement.

    For each hard macro pair (i, k) where i < k, determines which separation
    constraint is active (tightest).

    Args:
        positions: [N, 2] center coordinates (x, y) for all macros
        sizes: [N, 2] (width, height) for all macros
        hard_indices: indices of hard macros to consider

    Returns:
        dict mapping (i, k) -> direction in {L, R, B, A}
    """
    n = len(hard_indices)
    pos = positions[hard_indices]
    sz = sizes[hard_indices]

    # Compute all pairwise gaps at once
    x, y = pos[:, 0], pos[:, 1]
    w, h = sz[:, 0], sz[:, 1]
    hw, hh = w / 2, h / 2

    right_x = x + hw
    left_x = x - hw
    top_y = y + hh
    bottom_y = y - hh

    ai, bi = np.triu_indices(n, k=1)

    gap_l = left_x[bi] - right_x[ai]
    gap_r = left_x[ai] - right_x[bi]
    gap_b = bottom_y[bi] - top_y[ai]
    gap_a = bottom_y[ai] - top_y[bi]

    gaps = np.stack([gap_l, gap_r, gap_b, gap_a], axis=1)
    best_dirs = np.argmax(gaps, axis=1)

    assignment = {}
    for idx in range(len(ai)):
        i = int(hard_indices[ai[idx]])
        k = int(hard_indices[bi[idx]])
        assignment[(i, k)] = int(best_dirs[idx])

    return assignment
