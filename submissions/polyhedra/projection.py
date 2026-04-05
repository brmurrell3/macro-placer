"""
Constraint projection and overlap detection for macro placements.

Centralizes the geometry logic that was previously duplicated across
Navigator._project_flip, refine_density, and _perturb_positions.
"""

import numpy as np
from assignment import L, R, B, A


def check_overlaps(positions: np.ndarray, sizes: np.ndarray,
                   n_hard: int, tol: float = 1e-6) -> bool:
    """Fast vectorized overlap check for all hard macro pairs.
    Returns True if ANY overlap exists."""
    pos = positions[:n_hard]
    sz = sizes[:n_hard]

    dx = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
    dy = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
    min_dx = (sz[:, 0:1] + sz[:, 0:1].T) / 2
    min_dy = (sz[:, 1:2] + sz[:, 1:2].T) / 2

    overlap = (dx < min_dx - tol) & (dy < min_dy - tol)
    np.fill_diagonal(overlap, False)
    return bool(np.any(overlap))


def separation_gap(pos: np.ndarray, sizes: np.ndarray,
                   a: int, b: int, direction: int, eps: float = 0.002) -> float:
    """Compute the gap (positive = satisfied) for a separation constraint.

    Returns how much slack remains. Negative means violated by that amount.
    """
    wa, ha = float(sizes[a, 0]), float(sizes[a, 1])
    wb, hb = float(sizes[b, 0]), float(sizes[b, 1])

    if direction == L:
        return pos[b, 0] - pos[a, 0] - (wa + wb) / 2 - eps
    elif direction == R:
        return pos[a, 0] - pos[b, 0] - (wa + wb) / 2 - eps
    elif direction == B:
        return pos[b, 1] - pos[a, 1] - (ha + hb) / 2 - eps
    elif direction == A:
        return pos[a, 1] - pos[b, 1] - (ha + hb) / 2 - eps
    return 0.0


def push_apart(pos: np.ndarray, sizes: np.ndarray,
               a: int, b: int, direction: int,
               movable_a: bool, movable_b: bool,
               eps: float = 0.002) -> bool:
    """Push two macros apart to satisfy a separation constraint. Modifies pos in-place.

    Returns True if the fix was applied, False if neither macro is movable.
    """
    gap = separation_gap(pos, sizes, a, b, direction, eps)
    if gap >= 0:
        return True  # already satisfied

    fix = -gap
    axis = 0 if direction in (L, R) else 1
    # sign: which direction does a need to move?
    # L: a goes negative, b goes positive on x-axis
    # R: a goes positive, b goes negative on x-axis
    # B: a goes negative, b goes positive on y-axis
    # A: a goes positive, b goes negative on y-axis
    sign_a = -1.0 if direction in (L, B) else 1.0

    if movable_a and movable_b:
        pos[a, axis] += sign_a * fix / 2
        pos[b, axis] -= sign_a * fix / 2
    elif movable_a:
        pos[a, axis] += sign_a * fix
    elif movable_b:
        pos[b, axis] -= sign_a * fix
    else:
        return False

    return True


def clamp_to_canvas(pos: np.ndarray, sizes: np.ndarray,
                    indices, canvas_w: float, canvas_h: float):
    """Clamp macro centers so they stay fully within the canvas. Modifies pos in-place."""
    for idx in indices:
        hw = sizes[idx, 0] / 2
        hh = sizes[idx, 1] / 2
        pos[idx, 0] = np.clip(pos[idx, 0], hw, canvas_w - hw)
        pos[idx, 1] = np.clip(pos[idx, 1], hh, canvas_h - hh)


def project_assignment(pos: np.ndarray, sizes: np.ndarray,
                       assignment: dict, movable_mask: np.ndarray,
                       canvas_w: float, canvas_h: float,
                       n_rounds: int = 3, eps: float = 0.002):
    """Project positions onto the feasible set defined by an assignment.

    Iteratively enforces separation constraints + canvas bounds (Dykstra-like).
    Modifies pos in-place.
    """
    for _ in range(n_rounds):
        for (a, b), direction in assignment.items():
            if not (movable_mask[a] or movable_mask[b]):
                continue
            push_apart(pos, sizes, a, b, direction,
                       bool(movable_mask[a]), bool(movable_mask[b]), eps)

        # Clamp all movable macros
        movable_indices = np.where(movable_mask)[0]
        clamp_to_canvas(pos, sizes, movable_indices, canvas_w, canvas_h)


def cascade_repair(pos: np.ndarray, sizes: np.ndarray,
                   assignment: dict, macro_to_pairs: dict,
                   moved: set, fixed_mask: np.ndarray, n_hard: int,
                   canvas_w: float, canvas_h: float,
                   max_rounds: int = 10, eps: float = 0.002):
    """Fix constraint violations caused by moving macros, cascading to neighbors.

    Modifies pos and moved in-place. After this, also call overlap_repair
    if hard overlap freedom is required.
    """
    for _ in range(max_rounds):
        violations = 0
        checked = set()
        for m in list(moved):
            for pair_key in macro_to_pairs.get(m, []):
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                a, b = pair_key
                d = assignment.get(pair_key)
                if d is None:
                    continue

                gap = separation_gap(pos, sizes, a, b, d, eps)
                if gap >= -1e-6:
                    continue

                violations += 1
                mov_a = a < n_hard and not fixed_mask[a]
                mov_b = b < n_hard and not fixed_mask[b]
                push_apart(pos, sizes, a, b, d, mov_a, mov_b, eps)
                moved.add(a)
                moved.add(b)

        clamp_to_canvas(pos, sizes, list(moved), canvas_w, canvas_h)

        if violations == 0:
            break


def overlap_repair(pos: np.ndarray, sizes: np.ndarray,
                   moved: set, fixed_mask: np.ndarray, n_hard: int,
                   canvas_w: float, canvas_h: float,
                   max_rounds: int = 10, eps: float = 0.002):
    """Direct overlap repair for hard macros near the moved set.

    Pushes overlapping pairs apart in the direction of smallest violation.
    Modifies pos and moved in-place.
    """
    for _ in range(max_rounds):
        any_overlap = False
        for a in list(moved):
            for b in range(n_hard):
                if b == a:
                    continue
                wa, ha = float(sizes[a, 0]), float(sizes[a, 1])
                wb, hb = float(sizes[b, 0]), float(sizes[b, 1])
                dx = abs(pos[a, 0] - pos[b, 0])
                dy = abs(pos[a, 1] - pos[b, 1])
                min_dx = (wa + wb) / 2 + eps
                min_dy = (ha + hb) / 2 + eps

                if dx < min_dx and dy < min_dy:
                    mov_a = a < n_hard and not fixed_mask[a]
                    mov_b = b < n_hard and not fixed_mask[b]
                    viol_x = min_dx - dx
                    viol_y = min_dy - dy

                    if viol_x < viol_y:
                        sign = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                        if mov_a and mov_b:
                            pos[a, 0] -= sign * viol_x / 2
                            pos[b, 0] += sign * viol_x / 2
                        elif mov_a:
                            pos[a, 0] -= sign * viol_x
                        elif mov_b:
                            pos[b, 0] += sign * viol_x
                    else:
                        sign = 1.0 if pos[a, 1] < pos[b, 1] else -1.0
                        if mov_a and mov_b:
                            pos[a, 1] -= sign * viol_y / 2
                            pos[b, 1] += sign * viol_y / 2
                        elif mov_a:
                            pos[a, 1] -= sign * viol_y
                        elif mov_b:
                            pos[b, 1] += sign * viol_y

                    any_overlap = True
                    moved.add(a)
                    moved.add(b)

        if not any_overlap:
            break

        clamp_to_canvas(pos, sizes, list(moved), canvas_w, canvas_h)


def legalize(pos: np.ndarray, sizes: np.ndarray,
             n_hard: int, movable_mask: np.ndarray,
             canvas_w: float, canvas_h: float,
             max_rounds: int = 15) -> np.ndarray:
    """Iterative overlap resolution for a placement. Returns legalized positions.

    Used after perturbation to resolve any overlaps introduced.
    """
    pos = pos.copy()
    half_w = sizes[:n_hard, 0] / 2
    half_h = sizes[:n_hard, 1] / 2

    for _ in range(max_rounds):
        dx_mat = np.abs(pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T)
        dy_mat = np.abs(pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T)
        min_dx_mat = half_w[:, None] + half_w[None, :]
        min_dy_mat = half_h[:, None] + half_h[None, :]
        overlap = (dx_mat < min_dx_mat) & (dy_mat < min_dy_mat)
        np.fill_diagonal(overlap, False)

        pairs = np.argwhere(np.triu(overlap))
        if len(pairs) == 0:
            break

        for a, b in pairs:
            if not (movable_mask[a] or movable_mask[b]):
                continue
            viol_x = min_dx_mat[a, b] - dx_mat[a, b]
            viol_y = min_dy_mat[a, b] - dy_mat[a, b]
            if viol_x < viol_y:
                sign = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                if movable_mask[a] and movable_mask[b]:
                    pos[a, 0] -= sign * viol_x / 2
                    pos[b, 0] += sign * viol_x / 2
                elif movable_mask[a]:
                    pos[a, 0] -= sign * viol_x
                else:
                    pos[b, 0] += sign * viol_x
            else:
                sign = 1.0 if pos[a, 1] < pos[b, 1] else -1.0
                if movable_mask[a] and movable_mask[b]:
                    pos[a, 1] -= sign * viol_y / 2
                    pos[b, 1] += sign * viol_y / 2
                elif movable_mask[a]:
                    pos[a, 1] -= sign * viol_y
                else:
                    pos[b, 1] += sign * viol_y

        for i in range(n_hard):
            if movable_mask[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i], canvas_w - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1], half_h[i], canvas_h - half_h[i])

    return pos


def refine_density(positions: np.ndarray, sizes: np.ndarray,
                   assignment: dict, n_macros: int, n_hard: int,
                   movable_mask: np.ndarray,
                   canvas_w: float, canvas_h: float,
                   n_steps: int = 100, lr: float = 0.1,
                   grid_rows: int = 32, grid_cols: int = 32) -> np.ndarray:
    """
    Projected gradient descent to reduce density while staying in the polyhedron.

    Pushes macros away from dense grid cells, then projects back onto
    separation constraints + canvas bounds.
    """
    pos = positions.copy()

    hard_mask = np.zeros(n_macros, dtype=bool)
    hard_mask[:n_hard] = True

    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    for step in range(n_steps):
        grad = np.zeros_like(pos)
        density_grid = np.zeros((grid_rows, grid_cols))
        macro_cells = {}

        for i in range(n_macros):
            if not hard_mask[i]:
                continue
            x, y = pos[i]
            w, h = sizes[i]

            x_lo = max(0, x - w / 2)
            x_hi = min(canvas_w, x + w / 2)
            y_lo = max(0, y - h / 2)
            y_hi = min(canvas_h, y + h / 2)

            c_lo = max(0, int(x_lo / cell_w))
            c_hi = min(grid_cols - 1, int(x_hi / cell_w))
            r_lo = max(0, int(y_lo / cell_h))
            r_hi = min(grid_rows - 1, int(y_hi / cell_h))

            cells = []
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    ox = max(0, min(x_hi, (c + 1) * cell_w) - max(x_lo, c * cell_w))
                    oy = max(0, min(y_hi, (r + 1) * cell_h) - max(y_lo, r * cell_h))
                    area = ox * oy
                    if area > 0:
                        density_grid[r, c] += area / (cell_w * cell_h)
                        cells.append((r, c, area))
            macro_cells[i] = cells

        threshold = np.percentile(density_grid, 90)

        for i in range(n_macros):
            if not movable_mask[i]:
                continue
            if i not in macro_cells:
                continue
            for r, c, area in macro_cells[i]:
                if density_grid[r, c] > threshold:
                    excess = density_grid[r, c] - threshold
                    cell_cx = (c + 0.5) * cell_w
                    cell_cy = (r + 0.5) * cell_h
                    dx = pos[i, 0] - cell_cx
                    dy = pos[i, 1] - cell_cy
                    dist = max(np.sqrt(dx * dx + dy * dy), 1e-6)
                    grad[i, 0] += excess * dx / dist
                    grad[i, 1] += excess * dy / dist

        for i in range(n_macros):
            if movable_mask[i]:
                pos[i] += lr * grad[i]

        # Project onto feasible set
        clamp_to_canvas(pos, sizes, np.where(movable_mask)[0], canvas_w, canvas_h)
        project_assignment(pos, sizes, assignment, movable_mask, canvas_w, canvas_h)

    return pos
