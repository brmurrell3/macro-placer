"""
Polyhedra Navigation Placer

Decomposes macro placement into:
1. Discrete topology: which polyhedron (L/R/A/B assignment for each macro pair)
2. Continuous optimization: LP solve within the chosen polyhedron

The LP gives exact optimal HPWL positions + dual variables that guide navigation
between polyhedra. Always stays in feasible (non-overlapping) space.

Modules:
  1. AssignmentExtractor: legal placement -> pairwise L/R/A/B assignment
  2. LPSolver: assignment -> optimal positions + duals (via HiGHS)
  3. NeighborGenerator: duals -> ranked candidate flips
  4. Navigator: search loop (propose, evaluate, accept/reject)
"""

import time
import numpy as np
import torch
import highspy

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Module 1: Assignment Extractor
# ---------------------------------------------------------------------------

# Directions: for pair (i, k), the assignment says how i relates to k
L = 0  # i is left of k
R = 1  # i is right of k (k is left of i)
B = 2  # i is below k
A = 3  # i is above k (k is below i)

DIR_NAMES = ["L", "R", "B", "A"]


def extract_assignment(positions: np.ndarray, sizes: np.ndarray,
                       hard_indices: np.ndarray) -> dict:
    """
    Extract pairwise L/R/A/B assignment from a legal (non-overlapping) placement.

    For each hard macro pair (i, k) where i < k, determines which separation
    constraint is active (tightest). If macros overlap, raises an error.

    Args:
        positions: [N, 2] center coordinates (x, y) for all macros
        sizes: [N, 2] (width, height) for all macros
        hard_indices: indices of hard macros to consider

    Returns:
        dict mapping (i, k) -> direction in {L, R, B, A}
    """
    n = len(hard_indices)
    pos = positions[hard_indices]  # [n, 2]
    sz = sizes[hard_indices]  # [n, 2]

    assignment = {}

    for a in range(n):
        i = hard_indices[a]
        xi, yi = pos[a]
        wi, hi = sz[a]

        for b in range(a + 1, n):
            k = hard_indices[b]
            xk, yk = pos[b]
            wk, hk = sz[b]

            # Compute separation gaps in each direction
            # "i left of k" means xi + wi/2 <= xk - wk/2
            gap_l = (xk - wk / 2) - (xi + wi / 2)  # i left of k
            gap_r = (xi - wi / 2) - (xk + wk / 2)  # i right of k
            gap_b = (yk - hk / 2) - (yi + hi / 2)  # i below k
            gap_a = (yi - hi / 2) - (yk + hk / 2)  # i above k

            gaps = [gap_l, gap_r, gap_b, gap_a]

            # At least one gap must be >= 0 for non-overlap
            # Pick the direction with largest gap (most natural separation)
            best_dir = int(np.argmax(gaps))

            if gaps[best_dir] < -1e-3:
                # Overlapping pair - shouldn't happen with legal placement
                # Fall back to direction with least violation
                pass

            assignment[(i, k)] = best_dir

    return assignment


def extract_assignment_vectorized(positions: np.ndarray, sizes: np.ndarray,
                                  hard_indices: np.ndarray) -> dict:
    """Vectorized version of extract_assignment for speed."""
    n = len(hard_indices)
    pos = positions[hard_indices]  # [n, 2]
    sz = sizes[hard_indices]  # [n, 2]

    # Compute all pairwise gaps at once
    # For pair (a, b): gap_l = (xb - wb/2) - (xa + wa/2)
    x = pos[:, 0]  # [n]
    y = pos[:, 1]  # [n]
    w = sz[:, 0]  # [n]
    h = sz[:, 1]  # [n]

    # half-widths and half-heights
    hw = w / 2
    hh = h / 2

    # Right edges and left edges
    right_x = x + hw  # [n]
    left_x = x - hw  # [n]
    top_y = y + hh  # [n]
    bottom_y = y - hh  # [n]

    assignment = {}

    # Build upper triangle indices
    ai, bi = np.triu_indices(n, k=1)

    # gap_l[a,b] = left_x[b] - right_x[a]  (a left of b)
    gap_l = left_x[bi] - right_x[ai]
    # gap_r[a,b] = left_x[a] - right_x[b]  (a right of b)
    gap_r = left_x[ai] - right_x[bi]
    # gap_b[a,b] = bottom_y[b] - top_y[a]  (a below b)
    gap_b = bottom_y[bi] - top_y[ai]
    # gap_a[a,b] = bottom_y[a] - top_y[b]  (a above b)
    gap_a = bottom_y[ai] - top_y[bi]

    gaps = np.stack([gap_l, gap_r, gap_b, gap_a], axis=1)  # [n_pairs, 4]
    best_dirs = np.argmax(gaps, axis=1)

    for idx in range(len(ai)):
        i = int(hard_indices[ai[idx]])
        k = int(hard_indices[bi[idx]])
        assignment[(i, k)] = int(best_dirs[idx])

    return assignment


# ---------------------------------------------------------------------------
# Module 2: LP Solver
# ---------------------------------------------------------------------------

class LPSolver:
    """
    Solves the HPWL LP within a given polyhedron (assignment).

    LP formulation:
        min sum_j (u_j_max - u_j_min + v_j_max - v_j_min)  [HPWL]

        s.t. for each net j, for each macro i in net j:
            u_j_max >= x_i    (bounding box max x)
            u_j_min <= x_i    (bounding box min x)
            v_j_max >= y_i    (bounding box max y)
            v_j_min <= y_i    (bounding box min y)

        For each pair (i,k) with assignment sigma:
            separation constraint (one of L/R/B/A)

        Canvas bounds for all macros.
        Fixed macros pinned.

    Returns optimal positions and dual variables for separation constraints.
    """

    def __init__(self, benchmark: Benchmark, plc, net_macros: list = None):
        """
        Args:
            benchmark: Benchmark object
            plc: PlacementCost object (for net connectivity)
            net_macros: precomputed list of (net_idx, [macro_indices]) pairs
        """
        self.benchmark = benchmark
        self.plc = plc
        self.n_hard = benchmark.num_hard_macros
        self.n_total = benchmark.num_macros
        self.canvas_w = benchmark.canvas_width
        self.canvas_h = benchmark.canvas_height
        self.sizes = benchmark.macro_sizes.numpy()
        self.fixed = benchmark.macro_fixed.numpy()
        self.fixed_pos = benchmark.macro_positions.numpy()

        # Build net connectivity: list of nets, each net is list of macro indices
        if net_macros is not None:
            self.net_macros = net_macros
        else:
            self.net_macros = self._build_net_macros()

        self._last_h = None  # cached HiGHS model for warm-starting

    def _build_net_macros(self) -> list:
        """Extract net-to-macro mapping from plc."""
        b = self.benchmark
        # Build name -> tensor index
        name_to_idx = {}
        for i, name in enumerate(b.macro_names):
            name_to_idx[name] = i

        net_macros = []
        for net_name, pin_names in self.plc.nets.items():
            macros_in_net = set()
            for pn in pin_names:
                if "/" in pn:
                    macro_name = pn.split("/")[0]
                else:
                    macro_name = pn
                if macro_name in name_to_idx:
                    macros_in_net.add(name_to_idx[macro_name])
            if len(macros_in_net) >= 2:
                net_macros.append(sorted(macros_in_net))
        return net_macros

    def solve(self, assignment: dict, warm_start: bool = False,
              time_limit: float = 60.0) -> dict:
        """
        Solve the HPWL LP for the given assignment.

        Args:
            assignment: dict (i, k) -> direction for all hard macro pairs
            warm_start: if True, modify existing model rather than rebuilding
            time_limit: solver time limit in seconds

        Returns:
            dict with:
                positions: [n_total, 2] numpy array of optimal positions
                hpwl: optimal HPWL value
                duals: dict (i,k) -> dual value for separation constraint
                status: solver status string
                solve_time: time in seconds
        """
        t0 = time.time()

        n = self.n_total
        n_nets = len(self.net_macros)
        inf = highspy.kHighsInf

        # Variables layout:
        # x_0..x_{n-1}: x positions of all macros
        # y_0..y_{n-1}: y positions of all macros
        # For each net j: u_max_j, u_min_j, v_max_j, v_min_j
        n_vars = 2 * n + 4 * n_nets
        x_start = 0
        y_start = n
        net_var_start = 2 * n

        # Column costs
        col_cost = [0.0] * n_vars
        for j in range(n_nets):
            base = net_var_start + 4 * j
            col_cost[base + 0] = 1.0   # u_max
            col_cost[base + 1] = -1.0  # u_min
            col_cost[base + 2] = 1.0   # v_max
            col_cost[base + 3] = -1.0  # v_min

        # Column bounds
        col_lower = [-inf] * n_vars
        col_upper = [inf] * n_vars

        for i in range(n):
            hw = self.sizes[i, 0] / 2
            hh = self.sizes[i, 1] / 2
            col_lower[x_start + i] = float(hw)
            col_upper[x_start + i] = float(self.canvas_w - hw)
            col_lower[y_start + i] = float(hh)
            col_upper[y_start + i] = float(self.canvas_h - hh)

        for i in range(n):
            if self.fixed[i]:
                col_lower[x_start + i] = float(self.fixed_pos[i, 0])
                col_upper[x_start + i] = float(self.fixed_pos[i, 0])
                col_lower[y_start + i] = float(self.fixed_pos[i, 1])
                col_upper[y_start + i] = float(self.fixed_pos[i, 1])

        # Build sparse constraint matrix (row-wise)
        row_starts = [0]
        col_indices = []
        values = []
        row_lower = []
        row_upper = []

        # 1. Bounding box constraints for nets
        for j, macros in enumerate(self.net_macros):
            base = net_var_start + 4 * j
            for i in macros:
                # u_max_j >= x_i
                col_indices.extend([base + 0, x_start + i])
                values.extend([1.0, -1.0])
                row_starts.append(len(col_indices))
                row_lower.append(0.0)
                row_upper.append(inf)

                # u_min_j <= x_i  =>  x_i - u_min_j >= 0
                col_indices.extend([base + 1, x_start + i])
                values.extend([-1.0, 1.0])
                row_starts.append(len(col_indices))
                row_lower.append(0.0)
                row_upper.append(inf)

                # v_max_j >= y_i
                col_indices.extend([base + 2, y_start + i])
                values.extend([1.0, -1.0])
                row_starts.append(len(col_indices))
                row_lower.append(0.0)
                row_upper.append(inf)

                # v_min_j <= y_i
                col_indices.extend([base + 3, y_start + i])
                values.extend([-1.0, 1.0])
                row_starts.append(len(col_indices))
                row_lower.append(0.0)
                row_upper.append(inf)

        # 2. Separation constraints from assignment
        # Add small epsilon to ensure strict non-overlap after LP solve
        eps = 0.002
        sep_constraint_start = len(row_lower)
        sep_constraint_map = {}

        for (i, k), direction in assignment.items():
            wi = float(self.sizes[i, 0])
            hi_h = float(self.sizes[i, 1])
            wk = float(self.sizes[k, 0])
            hk = float(self.sizes[k, 1])

            if direction == L:
                col_indices.extend([x_start + k, x_start + i])
                values.extend([1.0, -1.0])
                row_lower.append((wi + wk) / 2 + eps)
            elif direction == R:
                col_indices.extend([x_start + i, x_start + k])
                values.extend([1.0, -1.0])
                row_lower.append((wi + wk) / 2 + eps)
            elif direction == B:
                col_indices.extend([y_start + k, y_start + i])
                values.extend([1.0, -1.0])
                row_lower.append((hi_h + hk) / 2 + eps)
            elif direction == A:
                col_indices.extend([y_start + i, y_start + k])
                values.extend([1.0, -1.0])
                row_lower.append((hi_h + hk) / 2 + eps)

            row_starts.append(len(col_indices))
            row_upper.append(inf)
            sep_constraint_map[(i, k)] = len(row_lower) - 1

        n_rows = len(row_lower)

        # Build HighsLp model
        lp = highspy.HighsLp()
        lp.num_col_ = n_vars
        lp.num_row_ = n_rows
        lp.col_cost_ = col_cost
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper

        lp.a_matrix_ = highspy.HighsSparseMatrix()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.num_col_ = n_vars
        lp.a_matrix_.start_ = row_starts
        lp.a_matrix_.index_ = col_indices
        lp.a_matrix_.value_ = values

        h = highspy.Highs()
        h.silent()
        h.setOptionValue("time_limit", time_limit)
        h.passModel(lp)
        h.run()

        status = h.getInfoValue("primal_solution_status")[1]
        solve_time = time.time() - t0

        if status != 2:  # 2 = feasible
            return {
                "positions": None,
                "hpwl": float("inf"),
                "duals": {},
                "status": f"infeasible (status={status})",
                "solve_time": solve_time,
            }

        # Extract solution
        sol = h.getSolution()
        col_vals = list(sol.col_value)
        x_pos = np.array(col_vals[x_start:x_start + n])
        y_pos = np.array(col_vals[y_start:y_start + n])
        positions = np.stack([x_pos, y_pos], axis=1)

        hpwl = h.getInfoValue("objective_function_value")[1]

        # Extract dual variables for separation constraints
        row_duals = list(sol.row_dual)
        duals = {}
        if len(row_duals) == n_rows:
            for (i, k), row_idx in sep_constraint_map.items():
                duals[(i, k)] = row_duals[row_idx]
        else:
            for (i, k) in sep_constraint_map:
                duals[(i, k)] = 0.0

        self._last_h = h

        return {
            "positions": positions,
            "hpwl": hpwl,
            "duals": duals,
            "status": "optimal",
            "solve_time": solve_time,
            "sep_constraint_map": sep_constraint_map,
        }

    def solve_min_displacement(self, assignment: dict, ref_positions: np.ndarray,
                                time_limit: float = 30.0) -> dict:
        """
        Solve min-displacement LP: find closest feasible positions to ref_positions
        under the given topology.

        Minimizes sum_i |x_i - x_ref_i| + |y_i - y_ref_i| (L1 displacement)
        using standard LP reformulation with slack variables.

        This preserves density/congestion structure while satisfying the new topology.
        """
        t0 = time.time()

        n = self.n_total
        inf = highspy.kHighsInf

        # Variables: x_i, y_i for each macro, plus dx+_i, dx-_i, dy+_i, dy-_i
        # x_i = x_ref_i + dx+_i - dx-_i
        # min sum (dx+_i + dx-_i + dy+_i + dy-_i) for movable hard macros
        n_vars = 2 * n + 4 * n  # positions + slack pairs
        x_start = 0
        y_start = n
        slack_start = 2 * n  # dx+, dx-, dy+, dy- per macro

        col_cost = [0.0] * n_vars
        col_lower = [-inf] * n_vars
        col_upper = [inf] * n_vars

        movable_hard = set()
        for i in range(self.n_hard):
            if not self.fixed[i]:
                movable_hard.add(i)

        # Position bounds
        for i in range(n):
            hw = self.sizes[i, 0] / 2
            hh = self.sizes[i, 1] / 2
            col_lower[x_start + i] = float(hw)
            col_upper[x_start + i] = float(self.canvas_w - hw)
            col_lower[y_start + i] = float(hh)
            col_upper[y_start + i] = float(self.canvas_h - hh)

        # Fix non-movable macros and soft macros
        for i in range(n):
            if self.fixed[i] or i >= self.n_hard:
                col_lower[x_start + i] = float(ref_positions[i, 0])
                col_upper[x_start + i] = float(ref_positions[i, 0])
                col_lower[y_start + i] = float(ref_positions[i, 1])
                col_upper[y_start + i] = float(ref_positions[i, 1])

        # Slack variables: non-negative, cost 1.0 for movable hard macros
        for i in range(n):
            base = slack_start + 4 * i
            for j in range(4):
                col_lower[base + j] = 0.0
            if i in movable_hard:
                col_cost[base + 0] = 1.0  # dx+
                col_cost[base + 1] = 1.0  # dx-
                col_cost[base + 2] = 1.0  # dy+
                col_cost[base + 3] = 1.0  # dy-

        # Constraints
        row_starts = [0]
        col_indices = []
        values = []
        row_lower = []
        row_upper = []

        # 1. Linking constraints: x_i - dx+_i + dx-_i = x_ref_i
        for i in movable_hard:
            base = slack_start + 4 * i
            # x_i - dx+_i + dx-_i = x_ref_i
            col_indices.extend([x_start + i, base + 0, base + 1])
            values.extend([1.0, -1.0, 1.0])
            row_starts.append(len(col_indices))
            row_lower.append(float(ref_positions[i, 0]))
            row_upper.append(float(ref_positions[i, 0]))

            # y_i - dy+_i + dy-_i = y_ref_i
            col_indices.extend([y_start + i, base + 2, base + 3])
            values.extend([1.0, -1.0, 1.0])
            row_starts.append(len(col_indices))
            row_lower.append(float(ref_positions[i, 1]))
            row_upper.append(float(ref_positions[i, 1]))

        # 2. Separation constraints — only include "nearby" pairs
        # Pairs far apart are trivially satisfied; including all 30K+ is too slow
        eps = 0.002
        margin = 5.0  # only constrain pairs within this margin of their min separation
        sep_constraint_map = {}
        for (i, k), direction in assignment.items():
            wi = float(self.sizes[i, 0])
            hi_h = float(self.sizes[i, 1])
            wk = float(self.sizes[k, 0])
            hk = float(self.sizes[k, 1])

            # Check if this constraint is "tight" at reference positions
            if direction == L or direction == R:
                min_sep = (wi + wk) / 2
                actual_sep = abs(ref_positions[i, 0] - ref_positions[k, 0])
                if actual_sep > min_sep + margin:
                    continue
            else:
                min_sep = (hi_h + hk) / 2
                actual_sep = abs(ref_positions[i, 1] - ref_positions[k, 1])
                if actual_sep > min_sep + margin:
                    continue

            if direction == L:
                col_indices.extend([x_start + k, x_start + i])
                values.extend([1.0, -1.0])
                row_lower.append((wi + wk) / 2 + eps)
            elif direction == R:
                col_indices.extend([x_start + i, x_start + k])
                values.extend([1.0, -1.0])
                row_lower.append((wi + wk) / 2 + eps)
            elif direction == B:
                col_indices.extend([y_start + k, y_start + i])
                values.extend([1.0, -1.0])
                row_lower.append((hi_h + hk) / 2 + eps)
            elif direction == A:
                col_indices.extend([y_start + i, y_start + k])
                values.extend([1.0, -1.0])
                row_lower.append((hi_h + hk) / 2 + eps)

            row_starts.append(len(col_indices))
            row_upper.append(inf)
            sep_constraint_map[(i, k)] = len(row_lower) - 1

        n_rows = len(row_lower)

        lp = highspy.HighsLp()
        lp.num_col_ = n_vars
        lp.num_row_ = n_rows
        lp.col_cost_ = col_cost
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper

        lp.a_matrix_ = highspy.HighsSparseMatrix()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.num_col_ = n_vars
        lp.a_matrix_.start_ = row_starts
        lp.a_matrix_.index_ = col_indices
        lp.a_matrix_.value_ = values

        h = highspy.Highs()
        h.silent()
        h.setOptionValue("time_limit", time_limit)
        h.passModel(lp)
        h.run()

        status = h.getInfoValue("primal_solution_status")[1]
        solve_time = time.time() - t0

        if status != 2:
            return {
                "positions": None,
                "displacement": float("inf"),
                "status": f"infeasible (status={status})",
                "solve_time": solve_time,
            }

        sol = h.getSolution()
        col_vals = list(sol.col_value)
        x_pos = np.array(col_vals[x_start:x_start + n])
        y_pos = np.array(col_vals[y_start:y_start + n])
        positions = np.stack([x_pos, y_pos], axis=1)

        displacement = h.getInfoValue("objective_function_value")[1]

        return {
            "positions": positions,
            "displacement": displacement,
            "status": "optimal",
            "solve_time": solve_time,
        }


# ---------------------------------------------------------------------------
# Module 3: Neighbor Generator
# ---------------------------------------------------------------------------

class NeighborGenerator:
    """
    Proposes candidate assignment flips ranked by dual variable magnitude.

    P(flip pair (i,k)) proportional to |d_{ik}|^alpha.
    Large dual = expensive constraint = most promising to flip.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def rank_candidates(self, duals: dict, assignment: dict,
                        top_k: int = 50) -> list:
        """
        Rank pairs by dual magnitude and suggest flip directions.

        Returns list of (pair, new_direction, dual_magnitude) tuples,
        sorted by dual magnitude descending.
        """
        candidates = []
        for (i, k), dual_val in duals.items():
            mag = abs(dual_val)
            if mag < 1e-10:
                continue
            current_dir = assignment[(i, k)]
            # Try all other directions
            for new_dir in range(4):
                if new_dir != current_dir:
                    candidates.append(((i, k), new_dir, mag))

        # Sort by dual magnitude descending
        candidates.sort(key=lambda x: -x[2])
        return candidates[:top_k * 3]  # 3 alternatives per pair

    def sample_candidates(self, duals: dict, assignment: dict,
                          n_samples: int = 20, rng: np.random.Generator = None) -> list:
        """
        Sample candidate flips weighted by |dual|^alpha.

        Returns list of (pair, new_direction) tuples.
        """
        if rng is None:
            rng = np.random.default_rng()

        pairs = list(duals.keys())
        magnitudes = np.array([abs(duals[p]) for p in pairs])

        if magnitudes.sum() < 1e-15:
            return []

        weights = magnitudes ** self.alpha
        weights /= weights.sum()

        # Sample pairs
        chosen = rng.choice(len(pairs), size=min(n_samples, len(pairs)),
                            replace=False, p=weights)

        candidates = []
        for idx in chosen:
            pair = pairs[idx]
            current_dir = assignment[pair]
            # Pick a random alternative direction
            alternatives = [d for d in range(4) if d != current_dir]
            new_dir = rng.choice(alternatives)
            candidates.append((pair, new_dir))

        return candidates


# ---------------------------------------------------------------------------
# Module 4: Navigator
# ---------------------------------------------------------------------------

class Navigator:
    """
    Search loop: propose flips, evaluate via proxy cost, accept/reject.

    Uses min-displacement LP to find nearest feasible positions after a flip,
    then evaluates full proxy cost to decide acceptance.

    Dual variables from HPWL LP guide which pairs to flip.
    """

    def __init__(self, lp_solver: LPSolver, neighbor_gen: NeighborGenerator,
                 benchmark: Benchmark, plc):
        self.lp = lp_solver
        self.ng = neighbor_gen
        self.benchmark = benchmark
        self.plc = plc
        self.rng = np.random.default_rng(42)

    def _eval_proxy(self, positions: np.ndarray) -> dict:
        """Evaluate full proxy cost."""
        from macro_place.objective import compute_proxy_cost
        placement = torch.tensor(positions, dtype=torch.float32)
        return compute_proxy_cost(placement, self.benchmark, self.plc)

    def _project_flip(self, positions: np.ndarray, pair: tuple,
                       new_dir: int, assignment: dict) -> np.ndarray:
        """
        Apply a single pair flip via direct constraint projection.

        Push the two macros apart minimally to satisfy the new separation direction.
        Then cascade: fix any newly violated constraints with neighbors.
        Returns new positions or None if infeasible (out of canvas).
        """
        i, k = pair
        pos = positions.copy()
        sizes = self.benchmark.macro_sizes.numpy()
        movable_i = i < self.benchmark.num_hard_macros and not self.benchmark.macro_fixed[i]
        movable_k = k < self.benchmark.num_hard_macros and not self.benchmark.macro_fixed[k]
        cw = self.benchmark.canvas_width
        ch = self.benchmark.canvas_height

        # Apply the new separation constraint
        wi, hi = float(sizes[i, 0]), float(sizes[i, 1])
        wk, hk = float(sizes[k, 0]), float(sizes[k, 1])
        eps = 0.002

        if new_dir == L:  # i left of k
            min_sep = (wi + wk) / 2 + eps
            gap = pos[k, 0] - pos[i, 0]
            if gap < min_sep:
                fix = min_sep - gap
                if movable_i and movable_k:
                    pos[i, 0] -= fix / 2
                    pos[k, 0] += fix / 2
                elif movable_i:
                    pos[i, 0] -= fix
                elif movable_k:
                    pos[k, 0] += fix
                else:
                    return None
        elif new_dir == R:  # i right of k
            min_sep = (wi + wk) / 2 + eps
            gap = pos[i, 0] - pos[k, 0]
            if gap < min_sep:
                fix = min_sep - gap
                if movable_i and movable_k:
                    pos[i, 0] += fix / 2
                    pos[k, 0] -= fix / 2
                elif movable_i:
                    pos[i, 0] += fix
                elif movable_k:
                    pos[k, 0] -= fix
                else:
                    return None
        elif new_dir == B:  # i below k
            min_sep = (hi + hk) / 2 + eps
            gap = pos[k, 1] - pos[i, 1]
            if gap < min_sep:
                fix = min_sep - gap
                if movable_i and movable_k:
                    pos[i, 1] -= fix / 2
                    pos[k, 1] += fix / 2
                elif movable_i:
                    pos[i, 1] -= fix
                elif movable_k:
                    pos[k, 1] += fix
                else:
                    return None
        elif new_dir == A:  # i above k
            min_sep = (hi + hk) / 2 + eps
            gap = pos[i, 1] - pos[k, 1]
            if gap < min_sep:
                fix = min_sep - gap
                if movable_i and movable_k:
                    pos[i, 1] += fix / 2
                    pos[k, 1] -= fix / 2
                elif movable_i:
                    pos[i, 1] += fix
                elif movable_k:
                    pos[k, 1] -= fix
                else:
                    return None

        # Clamp to canvas
        for idx in [i, k]:
            hw = sizes[idx, 0] / 2
            hh = sizes[idx, 1] / 2
            pos[idx, 0] = np.clip(pos[idx, 0], hw, cw - hw)
            pos[idx, 1] = np.clip(pos[idx, 1], hh, ch - hh)

        # Cascade: fix violations created by moving i and k
        # Check all pairs involving i or k
        moved = {i, k}
        n_hard = self.benchmark.num_hard_macros
        for cascade_round in range(3):
            violations = 0
            for (a, b), d in assignment.items():
                if a not in moved and b not in moved:
                    continue
                wa, ha = float(sizes[a, 0]), float(sizes[a, 1])
                wb, hb = float(sizes[b, 0]), float(sizes[b, 1])
                mov_a = a < n_hard and not self.benchmark.macro_fixed[a]
                mov_b = b < n_hard and not self.benchmark.macro_fixed[b]

                if d == L:
                    gap = pos[b, 0] - pos[a, 0] - (wa + wb) / 2 - eps
                elif d == R:
                    gap = pos[a, 0] - pos[b, 0] - (wa + wb) / 2 - eps
                elif d == B:
                    gap = pos[b, 1] - pos[a, 1] - (ha + hb) / 2 - eps
                elif d == A:
                    gap = pos[a, 1] - pos[b, 1] - (ha + hb) / 2 - eps
                else:
                    continue

                if gap < -0.001:
                    violations += 1
                    fix = -gap
                    if d in (L, R):
                        if mov_a and mov_b:
                            shift = fix / 2
                            if d == L:
                                pos[a, 0] -= shift
                                pos[b, 0] += shift
                            else:
                                pos[a, 0] += shift
                                pos[b, 0] -= shift
                        elif mov_a:
                            if d == L:
                                pos[a, 0] -= fix
                            else:
                                pos[a, 0] += fix
                        elif mov_b:
                            if d == L:
                                pos[b, 0] += fix
                            else:
                                pos[b, 0] -= fix
                    else:
                        if mov_a and mov_b:
                            shift = fix / 2
                            if d == B:
                                pos[a, 1] -= shift
                                pos[b, 1] += shift
                            else:
                                pos[a, 1] += shift
                                pos[b, 1] -= shift
                        elif mov_a:
                            if d == B:
                                pos[a, 1] -= fix
                            else:
                                pos[a, 1] += fix
                        elif mov_b:
                            if d == B:
                                pos[b, 1] += fix
                            else:
                                pos[b, 1] -= fix

                    moved.add(a)
                    moved.add(b)

            # Re-clamp
            for idx in moved:
                hw = sizes[idx, 0] / 2
                hh = sizes[idx, 1] / 2
                pos[idx, 0] = np.clip(pos[idx, 0], hw, cw - hw)
                pos[idx, 1] = np.clip(pos[idx, 1], hh, ch - hh)

            if violations == 0:
                break

        return pos

    def greedy_descent(self, assignment: dict, initial_result: dict,
                       ref_positions: np.ndarray,
                       max_iters: int = 500, time_budget: float = 300.0,
                       verbose: bool = True) -> dict:
        """
        Greedy descent on proxy cost using fast constraint projection.

        For each candidate flip:
        1. Project macros to satisfy new constraint (O(1) per flip)
        2. Evaluate full proxy cost
        3. Accept if proxy cost improves
        """
        t0 = time.time()

        best_assignment = dict(assignment)
        best_positions = ref_positions.copy()
        best_proxy = self._eval_proxy(ref_positions)["proxy_cost"]
        current_assignment = dict(assignment)
        current_positions = ref_positions.copy()

        hpwl_result = initial_result

        improvements = 0
        evaluations = 0
        stale_iters = 0

        if verbose:
            print(f"  Starting navigation: proxy={best_proxy:.4f}")

        for iteration in range(max_iters):
            elapsed = time.time() - t0
            if elapsed > time_budget:
                if verbose:
                    print(f"  Time budget exhausted at iter {iteration}")
                break

            if stale_iters > 50:
                if verbose:
                    print(f"  Stale for {stale_iters} iters, stopping")
                break

            # Get candidates from HPWL dual variables
            candidates = self.ng.rank_candidates(
                hpwl_result["duals"], current_assignment, top_k=50
            )

            if not candidates:
                break

            improved_this_iter = False

            for pair, new_dir, dual_mag in candidates:
                if time.time() - t0 > time_budget:
                    break

                old_dir = current_assignment[pair]

                # Fast projection
                new_pos = self._project_flip(
                    current_positions, pair, new_dir, current_assignment
                )
                if new_pos is None:
                    continue

                # Update assignment for cascade check
                current_assignment[pair] = new_dir
                evaluations += 1

                costs = self._eval_proxy(new_pos)

                if costs["overlap_count"] == 0 and costs["proxy_cost"] < best_proxy:
                    improvement = best_proxy - costs["proxy_cost"]
                    best_proxy = costs["proxy_cost"]
                    best_assignment = dict(current_assignment)
                    best_positions = new_pos.copy()
                    current_positions = new_pos.copy()
                    improvements += 1
                    improved_this_iter = True
                    stale_iters = 0

                    if verbose:
                        elapsed = time.time() - t0
                        print(f"  iter {iteration}: proxy={best_proxy:.4f} "
                              f"(delta={-improvement:.4f}, "
                              f"pair={pair}, {DIR_NAMES[old_dir]}->{DIR_NAMES[new_dir]}, "
                              f"{elapsed:.1f}s)")

                    # Refresh duals periodically
                    if improvements % 10 == 0:
                        hpwl_result = self.lp.solve(
                            current_assignment, time_limit=30.0
                        )
                    break
                else:
                    current_assignment[pair] = old_dir

            if not improved_this_iter:
                stale_iters += 1

        if verbose:
            elapsed = time.time() - t0
            init_proxy = self._eval_proxy(ref_positions)["proxy_cost"]
            print(f"  Navigation done: {evaluations} evals, "
                  f"{improvements} improvements, {elapsed:.1f}s")
            print(f"  Proxy: {init_proxy:.4f} -> {best_proxy:.4f} "
                  f"({(best_proxy - init_proxy) / init_proxy * 100:+.2f}%)")

        return {
            "assignment": best_assignment,
            "positions": best_positions,
            "proxy_cost": best_proxy,
            "improvements": improvements,
            "evaluations": evaluations,
        }


# ---------------------------------------------------------------------------
# Density refinement within polyhedron
# ---------------------------------------------------------------------------

def refine_density(positions: np.ndarray, sizes: np.ndarray,
                   assignment: dict, benchmark: Benchmark,
                   canvas_w: float, canvas_h: float,
                   n_steps: int = 100, lr: float = 0.1,
                   grid_rows: int = 32, grid_cols: int = 32) -> np.ndarray:
    """
    Projected gradient descent to reduce density while staying in the polyhedron.

    Adds a density penalty and projects back onto the polyhedron constraints
    (separation + canvas bounds) after each step.
    """
    n = benchmark.num_macros
    pos = positions.copy()

    hard_mask = np.zeros(n, dtype=bool)
    hard_mask[:benchmark.num_hard_macros] = True
    movable = (~benchmark.macro_fixed.numpy()) & hard_mask

    # Precompute grid
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    for step in range(n_steps):
        # Compute density gradient
        grad = np.zeros_like(pos)

        # Simple density: count area in each grid cell, penalize hotspots
        density_grid = np.zeros((grid_rows, grid_cols))
        macro_cells = {}  # macro -> list of (row, col, area_fraction)

        for i in range(n):
            if not hard_mask[i]:
                continue
            x, y = pos[i]
            w, h = sizes[i]

            # Find grid cells this macro overlaps
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
                    # Overlap area between macro and cell
                    ox = max(0, min(x_hi, (c + 1) * cell_w) - max(x_lo, c * cell_w))
                    oy = max(0, min(y_hi, (r + 1) * cell_h) - max(y_lo, r * cell_h))
                    area = ox * oy
                    if area > 0:
                        density_grid[r, c] += area / (cell_w * cell_h)
                        cells.append((r, c, area))
            macro_cells[i] = cells

        # Only penalize top 10% dense cells (matching proxy cost metric)
        threshold = np.percentile(density_grid, 90)

        for i in range(n):
            if not movable[i]:
                continue
            if i not in macro_cells:
                continue
            for r, c, area in macro_cells[i]:
                if density_grid[r, c] > threshold:
                    excess = density_grid[r, c] - threshold
                    # Push macro away from dense cell center
                    cell_cx = (c + 0.5) * cell_w
                    cell_cy = (r + 0.5) * cell_h
                    dx = pos[i, 0] - cell_cx
                    dy = pos[i, 1] - cell_cy
                    dist = max(np.sqrt(dx * dx + dy * dy), 1e-6)
                    grad[i, 0] += excess * dx / dist
                    grad[i, 1] += excess * dy / dist

        # Gradient step
        for i in range(n):
            if movable[i]:
                pos[i] += lr * grad[i]

        # Project: enforce canvas bounds
        for i in range(n):
            if not movable[i]:
                continue
            hw = sizes[i, 0] / 2
            hh = sizes[i, 1] / 2
            pos[i, 0] = np.clip(pos[i, 0], hw, canvas_w - hw)
            pos[i, 1] = np.clip(pos[i, 1], hh, canvas_h - hh)

        # Project: enforce separation constraints
        # Do a few rounds of constraint projection (Dykstra-like)
        for _ in range(3):
            for (i, k), direction in assignment.items():
                if not (movable[i] or movable[k]):
                    continue
                wi = sizes[i, 0]
                hi_h = sizes[i, 1]
                wk = sizes[k, 0]
                hk = sizes[k, 1]

                if direction == L:
                    # x_k - x_i >= (wi + wk) / 2
                    min_sep = (wi + wk) / 2
                    gap = pos[k, 0] - pos[i, 0]
                    if gap < min_sep:
                        fix = (min_sep - gap) / 2
                        if movable[i] and movable[k]:
                            pos[i, 0] -= fix
                            pos[k, 0] += fix
                        elif movable[i]:
                            pos[i, 0] -= 2 * fix
                        else:
                            pos[k, 0] += 2 * fix
                elif direction == R:
                    min_sep = (wi + wk) / 2
                    gap = pos[i, 0] - pos[k, 0]
                    if gap < min_sep:
                        fix = (min_sep - gap) / 2
                        if movable[i] and movable[k]:
                            pos[i, 0] += fix
                            pos[k, 0] -= fix
                        elif movable[i]:
                            pos[i, 0] += 2 * fix
                        else:
                            pos[k, 0] -= 2 * fix
                elif direction == B:
                    min_sep = (hi_h + hk) / 2
                    gap = pos[k, 1] - pos[i, 1]
                    if gap < min_sep:
                        fix = (min_sep - gap) / 2
                        if movable[i] and movable[k]:
                            pos[i, 1] -= fix
                            pos[k, 1] += fix
                        elif movable[i]:
                            pos[i, 1] -= 2 * fix
                        else:
                            pos[k, 1] += 2 * fix
                elif direction == A:
                    min_sep = (hi_h + hk) / 2
                    gap = pos[i, 1] - pos[k, 1]
                    if gap < min_sep:
                        fix = (min_sep - gap) / 2
                        if movable[i] and movable[k]:
                            pos[i, 1] += fix
                            pos[k, 1] -= fix
                        elif movable[i]:
                            pos[i, 1] += 2 * fix
                        else:
                            pos[k, 1] -= 2 * fix

    return pos


# ---------------------------------------------------------------------------
# Main Placer
# ---------------------------------------------------------------------------

class PolyhedraNavigationPlacer:
    """
    Polyhedra Navigation placer.

    1. Start from a good initial placement (greedy legalization)
    2. Extract pairwise L/R/A/B assignment (defines which polyhedron we're in)
    3. Solve LP within that polyhedron (optimal HPWL positions + duals)
    4. Navigate to neighboring polyhedra using dual-guided search
    5. Refine density within the best polyhedron found
    """

    def __init__(self, navigate: bool = True, refine: bool = False,
                 nav_iters: int = 100, nav_time: float = 30.0,
                 density_steps: int = 80, verbose: bool = True):
        self.navigate = navigate
        self.refine = refine
        self.nav_iters = nav_iters
        self.nav_time = nav_time
        self.density_steps = density_steps
        self.verbose = verbose

    def _initial_placement(self, benchmark: Benchmark) -> np.ndarray:
        """Generate initial legal placement.

        Tries to use SDF placer (best known topology) if available,
        falls back to greedy shelf packing.
        """
        try:
            import importlib.util, sys
            sdf_path = str(
                __import__("pathlib").Path(__file__).parent.parent
                / "sdf_density" / "placer.py"
            )
            spec = importlib.util.spec_from_file_location("sdf_placer", sdf_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sdf = mod.SDFPlacer()
            placement = sdf.place(benchmark)
            return placement.numpy()
        except Exception as e:
            import sys
            print(f"  [SDF fallback: {e}]", file=sys.stderr)
            pass

        # Fallback: greedy shelf packing
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        movable_indices.sort(key=lambda i: -sizes[i, 1].item())

        gap = 0.001
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        for idx in movable_indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0

            if cursor_y + h > canvas_h:
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue

            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2

            cursor_x += w + gap
            row_height = max(row_height, h)

        return placement.numpy()

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        t_start = time.time()

        # Load plc if not provided (needed for net connectivity and proxy cost)
        if plc is None:
            from macro_place.loader import load_benchmark_from_dir
            _, plc = load_benchmark_from_dir(
                f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}"
            )

        if self.verbose:
            print(f"\n=== Polyhedra Navigation: {benchmark.name} ===")
            print(f"  {benchmark.num_hard_macros} hard macros, "
                  f"{benchmark.num_nets} nets, "
                  f"canvas {benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}")

        sizes = benchmark.macro_sizes.numpy()
        hard_indices = np.arange(benchmark.num_hard_macros)
        movable_hard = hard_indices[~benchmark.macro_fixed[:benchmark.num_hard_macros].numpy()]

        # Step 1: Initial legal placement
        t0 = time.time()
        init_pos = self._initial_placement(benchmark)
        if self.verbose:
            print(f"  Initial placement: {time.time() - t0:.2f}s")

        # Step 2: Extract assignment
        t0 = time.time()
        assignment = extract_assignment_vectorized(init_pos, sizes, movable_hard)
        n_pairs = len(assignment)
        if self.verbose:
            print(f"  Assignment extracted: {n_pairs} pairs, {time.time() - t0:.2f}s")

        # Step 3: Build LP solver and solve
        t0 = time.time()
        lp = LPSolver(benchmark, plc)
        if self.verbose:
            print(f"  LP solver built ({len(lp.net_macros)} nets with >=2 macros), "
                  f"{time.time() - t0:.2f}s")

        t0 = time.time()
        result = lp.solve(assignment, time_limit=60.0)
        if self.verbose:
            print(f"  LP solve: status={result['status']}, "
                  f"HPWL={result['hpwl']:.2f}, {result['solve_time']:.2f}s")

        if result["positions"] is None:
            if self.verbose:
                print("  LP infeasible! Returning initial placement.")
            return torch.tensor(init_pos, dtype=torch.float32)

        # Use initial positions (not LP positions) — they have much better density
        positions = init_pos

        # Step 4: Navigate (if enabled)
        if self.navigate and n_pairs > 0 and result["positions"] is not None:
            t0 = time.time()
            ng = NeighborGenerator(alpha=1.0)
            nav = Navigator(lp, ng, benchmark, plc)

            nav_result = nav.greedy_descent(
                assignment, result,
                ref_positions=init_pos,
                max_iters=self.nav_iters,
                time_budget=self.nav_time,
                verbose=self.verbose,
            )

            assignment = nav_result["assignment"]
            positions = nav_result["positions"]
            if self.verbose:
                print(f"  Navigation total: {time.time() - t0:.1f}s")

        if self.verbose:
            print(f"  Total time: {time.time() - t_start:.1f}s")

        return torch.tensor(positions, dtype=torch.float32)
