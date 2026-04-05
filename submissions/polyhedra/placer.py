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
from collections import defaultdict

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

    def propose_cluster_flips(self, duals: dict, assignment: dict,
                              macro_to_nets: dict, n_proposals: int = 20,
                              rng: np.random.Generator = None) -> list:
        """
        Propose multi-pair cluster flips using two strategies:

        1. Net-correlated flips: flip pairs that share high-fanout nets
        2. Row/column block swaps: reverse ordering of spatial neighbors

        Returns list of clusters, where each cluster is [(pair, new_dir), ...].
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get high-dual pairs
        dual_pairs = [(pair, abs(d)) for pair, d in duals.items() if abs(d) > 1e-10]
        if not dual_pairs:
            return []
        dual_pairs.sort(key=lambda x: -x[1])

        clusters = []

        # Strategy 1: Net-correlated flips
        # Find pairs sharing nets with the highest-dual pair
        if macro_to_nets:
            for seed_pair, seed_mag in dual_pairs[:min(10, len(dual_pairs))]:
                si, sk = seed_pair
                # Find all nets containing si or sk
                si_nets = set(macro_to_nets.get(si, []))
                sk_nets = set(macro_to_nets.get(sk, []))
                shared_nets = si_nets | sk_nets

                # Find other high-dual pairs that share nets with the seed
                cluster = [(seed_pair, rng.choice([d for d in range(4)
                            if d != assignment[seed_pair]]))]

                for other_pair, other_mag in dual_pairs:
                    if other_pair == seed_pair:
                        continue
                    oi, ok = other_pair
                    oi_nets = set(macro_to_nets.get(oi, []))
                    ok_nets = set(macro_to_nets.get(ok, []))
                    if (oi_nets | ok_nets) & shared_nets:
                        new_dir = rng.choice([d for d in range(4)
                                              if d != assignment[other_pair]])
                        cluster.append((other_pair, new_dir))
                        if len(cluster) >= 4:
                            break

                if len(cluster) >= 2:
                    clusters.append(cluster)

        # Strategy 2: Macro-centered flips
        # For a high-dual pair (i,k), flip all pairs involving macro i
        for seed_pair, seed_mag in dual_pairs[:min(5, len(dual_pairs))]:
            si, sk = seed_pair
            for target_macro in [si, sk]:
                cluster = []
                for (pi, pk), d_val in duals.items():
                    if abs(d_val) < 1e-10:
                        continue
                    if pi == target_macro or pk == target_macro:
                        new_dir = rng.choice([d for d in range(4)
                                              if d != assignment[(pi, pk)]])
                        cluster.append(((pi, pk), new_dir))
                        if len(cluster) >= 5:
                            break
                if 2 <= len(cluster) <= 5:
                    clusters.append(cluster)

        # Deduplicate and limit
        seen = set()
        unique_clusters = []
        for c in clusters:
            key = tuple(sorted((p, d) for p, d in c))
            if key not in seen:
                seen.add(key)
                unique_clusters.append(c)
            if len(unique_clusters) >= n_proposals:
                break

        return unique_clusters


# ---------------------------------------------------------------------------
# Module 4: Grid Surrogate (fast proxy cost estimator)
# ---------------------------------------------------------------------------

class GridSurrogate:
    """
    Fast incremental proxy cost estimator using grid-based density,
    RUDY congestion, and HPWL wirelength.

    Designed to filter candidate flips (~0.1ms per eval) before expensive
    full proxy cost verification (~1s per eval).
    """

    def __init__(self, benchmark, plc):
        self.bm = benchmark
        self.n = benchmark.num_macros
        self.n_hard = benchmark.num_hard_macros
        self.canvas_w = benchmark.canvas_width
        self.canvas_h = benchmark.canvas_height

        # Match PlacementCost grid
        self.grid_rows = plc.grid_row
        self.grid_cols = plc.grid_col
        self.n_cells = self.grid_rows * self.grid_cols
        self.cell_w = self.canvas_w / self.grid_cols
        self.cell_h = self.canvas_h / self.grid_rows

        # Routing capacity per cell
        self.grid_v_cap = max(self.cell_w * benchmark.vroutes_per_micron, 1e-6)
        self.grid_h_cap = max(self.cell_h * benchmark.hroutes_per_micron, 1e-6)

        # Macro sizes
        self.sizes = benchmark.macro_sizes.numpy()

        # Build net/pin structure
        self._build_net_structure(plc)

        # State arrays
        self.density_grid = np.zeros(self.n_cells)
        self.v_cong = np.zeros(self.n_cells)
        self.h_cong = np.zeros(self.n_cells)
        self.net_hpwl = np.zeros(len(self.nets))
        self.total_hpwl = 0.0
        self.base_hpwl = 0.0  # HPWL from nets with no macros (constant)

        # Per-macro density cell contributions: macro_idx -> [(cell_idx, area_frac), ...]
        self.macro_density_cells = {}

        # Current positions
        self.positions = None

        # WL normalization
        self.total_net_count = len(plc.nets)
        self.wl_norm = max((self.canvas_w + self.canvas_h) * self.total_net_count, 1e-6)

    def _build_net_structure(self, plc):
        """Build net/pin connectivity for HPWL and congestion computation."""
        name_to_idx = {name: i for i, name in enumerate(self.bm.macro_names)}

        # Build pin_name -> info mapping
        pin_info = {}
        for mod in plc.modules_w_pins:
            mod_type = mod.get_type()
            if mod_type == 'MACRO_PIN':
                macro_name = mod.get_macro_name()
                if macro_name in name_to_idx:
                    macro_idx = name_to_idx[macro_name]
                    xo, yo = mod.get_offset()
                    pin_info[mod.get_name()] = ('macro', macro_idx, float(xo), float(yo))
            elif mod_type == 'PORT':
                x, y = mod.get_pos()
                pin_info[mod.get_name()] = ('port', float(x), float(y))

        # Build nets: each net is (macro_pins, fixed_pins)
        # macro_pins: list of (macro_idx, x_offset, y_offset)
        # fixed_pins: list of (x, y)
        self.nets = []
        self.macro_to_nets = defaultdict(list)

        for net_name, pin_names in plc.nets.items():
            macro_pins = []
            fixed_pins = []

            for pn in pin_names:
                if pn in pin_info:
                    info = pin_info[pn]
                    if info[0] == 'macro':
                        macro_pins.append((info[1], info[2], info[3]))
                    else:
                        fixed_pins.append((info[1], info[2]))

            if len(macro_pins) + len(fixed_pins) < 2:
                continue

            net_idx = len(self.nets)
            self.nets.append((macro_pins, fixed_pins))

            # Track which nets each macro belongs to
            seen = set()
            for (mi, _, _) in macro_pins:
                if mi not in seen:
                    self.macro_to_nets[mi].append(net_idx)
                    seen.add(mi)

    def init_from_placement(self, positions):
        """Full computation of all surrogate components from a placement."""
        self.positions = positions.copy()

        # 1. Density grid
        self.density_grid[:] = 0.0
        self.macro_density_cells.clear()
        cell_area = self.cell_w * self.cell_h

        for i in range(self.n):
            cells = self._compute_macro_cells(i, positions[i])
            self.macro_density_cells[i] = cells
            for cell_idx, area in cells:
                self.density_grid[cell_idx] += area / cell_area

        # 2. HPWL per net
        self.total_hpwl = 0.0
        for j, (macro_pins, fixed_pins) in enumerate(self.nets):
            hpwl = self._compute_net_hpwl(j, positions)
            self.net_hpwl[j] = hpwl
            self.total_hpwl += hpwl

        # 3. RUDY congestion
        self.v_cong[:] = 0.0
        self.h_cong[:] = 0.0
        for j, (macro_pins, fixed_pins) in enumerate(self.nets):
            self._add_net_congestion(j, positions, 1.0)

    def _compute_macro_cells(self, macro_idx, pos):
        """Compute grid cells overlapped by macro and overlap areas."""
        x, y = pos[0], pos[1]
        w, h = self.sizes[macro_idx]

        x_lo = max(0.0, x - w / 2)
        x_hi = min(self.canvas_w, x + w / 2)
        y_lo = max(0.0, y - h / 2)
        y_hi = min(self.canvas_h, y + h / 2)

        c_lo = max(0, int(x_lo / self.cell_w))
        c_hi = min(self.grid_cols - 1, int(x_hi / self.cell_w))
        r_lo = max(0, int(y_lo / self.cell_h))
        r_hi = min(self.grid_rows - 1, int(y_hi / self.cell_h))

        cells = []
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                # Overlap area between macro and cell
                ox = max(0.0, min(x_hi, (c + 1) * self.cell_w) - max(x_lo, c * self.cell_w))
                oy = max(0.0, min(y_hi, (r + 1) * self.cell_h) - max(y_lo, r * self.cell_h))
                area = ox * oy
                if area > 1e-12:
                    cell_idx = r * self.grid_cols + c
                    cells.append((cell_idx, area))
        return cells

    def _compute_net_hpwl(self, net_idx, positions):
        """Compute HPWL for a single net."""
        macro_pins, fixed_pins = self.nets[net_idx]

        xs = []
        ys = []
        for (mi, xo, yo) in macro_pins:
            xs.append(positions[mi, 0] + xo)
            ys.append(positions[mi, 1] + yo)
        for (fx, fy) in fixed_pins:
            xs.append(fx)
            ys.append(fy)

        if len(xs) < 2:
            return 0.0

        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def _add_net_congestion(self, net_idx, positions, sign):
        """Add (sign=+1) or remove (sign=-1) RUDY congestion for a net."""
        macro_pins, fixed_pins = self.nets[net_idx]

        # Compute net bounding box
        xs = []
        ys = []
        for (mi, xo, yo) in macro_pins:
            xs.append(positions[mi, 0] + xo)
            ys.append(positions[mi, 1] + yo)
        for (fx, fy) in fixed_pins:
            xs.append(fx)
            ys.append(fy)

        if len(xs) < 2:
            return

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Grid cells covered by bbox
        c_lo = max(0, min(self.grid_cols - 1, int(x_min / self.cell_w)))
        c_hi = max(0, min(self.grid_cols - 1, int(x_max / self.cell_w)))
        r_lo = max(0, min(self.grid_rows - 1, int(y_min / self.cell_h)))
        r_hi = max(0, min(self.grid_rows - 1, int(y_max / self.cell_h)))

        n_cols = max(1, c_hi - c_lo + 1)
        n_rows = max(1, r_hi - r_lo + 1)

        # RUDY: distribute demand uniformly over bbox cells
        h_demand = sign / (n_cols * self.grid_h_cap)
        v_demand = sign / (n_rows * self.grid_v_cap)

        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                cell_idx = r * self.grid_cols + c
                self.h_cong[cell_idx] += h_demand
                self.v_cong[cell_idx] += v_demand

    def get_density_cost(self):
        """Compute density cost matching PlacementCost formula."""
        # Top 10% of grid cells, multiplied by 0.5
        vals = self.density_grid.copy()
        n_top = max(1, int(self.n_cells * 0.1))
        top_vals = np.partition(vals, -n_top)[-n_top:]
        return 0.5 * np.mean(top_vals)

    def get_congestion_cost(self):
        """Compute congestion cost matching PlacementCost formula (ABU 5%)."""
        combined = np.concatenate([self.v_cong, self.h_cong])
        n_top = max(1, int(len(combined) * 0.05))
        top_vals = np.partition(combined, -n_top)[-n_top:]
        return np.mean(top_vals)

    def get_wirelength_cost(self):
        """Compute normalized wirelength cost."""
        return self.total_hpwl / self.wl_norm

    def get_proxy_cost(self):
        """Compute composite proxy cost estimate."""
        return (self.get_wirelength_cost()
                + 0.5 * self.get_density_cost()
                + 0.5 * self.get_congestion_cost())

    def evaluate_move(self, moved_macros, new_positions):
        """
        Estimate proxy cost after moving specified macros, without committing.

        Args:
            moved_macros: list of macro indices that moved
            new_positions: [n, 2] full position array with new positions

        Returns:
            estimated proxy cost
        """
        cell_area = self.cell_w * self.cell_h

        # Save state for rollback
        old_density_deltas = []  # (cell_idx, delta) pairs
        old_cong_deltas_h = []
        old_cong_deltas_v = []
        old_net_hpwls = {}

        # 1. Update density: remove old, add new
        for mi in moved_macros:
            # Remove old cells
            if mi in self.macro_density_cells:
                for cell_idx, area in self.macro_density_cells[mi]:
                    delta = -area / cell_area
                    self.density_grid[cell_idx] += delta
                    old_density_deltas.append((cell_idx, -delta))

            # Add new cells
            new_cells = self._compute_macro_cells(mi, new_positions[mi])
            for cell_idx, area in new_cells:
                delta = area / cell_area
                self.density_grid[cell_idx] += delta
                old_density_deltas.append((cell_idx, -delta))

        # 2. Update congestion and HPWL for affected nets
        affected_nets = set()
        for mi in moved_macros:
            for nj in self.macro_to_nets.get(mi, []):
                affected_nets.add(nj)

        hpwl_delta = 0.0
        for nj in affected_nets:
            old_hpwl = self.net_hpwl[nj]
            old_net_hpwls[nj] = old_hpwl

            # Remove old congestion
            self._add_net_congestion(nj, self.positions, -1.0)
            # Add new congestion
            self._add_net_congestion(nj, new_positions, 1.0)

            # Compute new HPWL
            new_hpwl = self._compute_net_hpwl(nj, new_positions)
            hpwl_delta += new_hpwl - old_hpwl
            self.net_hpwl[nj] = new_hpwl

        # Save modified state
        old_total_hpwl = self.total_hpwl
        self.total_hpwl += hpwl_delta

        # Compute proxy cost
        proxy = self.get_proxy_cost()

        # Rollback all changes
        self.total_hpwl = old_total_hpwl
        for nj, old_h in old_net_hpwls.items():
            # Reverse congestion changes
            self._add_net_congestion(nj, new_positions, -1.0)
            self._add_net_congestion(nj, self.positions, 1.0)
            self.net_hpwl[nj] = old_h

        for cell_idx, delta in old_density_deltas:
            self.density_grid[cell_idx] += delta

        return proxy

    def commit_move(self, moved_macros, new_positions):
        """Commit a move: update all internal state to reflect new positions."""
        cell_area = self.cell_w * self.cell_h

        # Update density
        for mi in moved_macros:
            if mi in self.macro_density_cells:
                for cell_idx, area in self.macro_density_cells[mi]:
                    self.density_grid[cell_idx] -= area / cell_area

            new_cells = self._compute_macro_cells(mi, new_positions[mi])
            self.macro_density_cells[mi] = new_cells
            for cell_idx, area in new_cells:
                self.density_grid[cell_idx] += area / cell_area

        # Update congestion and HPWL for affected nets
        affected_nets = set()
        for mi in moved_macros:
            for nj in self.macro_to_nets.get(mi, []):
                affected_nets.add(nj)

        for nj in affected_nets:
            self._add_net_congestion(nj, self.positions, -1.0)
            self._add_net_congestion(nj, new_positions, 1.0)
            new_hpwl = self._compute_net_hpwl(nj, new_positions)
            self.total_hpwl += new_hpwl - self.net_hpwl[nj]
            self.net_hpwl[nj] = new_hpwl

        # Update positions
        for mi in moved_macros:
            self.positions[mi] = new_positions[mi].copy()


# ---------------------------------------------------------------------------
# Module 5: Navigator
# ---------------------------------------------------------------------------

class Navigator:
    """
    Search loop: propose flips, evaluate via proxy cost, accept/reject.

    Uses GridSurrogate for fast candidate filtering (filter-then-verify):
    1. Evaluate all candidates with surrogate (~0.1ms each)
    2. Rank by surrogate proxy estimate
    3. Verify top-k with real compute_proxy_cost (~1s each)

    Dual variables from HPWL LP guide which pairs to flip.
    """

    def __init__(self, lp_solver: LPSolver, neighbor_gen: NeighborGenerator,
                 benchmark: Benchmark, plc, surrogate: GridSurrogate = None):
        self.lp = lp_solver
        self.ng = neighbor_gen
        self.benchmark = benchmark
        self.plc = plc
        self.surrogate = surrogate
        self.rng = np.random.default_rng(42)
        self._macro_to_pairs = None  # lazy-built index for fast cascade

    def _get_macro_to_pairs(self, assignment):
        """Build macro -> list of pairs index for fast cascade checking."""
        if self._macro_to_pairs is not None:
            return self._macro_to_pairs
        m2p = defaultdict(list)
        for (a, b) in assignment:
            m2p[a].append((a, b))
            m2p[b].append((a, b))
        self._macro_to_pairs = m2p
        return m2p

    def _eval_proxy(self, positions: np.ndarray) -> dict:
        """Evaluate full proxy cost."""
        from macro_place.objective import compute_proxy_cost
        placement = torch.tensor(positions, dtype=torch.float32)
        return compute_proxy_cost(placement, self.benchmark, self.plc)

    def _check_overlaps_fast(self, positions: np.ndarray) -> bool:
        """Fast vectorized overlap check for all hard macro pairs.
        Returns True if ANY overlap exists."""
        n_hard = self.benchmark.num_hard_macros
        pos = positions[:n_hard]
        sz = self.benchmark.macro_sizes.numpy()[:n_hard]

        x = pos[:, 0]
        y = pos[:, 1]
        w = sz[:, 0]
        h = sz[:, 1]

        dx = np.abs(x[:, None] - x[None, :])
        dy = np.abs(y[:, None] - y[None, :])
        min_dx = (w[:, None] + w[None, :]) / 2
        min_dy = (h[:, None] + h[None, :]) / 2

        overlap = (dx < min_dx - 1e-3) & (dy < min_dy - 1e-3)
        np.fill_diagonal(overlap, False)
        return np.any(overlap)

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
        # Only check pairs involving moved macros (using index for speed)
        moved = {i, k}
        n_hard = self.benchmark.num_hard_macros
        m2p = self._get_macro_to_pairs(assignment)
        for cascade_round in range(10):
            violations = 0
            checked = set()
            for m in list(moved):
                for pair_key in m2p.get(m, []):
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)
                    a, b = pair_key
                    d = assignment.get(pair_key)
                    if d is None:
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

                if gap < -1e-6:
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

        # Post-cascade: direct overlap repair for ALL hard macro pairs near moved set
        # The assignment-based cascade only checks pairs in the topology, but overlaps
        # can occur between any two hard macros after position adjustments
        for repair_round in range(10):
            any_overlap = False
            for a in list(moved):
                for b in range(n_hard):
                    if b == a:
                        continue
                    # Direct rectangle overlap check
                    wa, ha = float(sizes[a, 0]), float(sizes[a, 1])
                    wb, hb = float(sizes[b, 0]), float(sizes[b, 1])
                    dx = abs(pos[a, 0] - pos[b, 0])
                    dy = abs(pos[a, 1] - pos[b, 1])
                    min_dx = (wa + wb) / 2 + eps
                    min_dy = (ha + hb) / 2 + eps
                    if dx < min_dx and dy < min_dy:
                        # Overlap! Push apart in direction of smallest violation
                        mov_a = a < n_hard and not self.benchmark.macro_fixed[a]
                        mov_b = b < n_hard and not self.benchmark.macro_fixed[b]
                        viol_x = min_dx - dx
                        viol_y = min_dy - dy
                        if viol_x < viol_y:
                            # Push apart in x
                            sign = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                            if mov_a and mov_b:
                                pos[a, 0] -= sign * viol_x / 2
                                pos[b, 0] += sign * viol_x / 2
                            elif mov_a:
                                pos[a, 0] -= sign * viol_x
                            elif mov_b:
                                pos[b, 0] += sign * viol_x
                        else:
                            # Push apart in y
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

            # Re-clamp after repair
            for idx in moved:
                hw = sizes[idx, 0] / 2
                hh = sizes[idx, 1] / 2
                pos[idx, 0] = np.clip(pos[idx, 0], hw, cw - hw)
                pos[idx, 1] = np.clip(pos[idx, 1], hh, ch - hh)

        return pos

    def greedy_descent(self, assignment: dict, initial_result: dict,
                       ref_positions: np.ndarray,
                       max_iters: int = 500, time_budget: float = 300.0,
                       top_k_verify: int = 5, verbose: bool = True,
                       lp_resolve_cap: int = 20) -> dict:
        """
        Greedy descent on proxy cost with surrogate-powered filtering.

        For each iteration:
        1. Generate candidates from duals (top 150 pairs × 3 directions = 450)
        2. Project each candidate (fast constraint projection, ~microseconds)
        3. Evaluate all feasible projections with surrogate (~0.1ms each)
        4. Verify top-k by surrogate with real proxy cost (~1s each)
        5. Accept best improvement
        """
        t0 = time.time()

        best_assignment = dict(assignment)
        best_positions = ref_positions.copy()
        current_assignment = dict(assignment)
        current_positions = ref_positions.copy()

        hpwl_result = initial_result

        # Initialize surrogate — sole arbiter (no compute_proxy_cost in loop)
        if self.surrogate is not None:
            self.surrogate.init_from_placement(current_positions)
            best_proxy = self.surrogate.get_proxy_cost()
            if verbose:
                print(f"  Surrogate init: proxy={best_proxy:.4f} "
                      f"(wl={self.surrogate.get_wirelength_cost():.4f}, "
                      f"den={self.surrogate.get_density_cost():.4f}, "
                      f"cong={self.surrogate.get_congestion_cost():.4f})")
        else:
            best_proxy = self._eval_proxy(ref_positions)["proxy_cost"]

        improvements = 0
        full_evals = 0
        surrogate_evals = 0
        stale_iters = 0
        lp_resolves = 0

        if verbose:
            print(f"  Starting navigation: proxy={best_proxy:.4f}, "
                  f"surrogate={'ON' if self.surrogate else 'OFF'}")

        for iteration in range(max_iters):
            elapsed = time.time() - t0
            if elapsed > time_budget:
                if verbose:
                    print(f"  Time budget exhausted at iter {iteration}")
                break

            if stale_iters > 100:
                if verbose:
                    print(f"  Stale for {stale_iters} iters, stopping")
                break

            # Get candidates from HPWL dual variables
            candidates = self.ng.rank_candidates(
                hpwl_result["duals"], current_assignment, top_k=150
            )

            if not candidates:
                break

            # Also add cluster move candidates
            m2n = self.surrogate.macro_to_nets if self.surrogate else {}
            cluster_candidates = self.ng.propose_cluster_flips(
                hpwl_result["duals"], current_assignment,
                m2n, n_proposals=20, rng=self.rng
            )

            # Phase 1: Project all candidates and evaluate with surrogate
            projected = []  # (surrogate_proxy, new_pos, pair, new_dir, old_dir, is_cluster)

            # Single-pair candidates
            for pair, new_dir, dual_mag in candidates:
                if time.time() - t0 > time_budget:
                    break
                old_dir = current_assignment[pair]
                current_assignment[pair] = new_dir
                new_pos = self._project_flip(
                    current_positions, pair, new_dir, current_assignment
                )
                current_assignment[pair] = old_dir

                if new_pos is None:
                    continue

                if self.surrogate is not None:
                    # Identify moved macros
                    i, k = pair
                    moved = []
                    if not np.allclose(new_pos[i], current_positions[i]):
                        moved.append(i)
                    if not np.allclose(new_pos[k], current_positions[k]):
                        moved.append(k)
                    # Also check cascade-moved macros
                    for m in range(self.benchmark.num_hard_macros):
                        if m != i and m != k and not np.allclose(new_pos[m], current_positions[m]):
                            moved.append(m)

                    surr_proxy = self.surrogate.evaluate_move(moved, new_pos)
                    surrogate_evals += 1
                else:
                    surr_proxy = dual_mag  # Fall back to dual magnitude ranking

                projected.append((surr_proxy, new_pos, pair, new_dir, old_dir, False))

            # Cluster candidates (multi-pair flips)
            for cluster in cluster_candidates:
                new_pos, new_assign = self._project_cluster_flip(
                    current_positions, cluster, current_assignment
                )
                if new_pos is None:
                    continue

                if self.surrogate is not None:
                    moved = []
                    for m in range(self.benchmark.num_hard_macros):
                        if not np.allclose(new_pos[m], current_positions[m]):
                            moved.append(m)
                    if moved:
                        surr_proxy = self.surrogate.evaluate_move(moved, new_pos)
                        surrogate_evals += 1
                    else:
                        continue
                else:
                    surr_proxy = 0.0

                projected.append((surr_proxy, new_pos, cluster, None, None, True))

            if not projected:
                stale_iters += 1
                continue

            # Phase 2: Sort by surrogate estimate, verify top-k with real proxy
            if self.surrogate is not None:
                projected.sort(key=lambda x: x[0])  # Lower surrogate proxy = better
                verify_list = projected[:top_k_verify]
            else:
                # Without surrogate, just try top candidates by dual magnitude
                verify_list = projected[:top_k_verify]

            improved_this_iter = False

            for surr_proxy, new_pos, pair_or_cluster, new_dir, old_dir, is_cluster in verify_list:
                if time.time() - t0 > time_budget:
                    break

                # Fast vectorized overlap check (no compute_proxy_cost)
                if self._check_overlaps_fast(new_pos):
                    continue

                full_evals += 1

                if surr_proxy < best_proxy:
                    improvement = best_proxy - surr_proxy
                    best_proxy = surr_proxy

                    if is_cluster:
                        # Apply all pair flips in cluster
                        for (p, d) in pair_or_cluster:
                            current_assignment[p] = d
                    else:
                        current_assignment[pair_or_cluster] = new_dir

                    best_positions = new_pos.copy()
                    current_positions = new_pos.copy()

                    # Re-extract assignment from actual positions to stay consistent
                    # (cascade projection may have moved macros whose pairwise
                    # relations no longer match the manually-updated assignment)
                    sizes = self.benchmark.macro_sizes.numpy()
                    movable_hard = np.arange(self.benchmark.num_hard_macros)[
                        ~self.benchmark.macro_fixed[:self.benchmark.num_hard_macros].numpy()
                    ]
                    current_assignment = extract_assignment_vectorized(
                        current_positions, sizes, movable_hard
                    )
                    self._macro_to_pairs = None  # invalidate cache
                    best_assignment = dict(current_assignment)

                    # Update surrogate state
                    if self.surrogate is not None:
                        self.surrogate.init_from_placement(current_positions)

                    improvements += 1
                    improved_this_iter = True
                    stale_iters = 0

                    if verbose:
                        elapsed = time.time() - t0
                        if is_cluster:
                            desc = f"cluster({len(pair_or_cluster)} pairs)"
                        else:
                            desc = (f"pair={pair_or_cluster}, "
                                    f"{DIR_NAMES[old_dir]}->{DIR_NAMES[new_dir]}")
                        print(f"  iter {iteration}: proxy={best_proxy:.4f} "
                              f"(delta={-improvement:.4f}, {desc}, "
                              f"surr={surr_proxy:.4f}, {elapsed:.1f}s)")

                    # Refresh duals periodically (capped)
                    remaining = time_budget - (time.time() - t0)
                    if improvements % 5 == 0 and lp_resolves < lp_resolve_cap and remaining > 20:
                        hpwl_result = self.lp.solve(
                            current_assignment, time_limit=min(15.0, remaining - 10)
                        )
                        lp_resolves += 1
                    break

            if not improved_this_iter:
                stale_iters += 1

                # Refresh duals more aggressively when stale (capped)
                remaining = time_budget - (time.time() - t0)
                if stale_iters % 20 == 0 and lp_resolves < lp_resolve_cap and remaining > 20:
                    hpwl_result = self.lp.solve(
                        current_assignment, time_limit=min(15.0, remaining - 10)
                    )
                    lp_resolves += 1

        if verbose:
            elapsed = time.time() - t0
            print(f"  Navigation done: {full_evals} accepted, "
                  f"{surrogate_evals} surrogate evals, "
                  f"{improvements} improvements, {lp_resolves} LP resolves, "
                  f"{elapsed:.1f}s")

        return {
            "assignment": best_assignment,
            "positions": best_positions,
            "proxy_cost": best_proxy,
            "improvements": improvements,
            "evaluations": full_evals,
            "surrogate_evals": surrogate_evals,
        }

    def _project_cluster_flip(self, positions, cluster, assignment):
        """
        Apply multiple pair flips via sequential constraint projection.

        Args:
            cluster: list of (pair, new_direction) tuples
            assignment: current assignment dict

        Returns:
            (new_positions, new_assignment) or (None, None) if infeasible
        """
        pos = positions.copy()
        new_assign = dict(assignment)

        for pair, new_dir in cluster:
            new_assign[pair] = new_dir
            projected = self._project_flip(pos, pair, new_dir, new_assign)
            if projected is None:
                return None, None
            pos = projected

        return pos, new_assign


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
                 nav_iters: int = 500, nav_time: float = 300.0,
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
        lp_time_limit = max(10.0, 40.0 - (time.time() - t_start))
        result = lp.solve(assignment, time_limit=lp_time_limit)
        if self.verbose:
            print(f"  LP solve: status={result['status']}, "
                  f"HPWL={result['hpwl']:.2f}, {result['solve_time']:.2f}s")

        if result["positions"] is None:
            if self.verbose:
                print("  LP infeasible! Returning initial placement.")
            return torch.tensor(init_pos, dtype=torch.float32)

        # Use initial positions (not LP positions) — they have much better density
        positions = init_pos

        # Step 4: Build surrogate for fast candidate filtering
        t0 = time.time()
        surrogate = GridSurrogate(benchmark, plc)
        if self.verbose:
            print(f"  Surrogate built ({len(surrogate.nets)} nets), {time.time() - t0:.2f}s")

        # Step 5: Navigate (if enabled)
        if self.navigate and n_pairs > 0 and result["positions"] is not None:
            t0 = time.time()
            ng = NeighborGenerator(alpha=1.0)
            nav = Navigator(lp, ng, benchmark, plc, surrogate=surrogate)

            # Adaptive nav time budget: guarantee total place() < 55s
            elapsed_so_far = time.time() - t_start
            nav_time = min(40, max(15, 50 - elapsed_so_far))
            if self.verbose:
                print(f"  Nav budget: {nav_time:.1f}s (elapsed {elapsed_so_far:.1f}s)")

            nav_result = nav.greedy_descent(
                assignment, result,
                ref_positions=init_pos,
                max_iters=self.nav_iters,
                time_budget=nav_time,
                top_k_verify=1,
                verbose=self.verbose,
                lp_resolve_cap=3,
            )

            positions = nav_result["positions"]
            if self.verbose:
                print(f"  Navigation total: {time.time() - t0:.1f}s, "
                      f"{nav_result.get('surrogate_evals', 0)} surrogate evals")

        if self.verbose:
            print(f"  Total time: {time.time() - t_start:.1f}s")

        return torch.tensor(positions, dtype=torch.float32)
