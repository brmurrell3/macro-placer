"""
LP solver for the polyhedra placement framework.

Solves two LP variants within a given topology (assignment):
  1. HPWL minimization — optimal wirelength positions + dual variables
  2. Min-displacement — closest feasible positions to a reference placement

Both share the same separation constraint structure.
"""

import time
import numpy as np
import highspy

from assignment import L, R, B, A
from macro_place.benchmark import Benchmark


class LPSolver:
    """
    Solves LPs within a given polyhedron (assignment).

    The assignment defines pairwise separation directions for hard macros.
    The LP finds optimal positions subject to those constraints + canvas bounds.
    """

    def __init__(self, benchmark: Benchmark, plc, net_macros: list = None):
        self.benchmark = benchmark
        self.plc = plc
        self.n_hard = benchmark.num_hard_macros
        self.n_total = benchmark.num_macros
        self.canvas_w = benchmark.canvas_width
        self.canvas_h = benchmark.canvas_height
        self.sizes = benchmark.macro_sizes.numpy()
        self.fixed = benchmark.macro_fixed.numpy()
        self.fixed_pos = benchmark.macro_positions.numpy()

        if net_macros is not None:
            self.net_macros = net_macros
        else:
            self.net_macros = self._build_net_macros()

        self._last_h = None

    def _build_net_macros(self) -> list:
        """Extract net-to-macro mapping from plc."""
        b = self.benchmark
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

    # ------------------------------------------------------------------
    # Shared constraint building
    # ------------------------------------------------------------------

    def _build_position_bounds(self, n_vars, x_start, y_start,
                               ref_positions=None, pin_soft=False):
        """Build column bounds for position variables.

        Args:
            pin_soft: if True, pin soft macros (i >= n_hard) to ref_positions
        """
        n = self.n_total
        inf = highspy.kHighsInf
        col_lower = [-inf] * n_vars
        col_upper = [inf] * n_vars

        for i in range(n):
            hw = self.sizes[i, 0] / 2
            hh = self.sizes[i, 1] / 2
            col_lower[x_start + i] = float(hw)
            col_upper[x_start + i] = float(self.canvas_w - hw)
            col_lower[y_start + i] = float(hh)
            col_upper[y_start + i] = float(self.canvas_h - hh)

        # Fix fixed macros
        for i in range(n):
            if self.fixed[i]:
                col_lower[x_start + i] = float(self.fixed_pos[i, 0])
                col_upper[x_start + i] = float(self.fixed_pos[i, 0])
                col_lower[y_start + i] = float(self.fixed_pos[i, 1])
                col_upper[y_start + i] = float(self.fixed_pos[i, 1])

        # Pin soft macros if requested
        if pin_soft and ref_positions is not None:
            for i in range(n):
                if i >= self.n_hard:
                    col_lower[x_start + i] = float(ref_positions[i, 0])
                    col_upper[x_start + i] = float(ref_positions[i, 0])
                    col_lower[y_start + i] = float(ref_positions[i, 1])
                    col_upper[y_start + i] = float(ref_positions[i, 1])

        return col_lower, col_upper

    def _add_separation_constraints(self, assignment, col_indices, values,
                                    row_starts, row_lower, row_upper,
                                    x_start, y_start,
                                    ref_positions=None, sparse_margin=0.0,
                                    eps=0.002):
        """Add separation constraints from assignment to the constraint matrix.

        Returns sep_constraint_map: dict (i,k) -> row index (for dual extraction).
        """
        inf = highspy.kHighsInf
        sep_constraint_map = {}
        skipped_pairs = 0

        for (i, k), direction in assignment.items():
            # Sparse filtering: skip pairs far apart
            if ref_positions is not None and sparse_margin > 0:
                if direction in (L, R):
                    min_sep = (float(self.sizes[i, 0]) + float(self.sizes[k, 0])) / 2
                    actual_sep = abs(ref_positions[i, 0] - ref_positions[k, 0])
                    if actual_sep > min_sep + sparse_margin:
                        skipped_pairs += 1
                        continue
                else:
                    min_sep = (float(self.sizes[i, 1]) + float(self.sizes[k, 1])) / 2
                    actual_sep = abs(ref_positions[i, 1] - ref_positions[k, 1])
                    if actual_sep > min_sep + sparse_margin:
                        skipped_pairs += 1
                        continue

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

        return sep_constraint_map

    def _solve_lp(self, n_vars, n_rows, col_cost, col_lower, col_upper,
                  row_starts, col_indices, values, row_lower, row_upper,
                  time_limit):
        """Build and solve a HiGHS LP. Returns (Highs instance, status, solve_time)."""
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
        return h, status

    def _extract_positions(self, h, x_start, y_start, n):
        """Extract position arrays from a solved HiGHS model."""
        sol = h.getSolution()
        col_vals = list(sol.col_value)
        x_pos = np.array(col_vals[x_start:x_start + n])
        y_pos = np.array(col_vals[y_start:y_start + n])
        return np.stack([x_pos, y_pos], axis=1)

    # ------------------------------------------------------------------
    # Public solve methods
    # ------------------------------------------------------------------

    def solve(self, assignment: dict, warm_start: bool = False,
              time_limit: float = 60.0, ref_positions: np.ndarray = None,
              sparse_margin: float = 0.0) -> dict:
        """
        Solve the HPWL LP for the given assignment.

        Returns dict with: positions, hpwl, duals, status, solve_time, sep_constraint_map
        """
        t0 = time.time()

        n = self.n_total
        n_nets = len(self.net_macros)
        inf = highspy.kHighsInf

        # Variables: x_i, y_i, plus 4 bounding box vars per net
        n_vars = 2 * n + 4 * n_nets
        x_start = 0
        y_start = n
        net_var_start = 2 * n

        # Objective: minimize HPWL = sum(u_max - u_min + v_max - v_min)
        col_cost = [0.0] * n_vars
        for j in range(n_nets):
            base = net_var_start + 4 * j
            col_cost[base + 0] = 1.0   # u_max
            col_cost[base + 1] = -1.0  # u_min
            col_cost[base + 2] = 1.0   # v_max
            col_cost[base + 3] = -1.0  # v_min

        # Position bounds
        col_lower, col_upper = self._build_position_bounds(n_vars, x_start, y_start)
        # Net bounding box vars are unconstrained
        for j in range(n_nets):
            base = net_var_start + 4 * j
            for offset in range(4):
                col_lower[base + offset] = -inf
                col_upper[base + offset] = inf

        # Build constraints
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

                # u_min_j <= x_i
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

        # 2. Separation constraints
        sep_constraint_map = self._add_separation_constraints(
            assignment, col_indices, values, row_starts, row_lower, row_upper,
            x_start, y_start, ref_positions, sparse_margin
        )

        n_rows = len(row_lower)

        h, status = self._solve_lp(
            n_vars, n_rows, col_cost, col_lower, col_upper,
            row_starts, col_indices, values, row_lower, row_upper,
            time_limit
        )

        solve_time = time.time() - t0

        if status != 2:  # 2 = feasible
            return {
                "positions": None,
                "hpwl": float("inf"),
                "duals": {},
                "status": f"infeasible (status={status})",
                "solve_time": solve_time,
            }

        positions = self._extract_positions(h, x_start, y_start, n)
        hpwl = h.getInfoValue("objective_function_value")[1]

        # Extract dual variables for separation constraints
        row_duals = list(h.getSolution().row_dual)
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
                               time_limit: float = 30.0,
                               sparse_margin: float = 5.0) -> dict:
        """
        Find closest feasible positions to ref_positions under the given topology.

        Minimizes sum_i |x_i - x_ref_i| + |y_i - y_ref_i| (L1 displacement).
        """
        t0 = time.time()

        n = self.n_total
        inf = highspy.kHighsInf

        # Variables: x_i, y_i, plus dx+, dx-, dy+, dy- per macro
        n_vars = 2 * n + 4 * n
        x_start = 0
        y_start = n
        slack_start = 2 * n

        col_cost = [0.0] * n_vars

        # Position bounds (pin soft macros to ref)
        col_lower, col_upper = self._build_position_bounds(
            n_vars, x_start, y_start, ref_positions, pin_soft=True
        )

        # Also pin fixed hard macros to ref for displacement LP
        for i in range(n):
            if self.fixed[i]:
                col_lower[x_start + i] = float(ref_positions[i, 0])
                col_upper[x_start + i] = float(ref_positions[i, 0])
                col_lower[y_start + i] = float(ref_positions[i, 1])
                col_upper[y_start + i] = float(ref_positions[i, 1])

        movable_hard = set()
        for i in range(self.n_hard):
            if not self.fixed[i]:
                movable_hard.add(i)

        # Slack variables: non-negative, cost 1.0 for movable hard macros
        for i in range(n):
            base = slack_start + 4 * i
            for j in range(4):
                col_lower[base + j] = 0.0
                col_upper[base + j] = inf
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
            col_indices.extend([x_start + i, base + 0, base + 1])
            values.extend([1.0, -1.0, 1.0])
            row_starts.append(len(col_indices))
            row_lower.append(float(ref_positions[i, 0]))
            row_upper.append(float(ref_positions[i, 0]))

            col_indices.extend([y_start + i, base + 2, base + 3])
            values.extend([1.0, -1.0, 1.0])
            row_starts.append(len(col_indices))
            row_lower.append(float(ref_positions[i, 1]))
            row_upper.append(float(ref_positions[i, 1]))

        # 2. Separation constraints (sparse)
        self._add_separation_constraints(
            assignment, col_indices, values, row_starts, row_lower, row_upper,
            x_start, y_start, ref_positions, sparse_margin
        )

        n_rows = len(row_lower)

        h, status = self._solve_lp(
            n_vars, n_rows, col_cost, col_lower, col_upper,
            row_starts, col_indices, values, row_lower, row_upper,
            time_limit
        )

        solve_time = time.time() - t0

        if status != 2:
            return {
                "positions": None,
                "displacement": float("inf"),
                "status": f"infeasible (status={status})",
                "solve_time": solve_time,
            }

        positions = self._extract_positions(h, x_start, y_start, n)
        displacement = h.getInfoValue("objective_function_value")[1]

        return {
            "positions": positions,
            "displacement": displacement,
            "status": "optimal",
            "solve_time": solve_time,
        }
