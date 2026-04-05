"""
Surrogate proxy cost estimators for fast candidate filtering.

The Surrogate protocol defines what any surrogate must support.
GridSurrogate is the default implementation using grid-based density,
RUDY congestion, and HPWL wirelength.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Protocol, runtime_checkable


@runtime_checkable
class Surrogate(Protocol):
    """Protocol for surrogate proxy cost estimators.

    Any surrogate must support:
    - init_from_placement: full recompute from scratch
    - evaluate_move: estimate cost after moving macros (without committing)
    - commit_move: commit a move and update internal state
    - get_proxy_cost: return current estimated proxy cost
    """

    def init_from_placement(self, positions: np.ndarray) -> None: ...
    def evaluate_move(self, moved_macros: list, new_positions: np.ndarray) -> float: ...
    def commit_move(self, moved_macros: list, new_positions: np.ndarray) -> None: ...
    def get_proxy_cost(self) -> float: ...
    def get_wirelength_cost(self) -> float: ...
    def get_density_cost(self) -> float: ...
    def get_congestion_cost(self) -> float: ...


class GridSurrogate:
    """
    Fast incremental proxy cost estimator using grid-based density,
    RUDY congestion, and HPWL wirelength.

    ~0.1ms per evaluate_move call — designed for filtering candidates
    before expensive full proxy cost verification.
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
        self.base_hpwl = 0.0

        # Per-macro density cell contributions
        self.macro_density_cells = {}

        # Current positions
        self.positions = None

        # WL normalization
        self.total_net_count = len(plc.nets)
        self.wl_norm = max((self.canvas_w + self.canvas_h) * self.total_net_count, 1e-6)

    def _build_net_structure(self, plc):
        """Build net/pin connectivity for HPWL and congestion computation."""
        name_to_idx = {name: i for i, name in enumerate(self.bm.macro_names)}

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

        c_lo = max(0, min(self.grid_cols - 1, int(x_min / self.cell_w)))
        c_hi = max(0, min(self.grid_cols - 1, int(x_max / self.cell_w)))
        r_lo = max(0, min(self.grid_rows - 1, int(y_min / self.cell_h)))
        r_hi = max(0, min(self.grid_rows - 1, int(y_max / self.cell_h)))

        n_cols = max(1, c_hi - c_lo + 1)
        n_rows = max(1, r_hi - r_lo + 1)

        h_demand = sign / (n_cols * self.grid_h_cap)
        v_demand = sign / (n_rows * self.grid_v_cap)

        rows = np.arange(r_lo, r_hi + 1)
        cols = np.arange(c_lo, c_hi + 1)
        idxs = (rows[:, None] * self.grid_cols + cols).ravel()
        self.h_cong[idxs] += h_demand
        self.v_cong[idxs] += v_demand

    def get_density_cost(self):
        """Compute density cost matching PlacementCost formula."""
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

        Returns estimated proxy cost.
        """
        cell_area = self.cell_w * self.cell_h

        # Save state for rollback
        old_density_deltas = []
        old_cong_deltas_h = []
        old_cong_deltas_v = []
        old_net_hpwls = {}

        # 1. Update density: remove old, add new
        for mi in moved_macros:
            if mi in self.macro_density_cells:
                for cell_idx, area in self.macro_density_cells[mi]:
                    delta = -area / cell_area
                    self.density_grid[cell_idx] += delta
                    old_density_deltas.append((cell_idx, -delta))

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

            self._add_net_congestion(nj, self.positions, -1.0)
            self._add_net_congestion(nj, new_positions, 1.0)

            new_hpwl = self._compute_net_hpwl(nj, new_positions)
            hpwl_delta += new_hpwl - old_hpwl
            self.net_hpwl[nj] = new_hpwl

        old_total_hpwl = self.total_hpwl
        self.total_hpwl += hpwl_delta

        proxy = self.get_proxy_cost()

        # Rollback
        self.total_hpwl = old_total_hpwl
        for nj, old_h in old_net_hpwls.items():
            self._add_net_congestion(nj, new_positions, -1.0)
            self._add_net_congestion(nj, self.positions, 1.0)
            self.net_hpwl[nj] = old_h

        for cell_idx, delta in old_density_deltas:
            self.density_grid[cell_idx] += delta

        return proxy

    def commit_move(self, moved_macros, new_positions):
        """Commit a move: update all internal state to reflect new positions."""
        cell_area = self.cell_w * self.cell_h

        for mi in moved_macros:
            if mi in self.macro_density_cells:
                for cell_idx, area in self.macro_density_cells[mi]:
                    self.density_grid[cell_idx] -= area / cell_area

            new_cells = self._compute_macro_cells(mi, new_positions[mi])
            self.macro_density_cells[mi] = new_cells
            for cell_idx, area in new_cells:
                self.density_grid[cell_idx] += area / cell_area

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

        for mi in moved_macros:
            self.positions[mi] = new_positions[mi].copy()
