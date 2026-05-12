"""
Incremental proxy-cost evaluator for macro placement.

This module exposes ``IncrementalProxyEvaluator`` — a class that mirrors
``compute_proxy_cost`` (wirelength + density + congestion proxy cost from the
TILOS PlacementCost model) but is designed for **single-macro moves**:

    eval = IncrementalProxyEvaluator(benchmark, plc, placement)
    result = eval.move(macro_idx, new_xy)   # O(degree)
    eval.revert()                           # undoes the most recent move

This is the prerequisite for coordinate descent / large-neighborhood search
recipes (vmallela's leaderboard recipe). A full ``compute_proxy_cost`` call is
O(macros * nets); a ``move`` here is O(degree of moved macro).

Bit-for-bit parity with ``compute_proxy_cost`` is enforced by
``test/test_incremental_evaluator.py``.

Conventions:
    * Cost values match the un-weighted components (``wl``, ``density``,
      ``congestion``) and the weighted ``proxy`` (default weights {wl: 1.0,
      density: 0.5, congestion: 0.5}).
    * Coordinates are PlacementCost-style: macro center (x, y) in microns.
    * ``move`` only updates internal state — it does NOT mutate the underlying
      ``plc`` object. To apply the move to ``plc``, call
      ``compute_proxy_cost(eval.placement, benchmark, plc)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark


# ── Snapshot for revert() ────────────────────────────────────────────────────


@dataclass
class _MoveSnapshot:
    """Captured state needed to undo the most recent ``move`` call."""

    macro_idx: int
    old_xy: Tuple[float, float]
    # Per-net wirelength bbox + contributions before the move
    affected_net_indices: List[int]
    old_net_min_x: torch.Tensor  # [k]
    old_net_max_x: torch.Tensor
    old_net_min_y: torch.Tensor
    old_net_max_y: torch.Tensor
    old_net_hpwl: torch.Tensor   # [k] weighted (max_x-min_x + max_y-min_y) * weight
    # Density contribution before the move (dict cell_idx -> area)
    old_density_contrib: Dict[int, float]
    # Per-net (raw) congestion contributions before the move
    # net_idx -> dict of {(orient, cell_idx): increment}
    # orient: 0=H, 1=V
    old_net_cong_contrib: Dict[int, Dict[Tuple[int, int], float]]
    # Macro-routing contributions (this macro only) before the move
    # dict {(orient, cell_idx): value} — orient: 0=H, 1=V
    old_macro_cong_contrib: Dict[Tuple[int, int], float]


# ── Evaluator ────────────────────────────────────────────────────────────────


class IncrementalProxyEvaluator:
    """Incremental proxy-cost evaluator.

    Initialize from a benchmark + plc + placement, then call ``move`` to apply
    single-macro updates. Each ``move`` returns the updated cost dict.

    Args:
        benchmark: Benchmark dataclass (gives canvas/grid + macro indices map).
        plc: PlacementCost — the source of truth for net/pin topology and the
            congestion routing constants. Not mutated by this class.
        placement: ``[num_macros, 2]`` tensor of initial macro centers.
        weights: optional cost weights, default
            ``{wirelength: 1.0, density: 0.5, congestion: 0.5}``.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        plc: PlacementCost,
        placement: torch.Tensor,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.benchmark = benchmark
        self.plc = plc
        self.weights = weights or {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}

        # Canvas + grid (mirrors PlacementCost — single source of truth is plc)
        self.width = float(plc.width)
        self.height = float(plc.height)
        self.grid_col = int(plc.grid_col)
        self.grid_row = int(plc.grid_row)
        self.grid_width = self.width / self.grid_col
        self.grid_height = self.height / self.grid_row
        self.num_cells = self.grid_col * self.grid_row

        # Routing constants
        self.hroutes_per_micron = float(plc.hroutes_per_micron)
        self.vroutes_per_micron = float(plc.vroutes_per_micron)
        self.grid_v_routes = self.grid_width * self.vroutes_per_micron
        self.grid_h_routes = self.grid_height * self.hroutes_per_micron
        self.hrouting_alloc = float(plc.hrouting_alloc)
        self.vrouting_alloc = float(plc.vrouting_alloc)
        self.smooth_range = int(plc.smooth_range)
        self.net_cnt = float(plc.net_cnt) if plc.net_cnt > 0 else 1.0

        # Macro sizes (read from plc as Python floats — bench tensors are
        # float32 which loses precision and breaks bit-for-bit parity with plc).
        # Indexed [num_macros, 2] in the same order as benchmark.
        sizes = []
        for plc_idx in benchmark.hard_macro_indices + benchmark.soft_macro_indices:
            mod = plc.modules_w_pins[plc_idx]
            sizes.append((float(mod.get_width()), float(mod.get_height())))
        self.macro_sizes = torch.tensor(sizes, dtype=torch.float64)

        # Placement state (tensor, kept in sync via move/revert).
        # We store float64 but the values are whatever the caller hands in —
        # if they pass float32 (as compute_proxy_cost will round-trip), each
        # float() gives a float32-precision Python float, exactly matching
        # plc.set_pos(float32_value) behavior.
        self.placement = placement.detach().clone().to(torch.float64).cpu()

        # Build pin / net topology from plc
        self._build_topology()

        # Initialize cached pin positions
        self._compute_all_pin_positions()

        # Initialize per-net bbox, density grid, congestion contributions
        self._init_wirelength()
        self._init_density()
        self._init_congestion()

        # Snapshot stack for revert (single-step undo)
        self._snapshot: Optional[_MoveSnapshot] = None

    # ── Topology extraction ────────────────────────────────────────────────

    def _build_topology(self) -> None:
        """Extract per-pin parent-macro index, offset, and per-net pin lists.

        After this method:
            self.num_pins                int
            self.pin_parent              [num_pins] long; -1 if port (fixed)
            self.pin_offset_x/y          [num_pins] float64 (relative to parent)
            self.pin_fixed_x/y           [num_pins] float64 (port absolute pos)
            self.macro_to_pins           list of LongTensor (pins on each macro)
            self.macro_to_nets           list of LongTensor (unique nets touched)
            self.num_nets                int
            self.net_pins                list of LongTensor (pins per net)
            self.net_weight              [num_nets] float64
            self.pin_to_nets             list of LongTensor (nets per pin)
        """
        plc = self.plc
        bench = self.benchmark
        num_modules = len(plc.modules_w_pins)

        # Map plc module idx -> our macro idx (placement tensor index)
        self.plc_to_macro = {}
        for i, plc_idx in enumerate(bench.hard_macro_indices):
            self.plc_to_macro[plc_idx] = i
        for i, plc_idx in enumerate(bench.soft_macro_indices):
            self.plc_to_macro[plc_idx] = bench.num_hard_macros + i

        # Walk every module: build pin list (every PORT and every MACRO_PIN
        # is a "pin" for HPWL purposes). Macros themselves contribute to
        # density + macro routing congestion (not as pins).
        pin_parent = []
        pin_offset_x = []
        pin_offset_y = []
        pin_fixed_x = []
        pin_fixed_y = []
        # For each plc module idx that is a pin/port, record our pin idx
        plc_to_pin = {}

        for plc_idx, mod in enumerate(plc.modules_w_pins):
            mtype = mod.get_type()
            if mtype == "PORT":
                # Port pin — uses its own absolute position, no parent
                px, py = mod.get_pos()
                plc_to_pin[plc_idx] = len(pin_parent)
                pin_parent.append(-1)
                pin_offset_x.append(0.0)
                pin_offset_y.append(0.0)
                pin_fixed_x.append(float(px))
                pin_fixed_y.append(float(py))
            elif mtype == "MACRO_PIN":
                # Pin attached to a macro (soft or hard)
                macro_name = mod.get_macro_name()
                parent_plc_idx = plc.mod_name_to_indices[macro_name]
                macro_idx = self.plc_to_macro.get(parent_plc_idx, -1)
                ox, oy = mod.get_offset()
                plc_to_pin[plc_idx] = len(pin_parent)
                pin_parent.append(macro_idx)
                pin_offset_x.append(float(ox))
                pin_offset_y.append(float(oy))
                pin_fixed_x.append(0.0)
                pin_fixed_y.append(0.0)
            # MACRO modules themselves are not "pins" for HPWL.

        self.num_pins = len(pin_parent)
        self.pin_parent = torch.tensor(pin_parent, dtype=torch.long)
        self.pin_offset_x = torch.tensor(pin_offset_x, dtype=torch.float64)
        self.pin_offset_y = torch.tensor(pin_offset_y, dtype=torch.float64)
        self.pin_fixed_x = torch.tensor(pin_fixed_x, dtype=torch.float64)
        self.pin_fixed_y = torch.tensor(pin_fixed_y, dtype=torch.float64)

        # Build nets from plc.nets (driver_pin_name -> [sink_pin_names])
        net_pins: List[List[int]] = []
        net_weight: List[float] = []
        for driver_name, sink_names in plc.nets.items():
            driver_plc_idx = plc.mod_name_to_indices[driver_name]
            driver_pin_idx = plc_to_pin.get(driver_plc_idx)
            if driver_pin_idx is None:
                # Driver isn't a pin/port (shouldn't happen)
                continue
            pins = [driver_pin_idx]
            for sn in sink_names:
                sn_plc_idx = plc.mod_name_to_indices[sn]
                sn_pin_idx = plc_to_pin.get(sn_plc_idx)
                if sn_pin_idx is None:
                    continue
                pins.append(sn_pin_idx)
            # Net weight: from driver pin
            driver_mod = plc.modules_w_pins[driver_plc_idx]
            w = float(driver_mod.get_weight()) if hasattr(driver_mod, "get_weight") else 1.0
            net_pins.append(pins)
            net_weight.append(w)

        self.num_nets = len(net_pins)
        self.net_pins = [torch.tensor(p, dtype=torch.long) for p in net_pins]
        self.net_weight = torch.tensor(net_weight, dtype=torch.float64)

        # Per-pin -> nets index (for finding affected nets when a macro moves)
        pin_to_nets: List[List[int]] = [[] for _ in range(self.num_pins)]
        for net_idx, pins in enumerate(net_pins):
            for p in pins:
                pin_to_nets[p].append(net_idx)
        self.pin_to_nets = [torch.tensor(p, dtype=torch.long) for p in pin_to_nets]

        # Per-macro -> pins, and per-macro -> unique nets touched
        macro_to_pins: List[List[int]] = [[] for _ in range(bench.num_macros)]
        for pin_idx, parent in enumerate(pin_parent):
            if parent >= 0:
                macro_to_pins[parent].append(pin_idx)
        self.macro_to_pins = [torch.tensor(p, dtype=torch.long) for p in macro_to_pins]

        macro_to_nets: List[List[int]] = []
        for pin_list in macro_to_pins:
            seen = set()
            for p in pin_list:
                for n in pin_to_nets[p]:
                    seen.add(n)
            macro_to_nets.append(sorted(seen))
        self.macro_to_nets = [torch.tensor(n, dtype=torch.long) for n in macro_to_nets]

        # Hard macros mapping (only hard macros contribute to macro routing)
        self.is_hard_macro = torch.zeros(bench.num_macros, dtype=torch.bool)
        self.is_hard_macro[: bench.num_hard_macros] = True

    # ── Coordinate helpers ─────────────────────────────────────────────────

    def _grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Patched __get_grid_cell_location with bounds clamping (matches objective.py)."""
        row = int(math.floor(y / self.grid_height))
        col = int(math.floor(x / self.grid_width))
        row = max(0, min(row, self.grid_row - 1))
        col = max(0, min(col, self.grid_col - 1))
        return row, col

    def _pin_pos(self, pin_idx: int) -> Tuple[float, float]:
        parent = int(self.pin_parent[pin_idx])
        if parent < 0:
            return (
                float(self.pin_fixed_x[pin_idx]),
                float(self.pin_fixed_y[pin_idx]),
            )
        return (
            float(self.placement[parent, 0]) + float(self.pin_offset_x[pin_idx]),
            float(self.placement[parent, 1]) + float(self.pin_offset_y[pin_idx]),
        )

    def _all_pin_positions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached tensors of absolute (x, y) per pin.

        Recomputed once at init; ``move`` updates only the affected pins
        (those on the moved macro).
        """
        return self.pin_x, self.pin_y

    def _compute_all_pin_positions(self) -> None:
        """Initialize self.pin_x / self.pin_y from current placement."""
        parent = self.pin_parent
        is_port = parent < 0
        safe_parent = torch.where(is_port, torch.zeros_like(parent), parent)
        parent_xy = self.placement[safe_parent]
        self.pin_x = torch.where(
            is_port,
            self.pin_fixed_x,
            parent_xy[:, 0] + self.pin_offset_x,
        )
        self.pin_y = torch.where(
            is_port,
            self.pin_fixed_y,
            parent_xy[:, 1] + self.pin_offset_y,
        )

    def _update_macro_pin_positions(self, macro_idx: int) -> None:
        """Update cached pin_x/pin_y entries for pins on macro_idx."""
        pins = self.macro_to_pins[macro_idx]
        if len(pins) == 0:
            return
        px = float(self.placement[macro_idx, 0])
        py = float(self.placement[macro_idx, 1])
        self.pin_x[pins] = px + self.pin_offset_x[pins]
        self.pin_y[pins] = py + self.pin_offset_y[pins]

    # ── Wirelength initialization ──────────────────────────────────────────

    def _init_wirelength(self) -> None:
        """Compute per-net (min_x, max_x, min_y, max_y) from current placement."""
        x, y = self._all_pin_positions()
        self.net_min_x = torch.zeros(self.num_nets, dtype=torch.float64)
        self.net_max_x = torch.zeros(self.num_nets, dtype=torch.float64)
        self.net_min_y = torch.zeros(self.num_nets, dtype=torch.float64)
        self.net_max_y = torch.zeros(self.num_nets, dtype=torch.float64)
        self.net_hpwl = torch.zeros(self.num_nets, dtype=torch.float64)
        for n in range(self.num_nets):
            pins = self.net_pins[n]
            xs = x[pins]
            ys = y[pins]
            mnx, mxx = xs.min().item(), xs.max().item()
            mny, mxy = ys.min().item(), ys.max().item()
            self.net_min_x[n] = mnx
            self.net_max_x[n] = mxx
            self.net_min_y[n] = mny
            self.net_max_y[n] = mxy
            w = float(self.net_weight[n])
            self.net_hpwl[n] = w * ((mxx - mnx) + (mxy - mny))
        self.total_hpwl = float(self.net_hpwl.sum().item())

    # ── Density initialization ─────────────────────────────────────────────

    def _macro_cell_contrib(self, mod_x: float, mod_y: float, mod_w: float, mod_h: float) -> Dict[int, float]:
        """Compute per-cell occupied-area dict for a macro at (mod_x, mod_y).

        Mirrors PlacementCost.__add_module_to_grid_cells with the
        macro_place.objective bounds-clamping monkey-patch on
        __get_grid_cell_location applied.
        """
        ur_x = mod_x + mod_w / 2
        ur_y = mod_y + mod_h / 2
        bl_x = mod_x - mod_w / 2
        bl_y = mod_y - mod_h / 2

        # Use the SAME clamped-grid-cell call that the patched plc uses (see
        # macro_place/objective.py monkey-patch). Both ur and bl are clamped
        # into [0, grid_*-1], so the OOB skip branches never fire.
        ur_row, ur_col = self._grid_cell(ur_x, ur_y)
        bl_row, bl_col = self._grid_cell(bl_x, bl_y)

        out: Dict[int, float] = {}

        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                cell_x_min = c_i * self.grid_width
                cell_x_max = (c_i + 1) * self.grid_width
                cell_y_min = r_i * self.grid_height
                cell_y_max = (r_i + 1) * self.grid_height
                ox = min(cell_x_max, ur_x) - max(cell_x_min, bl_x)
                oy = min(cell_y_max, ur_y) - max(cell_y_min, bl_y)
                if ox > 0 and oy > 0:
                    out[r_i * self.grid_col + c_i] = ox * oy
        return out

    def _init_density(self) -> None:
        """Compute initial occupied-area per cell + per-macro contributions."""
        bench = self.benchmark
        self.grid_occupied = torch.zeros(self.num_cells, dtype=torch.float64)
        self.macro_density_contrib: List[Dict[int, float]] = [None] * bench.num_macros  # type: ignore
        for i in range(bench.num_macros):
            x = float(self.placement[i, 0])
            y = float(self.placement[i, 1])
            w = float(self.macro_sizes[i, 0])
            h = float(self.macro_sizes[i, 1])
            contrib = self._macro_cell_contrib(x, y, w, h)
            self.macro_density_contrib[i] = contrib
            for cell, area in contrib.items():
                self.grid_occupied[cell] += area
        self.grid_area = self.grid_width * self.grid_height

    # ── Congestion initialization ──────────────────────────────────────────

    def _net_cong_contrib(self, net_idx: int) -> Dict[Tuple[int, int], float]:
        """Compute per-cell V/H routing congestion increments contributed by one net.

        Mirrors PlacementCost.get_routing inner loop:
          - extract unique gcells of pins in net (driver_gcell separately)
          - dispatch on count: 2 -> two_pin, 3 -> three_pin, >3 -> split into 2-pin

        Returns dict {(orient, cell_idx): weight_increment}, where orient = 0
        for H_routing_cong, 1 for V_routing_cong.
        """
        pins = self.net_pins[net_idx]
        if len(pins) == 0:
            return {}
        # Driver is pins[0] by construction in _build_topology
        x, y = self._all_pin_positions()
        # Compute pin gcells
        pin_xs = x[pins]
        pin_ys = y[pins]
        # Use _grid_cell semantics (with clamp). NOTE: PlacementCost itself
        # uses an unpatched location for routing — we use the patched one to
        # match macro_place.objective.compute_proxy_cost (which monkey-patches
        # __get_grid_cell_location with clamping).
        gcells = []
        for i in range(len(pins)):
            gcells.append(self._grid_cell(float(pin_xs[i]), float(pin_ys[i])))
        source_gcell = gcells[0]
        unique = []
        seen = set()
        for g in gcells:
            if g not in seen:
                seen.add(g)
                unique.append(g)
        weight = float(self.net_weight[net_idx])
        # The driver pin's weight is what plc uses (driver_pin.get_weight()),
        # which we already captured into net_weight.

        out: Dict[Tuple[int, int], float] = {}

        def add(orient: int, row: int, col: int, w: float) -> None:
            key = (orient, row * self.grid_col + col)
            out[key] = out.get(key, 0.0) + w

        n = len(unique)
        if n == 2:
            self._two_pin_routing_dict(source_gcell, unique, weight, add)
        elif n == 3:
            self._three_pin_routing_dict(source_gcell, unique, weight, add)
        elif n > 3:
            for sink_g in unique:
                if sink_g == source_gcell:
                    continue
                self._two_pin_routing_dict(source_gcell, [source_gcell, sink_g], weight, add)
        return out

    def _two_pin_routing_dict(self, source_gcell, node_gcells, weight, add) -> None:
        """Mirror PlacementCost.__two_pin_net_routing."""
        if node_gcells[0] == source_gcell:
            sink_gcell = node_gcells[1]
        else:
            sink_gcell = node_gcells[0]
        row_min = min(sink_gcell[0], source_gcell[0])
        row_max = max(sink_gcell[0], source_gcell[0])
        col_min = min(sink_gcell[1], source_gcell[1])
        col_max = max(sink_gcell[1], source_gcell[1])
        for col_idx in range(col_min, col_max):
            add(0, source_gcell[0], col_idx, weight)  # H
        for row_idx in range(row_min, row_max):
            add(1, row_idx, sink_gcell[1], weight)  # V

    def _three_pin_routing_dict(self, source_gcell, node_gcells, weight, add) -> None:
        """Mirror PlacementCost.__three_pin_net_routing dispatch verbatim."""
        temp = list(node_gcells)
        temp.sort(key=lambda x: (x[1], x[0]))
        y1, x1 = temp[0]
        y2, x2 = temp[1]
        y3, x3 = temp[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._l_routing_dict(temp, weight, add)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col_idx in range(x1, x2):
                add(0, y1, col_idx, weight)
            for row_idx in range(y1, max(y2, y3)):
                add(1, row_idx, x2, weight)
        elif y2 == y3:
            # Verbatim from plc: separate H/V increments
            for col in range(x1, x2):
                add(0, y1, col, weight)
            for col in range(x2, x3):
                add(0, y2, col, weight)
            for row in range(min(y2, y1), max(y2, y1)):
                add(1, row, x2, weight)
        else:
            # plc passes temp_gcell (the (x,y)-sorted list) to __t_routing,
            # which then re-sorts by lex (y,x). Order-of-equal elements may
            # matter for tie-breaking, but tuple sort is deterministic.
            self._t_routing_dict(temp, weight, add)

    def _l_routing_dict(self, node_gcells, weight, add) -> None:
        node_gcells = sorted(node_gcells, key=lambda x: (x[1], x[0]))
        y1, x1 = node_gcells[0]
        y2, x2 = node_gcells[1]
        y3, x3 = node_gcells[2]
        for col in range(x1, x2):
            add(0, y1, col, weight)
        for col in range(x2, x3):
            add(0, y2, col, weight)
        for row in range(min(y1, y2), max(y1, y2)):
            add(1, row, x2, weight)
        for row in range(min(y2, y3), max(y2, y3)):
            add(1, row, x3, weight)

    def _t_routing_dict(self, node_gcells, weight, add) -> None:
        node_gcells = sorted(node_gcells)
        y1, x1 = node_gcells[0]
        y2, x2 = node_gcells[1]
        y3, x3 = node_gcells[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        for col in range(xmin, xmax):
            add(0, y2, col, weight)
        for row in range(min(y1, y2), max(y1, y2)):
            add(1, row, x1, weight)
        for row in range(min(y2, y3), max(y2, y3)):
            add(1, row, x3, weight)

    def _macro_cong_contrib(self, mod_x: float, mod_y: float, mod_w: float, mod_h: float) -> Dict[Tuple[int, int], float]:
        """Mirror PlacementCost.__macro_route_over_grid_cell with the
        macro_place.objective bounds-clamping monkey-patch on
        __get_grid_cell_location applied.

        Returns dict {(orient, cell_idx): unnormalized_value}, where orient=0 is
        H_macro_routing_cong, orient=1 is V_macro_routing_cong.
        """
        ur_x = mod_x + mod_w / 2
        ur_y = mod_y + mod_h / 2
        bl_x = mod_x - mod_w / 2
        bl_y = mod_y - mod_h / 2

        # Use clamped grid cell (same as patched plc).
        ur_row, ur_col = self._grid_cell(ur_x, ur_y)
        bl_row, bl_col = self._grid_cell(bl_x, bl_y)

        out: Dict[Tuple[int, int], float] = {}

        if_partial_v = False
        if_partial_h = False
        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                cell_x_min = c_i * self.grid_width
                cell_x_max = (c_i + 1) * self.grid_width
                cell_y_min = r_i * self.grid_height
                cell_y_max = (r_i + 1) * self.grid_height
                # __overlap_dist
                ox = min(cell_x_max, ur_x) - max(cell_x_min, bl_x)
                oy = min(cell_y_max, ur_y) - max(cell_y_min, bl_y)
                if ox <= 0 or oy <= 0:
                    continue
                if ur_row != bl_row:
                    if (r_i == bl_row and abs(oy - self.grid_height) > 1e-5) or (
                        r_i == ur_row and abs(oy - self.grid_height) > 1e-5
                    ):
                        if_partial_v = True
                if ur_col != bl_col:
                    if (c_i == bl_col and abs(ox - self.grid_width) > 1e-5) or (
                        c_i == ur_col and abs(ox - self.grid_width) > 1e-5
                    ):
                        if_partial_h = True
                # V_macro += x_dist * vrouting_alloc
                # H_macro += y_dist * hrouting_alloc
                out[(1, r_i * self.grid_col + c_i)] = (
                    out.get((1, r_i * self.grid_col + c_i), 0.0) + ox * self.vrouting_alloc
                )
                out[(0, r_i * self.grid_col + c_i)] = (
                    out.get((0, r_i * self.grid_col + c_i), 0.0) + oy * self.hrouting_alloc
                )
        # Partial-overlap correction at top edge
        if if_partial_v:
            for r_i in range(ur_row, ur_row + 1):
                for c_i in range(bl_col, ur_col + 1):
                    cell_x_min = c_i * self.grid_width
                    cell_x_max = (c_i + 1) * self.grid_width
                    cell_y_min = r_i * self.grid_height
                    cell_y_max = (r_i + 1) * self.grid_height
                    ox = min(cell_x_max, ur_x) - max(cell_x_min, bl_x)
                    oy = min(cell_y_max, ur_y) - max(cell_y_min, bl_y)
                    if ox <= 0 or oy <= 0:
                        continue
                    out[(1, r_i * self.grid_col + c_i)] = (
                        out.get((1, r_i * self.grid_col + c_i), 0.0) - ox * self.vrouting_alloc
                    )
        if if_partial_h:
            for r_i in range(bl_row, ur_row + 1):
                for c_i in range(ur_col, ur_col + 1):
                    cell_x_min = c_i * self.grid_width
                    cell_x_max = (c_i + 1) * self.grid_width
                    cell_y_min = r_i * self.grid_height
                    cell_y_max = (r_i + 1) * self.grid_height
                    ox = min(cell_x_max, ur_x) - max(cell_x_min, bl_x)
                    oy = min(cell_y_max, ur_y) - max(cell_y_min, bl_y)
                    if ox <= 0 or oy <= 0:
                        continue
                    out[(0, r_i * self.grid_col + c_i)] = (
                        out.get((0, r_i * self.grid_col + c_i), 0.0) - oy * self.hrouting_alloc
                    )
        return out

    def _init_congestion(self) -> None:
        """Compute per-net + per-hard-macro raw congestion contributions."""
        bench = self.benchmark
        # Raw (un-normalized, un-smoothed) net-routing congestion accumulators
        self.V_net_cong = torch.zeros(self.num_cells, dtype=torch.float64)
        self.H_net_cong = torch.zeros(self.num_cells, dtype=torch.float64)
        # Raw macro-routing congestion accumulators (already weighted by alloc)
        self.V_macro_cong = torch.zeros(self.num_cells, dtype=torch.float64)
        self.H_macro_cong = torch.zeros(self.num_cells, dtype=torch.float64)

        # Per-net contribution dict (cached for incremental updates)
        self.net_cong_contrib: List[Dict[Tuple[int, int], float]] = [None] * self.num_nets  # type: ignore
        for n in range(self.num_nets):
            contrib = self._net_cong_contrib(n)
            self.net_cong_contrib[n] = contrib
            for (orient, cell), w in contrib.items():
                if orient == 0:
                    self.H_net_cong[cell] += w
                else:
                    self.V_net_cong[cell] += w

        # Per-hard-macro congestion (only hard macros contribute)
        self.macro_cong_contrib: List[Dict[Tuple[int, int], float]] = [None] * bench.num_macros  # type: ignore
        for i in range(bench.num_macros):
            if not bool(self.is_hard_macro[i]):
                self.macro_cong_contrib[i] = {}
                continue
            x = float(self.placement[i, 0])
            y = float(self.placement[i, 1])
            w = float(self.macro_sizes[i, 0])
            h = float(self.macro_sizes[i, 1])
            contrib = self._macro_cong_contrib(x, y, w, h)
            self.macro_cong_contrib[i] = contrib
            for (orient, cell), val in contrib.items():
                if orient == 0:
                    self.H_macro_cong[cell] += val
                else:
                    self.V_macro_cong[cell] += val

    # ── Cost computation ───────────────────────────────────────────────────

    def _wirelength_cost(self) -> float:
        """Match plc.get_cost(): hpwl / ((W + H) * net_cnt)."""
        return float(self.total_hpwl / ((self.width + self.height) * self.net_cnt))

    def _density_cost_of(self, grid_occupied: torch.Tensor) -> float:
        """Density cost from an arbitrary grid_occupied tensor.

        Same logic as :meth:`_density_cost`, parameterized on the
        per-cell area tensor so it can be called with hypothetical state
        (e.g. by :meth:`delta_cost`).
        """
        cells = grid_occupied / self.grid_area
        density_cnt = int(math.floor(self.num_cells * 0.1))
        # Drop zeros (plc does `if gc != 0.0`).
        nonzero_mask = cells != 0.0
        occupied = cells[nonzero_mask]
        if self.num_cells < 10:
            return 0.5 * float(occupied.mean().item()) if len(occupied) > 0 else 0.0
        if density_cnt == 0:
            return 0.0
        # Take top-k of the *occupied* cells (k may exceed len(occupied)).
        k = min(density_cnt, occupied.numel())
        if k == 0:
            return 0.0
        topk, _ = torch.topk(occupied, k)
        return 0.5 * float(topk.sum().item() / density_cnt)

    def _density_cost(self) -> float:
        """Match plc.get_density_cost(): 0.5 * mean(top10% of nonzero cells)."""
        return self._density_cost_of(self.grid_occupied)

    def _smooth(self, raw: torch.Tensor, vertical: bool) -> torch.Tensor:
        """Mirror plc.__smooth_routing_cong on a [num_cells] tensor.

        Vectorized: each cell ``(r, c)`` distributes ``raw[r, c] / window_size``
        to its neighbors in the smoothing window. We pre-compute the window size
        per source cell, divide once, then perform a 1-D moving sum via
        ``cumsum`` along the smoothing axis.

        For V_routing_cong (vertical=True): smooth across cols, per row.
        For H_routing_cong (vertical=False): smooth across rows, per col.
        """
        sr = self.smooth_range
        gc = self.grid_col
        gr = self.grid_row
        flat = raw.reshape(gr, gc)
        if sr == 0:
            return flat.reshape(-1).clone()

        if vertical:
            # window size per source col (independent of row)
            cols = torch.arange(gc, dtype=torch.float64)
            lp = torch.clamp(cols - sr, min=0)
            rp = torch.clamp(cols + sr, max=gc - 1)
            window = rp - lp + 1  # [gc]
            divided = flat / window  # [gr, gc]
            # Each source cell (r, c) contributes `divided[r, c]` to
            # out[r, lp[c] : rp[c]+1]. Use 1-D moving sum: pad+cumsum along cols.
            # Build out by computing prefix sums then differences.
            # out[r, k] = sum over c with lp[c] <= k <= rp[c]
            # For c contributes: c in [k-sr, k+sr] (clamped). So
            # out[r, k] = sum_{c=max(0,k-sr)}^{min(gc-1,k+sr)} divided[r, c]
            # Use cumsum per row.
            ps = torch.zeros((gr, gc + 1), dtype=flat.dtype)
            ps[:, 1:] = torch.cumsum(divided, dim=1)
            ks = torch.arange(gc)
            lo = torch.clamp(ks - sr, min=0)
            hi = torch.clamp(ks + sr, max=gc - 1) + 1
            out = ps[:, hi] - ps[:, lo]
            return out.reshape(-1)
        else:
            rows = torch.arange(gr, dtype=torch.float64)
            lp = torch.clamp(rows - sr, min=0)
            up = torch.clamp(rows + sr, max=gr - 1)
            window = up - lp + 1  # [gr]
            divided = flat / window.unsqueeze(1)
            ps = torch.zeros((gr + 1, gc), dtype=flat.dtype)
            ps[1:] = torch.cumsum(divided, dim=0)
            ks = torch.arange(gr)
            lo = torch.clamp(ks - sr, min=0)
            hi = torch.clamp(ks + sr, max=gr - 1) + 1
            out = ps[hi] - ps[lo]
            return out.reshape(-1)

    def _congestion_cost_of(
        self,
        H_net_cong: torch.Tensor,
        V_net_cong: torch.Tensor,
        H_macro_cong: torch.Tensor,
        V_macro_cong: torch.Tensor,
    ) -> float:
        """Congestion cost from explicit raw congestion tensors.

        Same logic as :meth:`_congestion_cost`, parameterized so it can
        be called with hypothetical state (e.g. by :meth:`delta_cost`).
        """
        V_net_norm = V_net_cong / self.grid_v_routes
        H_net_norm = H_net_cong / self.grid_h_routes
        V_macro_norm = V_macro_cong / self.grid_v_routes
        H_macro_norm = H_macro_cong / self.grid_h_routes
        V_smoothed = self._smooth(V_net_norm, vertical=True)
        H_smoothed = self._smooth(H_net_norm, vertical=False)
        V_total = V_smoothed + V_macro_norm
        H_total = H_smoothed + H_macro_norm
        combined = torch.cat([V_total, H_total])
        cnt = int(math.floor(combined.numel() * 0.05))
        if cnt == 0:
            return float(combined.max().item())
        topk, _ = torch.topk(combined, cnt)
        return float(topk.sum().item() / cnt)

    def _congestion_cost(self) -> float:
        """Match plc.get_congestion_cost(): abu(V+H, 0.05) after normalize+smooth+macro-add."""
        return self._congestion_cost_of(
            self.H_net_cong, self.V_net_cong, self.H_macro_cong, self.V_macro_cong
        )

    def current_cost(self) -> Dict[str, float]:
        wl = self._wirelength_cost()
        density = self._density_cost()
        cong = self._congestion_cost()
        proxy = (
            self.weights["wirelength"] * wl
            + self.weights["density"] * density
            + self.weights["congestion"] * cong
        )
        return {"wl": wl, "density": density, "congestion": cong, "proxy": proxy}

    # ── Non-mutating cost peek ────────────────────────────────────────────

    def delta_cost(self, macro_idx: int, new_xy) -> Dict[str, float]:
        """Return the cost dict ``current_cost()`` would yield AFTER
        ``move(macro_idx, new_xy)``, without mutating evaluator state.

        Replacement for the ``(move → current_cost → revert)`` pattern in
        coordinate-descent search loops. Cheaper because:

        - No ``_MoveSnapshot`` is built (skips the dict copies of
          ``net_cong_contrib[n]`` for every affected net — usually the
          largest term in per-move overhead).
        - No revert pass (state is reconstructed in local clones rather
          than restored after the fact).

        On exit, every ``self.*`` field is bit-identical to entry. The
        method is reentrant within a single thread but is NOT thread-safe
        — it briefly mutates ``self.pin_x``/``self.pin_y`` for the moving
        macro's pins inside a ``try/finally`` so that
        ``_net_cong_contrib`` (which reads them) sees hypothetical
        positions.

        Returns the same ``{"wl", "density", "congestion", "proxy"}``
        dict shape as :meth:`current_cost`.
        """
        bench = self.benchmark
        nx, ny = float(new_xy[0]), float(new_xy[1])
        macro_pins = self.macro_to_pins[macro_idx]
        affected_nets = self.macro_to_nets[macro_idx].tolist()

        # Briefly retarget this macro's pins so _net_cong_contrib reads
        # hypothetical pin positions. Restored in `finally`.
        if len(macro_pins) > 0:
            saved_pin_x = self.pin_x[macro_pins].clone()
            saved_pin_y = self.pin_y[macro_pins].clone()
            self.pin_x[macro_pins] = nx + self.pin_offset_x[macro_pins]
            self.pin_y[macro_pins] = ny + self.pin_offset_y[macro_pins]
        else:
            saved_pin_x = None
            saved_pin_y = None

        try:
            # ── Hypothetical wirelength ──
            x_all, y_all = self.pin_x, self.pin_y  # now reflects hypothetical pins
            delta_hpwl = 0.0
            for n in affected_nets:
                pins = self.net_pins[n]
                xs = x_all[pins]
                ys = y_all[pins]
                mnx, mxx = float(xs.min()), float(xs.max())
                mny, mxy = float(ys.min()), float(ys.max())
                wnet = float(self.net_weight[n])
                new_hpwl_n = wnet * ((mxx - mnx) + (mxy - mny))
                delta_hpwl += new_hpwl_n - float(self.net_hpwl[n])
            new_total_hpwl = self.total_hpwl + delta_hpwl
            new_wl = new_total_hpwl / ((self.width + self.height) * self.net_cnt)

            # ── Hypothetical density: clone grid_occupied + apply macro delta ──
            w_size = float(self.macro_sizes[macro_idx, 0])
            h_size = float(self.macro_sizes[macro_idx, 1])
            old_macro_density = self.macro_density_contrib[macro_idx]
            new_macro_density = self._macro_cell_contrib(nx, ny, w_size, h_size)
            new_grid = self.grid_occupied.clone()
            for cell, area in old_macro_density.items():
                new_grid[cell] -= area
            for cell, area in new_macro_density.items():
                new_grid[cell] += area
            new_density = self._density_cost_of(new_grid)

            # ── Hypothetical congestion ──
            new_H_net = self.H_net_cong.clone()
            new_V_net = self.V_net_cong.clone()
            new_H_macro = self.H_macro_cong.clone()
            new_V_macro = self.V_macro_cong.clone()

            # Macro-routing delta (hard macros only contribute)
            if bool(self.is_hard_macro[macro_idx]):
                old_macro_cong = self.macro_cong_contrib[macro_idx]
                for (orient, cell), val in old_macro_cong.items():
                    if orient == 0:
                        new_H_macro[cell] -= val
                    else:
                        new_V_macro[cell] -= val
                new_macro_cong = self._macro_cong_contrib(nx, ny, w_size, h_size)
                for (orient, cell), val in new_macro_cong.items():
                    if orient == 0:
                        new_H_macro[cell] += val
                    else:
                        new_V_macro[cell] += val

            # Net-routing delta (uses hypothetical self.pin_x/pin_y)
            for n in affected_nets:
                old_contrib = self.net_cong_contrib[n]
                for (orient, cell), val in old_contrib.items():
                    if orient == 0:
                        new_H_net[cell] -= val
                    else:
                        new_V_net[cell] -= val
                new_contrib = self._net_cong_contrib(n)
                for (orient, cell), val in new_contrib.items():
                    if orient == 0:
                        new_H_net[cell] += val
                    else:
                        new_V_net[cell] += val

            new_cong = self._congestion_cost_of(
                new_H_net, new_V_net, new_H_macro, new_V_macro
            )

            new_proxy = (
                self.weights["wirelength"] * new_wl
                + self.weights["density"] * new_density
                + self.weights["congestion"] * new_cong
            )
            return {
                "wl": new_wl,
                "density": new_density,
                "congestion": new_cong,
                "proxy": new_proxy,
            }

        finally:
            # Restore pin positions
            if saved_pin_x is not None:
                self.pin_x[macro_pins] = saved_pin_x
                self.pin_y[macro_pins] = saved_pin_y

    # ── Move / revert ──────────────────────────────────────────────────────

    def move(self, macro_idx: int, new_xy) -> Dict[str, float]:
        """Move macro `macro_idx` to `new_xy` and return updated costs.

        Captures a snapshot for ``revert()``. Calling ``move`` again overwrites
        the snapshot, so revert is single-step only.
        """
        bench = self.benchmark
        if isinstance(new_xy, torch.Tensor):
            nx, ny = float(new_xy[0]), float(new_xy[1])
        else:
            nx, ny = float(new_xy[0]), float(new_xy[1])
        ox = float(self.placement[macro_idx, 0])
        oy = float(self.placement[macro_idx, 1])

        # ── Identify affected nets (any net touching a pin on this macro) ──
        affected_nets = self.macro_to_nets[macro_idx].tolist()

        # Snapshot
        snap = _MoveSnapshot(
            macro_idx=macro_idx,
            old_xy=(ox, oy),
            affected_net_indices=list(affected_nets),
            old_net_min_x=self.net_min_x[affected_nets].clone(),
            old_net_max_x=self.net_max_x[affected_nets].clone(),
            old_net_min_y=self.net_min_y[affected_nets].clone(),
            old_net_max_y=self.net_max_y[affected_nets].clone(),
            old_net_hpwl=self.net_hpwl[affected_nets].clone(),
            old_density_contrib=dict(self.macro_density_contrib[macro_idx]),
            old_net_cong_contrib={n: dict(self.net_cong_contrib[n]) for n in affected_nets},
            old_macro_cong_contrib=dict(self.macro_cong_contrib[macro_idx]),
        )
        self._snapshot = snap

        # ── Apply density delta (subtract old, add new) ──
        for cell, area in snap.old_density_contrib.items():
            self.grid_occupied[cell] -= area
        # ── Apply macro-routing delta (subtract old) ──
        for (orient, cell), val in snap.old_macro_cong_contrib.items():
            if orient == 0:
                self.H_macro_cong[cell] -= val
            else:
                self.V_macro_cong[cell] -= val
        # ── Apply per-net congestion delta (subtract old) ──
        for n, contrib in snap.old_net_cong_contrib.items():
            for (orient, cell), val in contrib.items():
                if orient == 0:
                    self.H_net_cong[cell] -= val
                else:
                    self.V_net_cong[cell] -= val
        # ── Subtract old per-net hpwl from total ──
        for i, n in enumerate(affected_nets):
            self.total_hpwl -= float(self.net_hpwl[n])

        # ── Update placement + cached pin positions ──
        self.placement[macro_idx, 0] = nx
        self.placement[macro_idx, 1] = ny
        self._update_macro_pin_positions(macro_idx)

        # ── Recompute density contribution for this macro ──
        w_size = float(self.macro_sizes[macro_idx, 0])
        h_size = float(self.macro_sizes[macro_idx, 1])
        new_density = self._macro_cell_contrib(nx, ny, w_size, h_size)
        self.macro_density_contrib[macro_idx] = new_density
        for cell, area in new_density.items():
            self.grid_occupied[cell] += area

        # ── Recompute macro-routing contribution (hard macros only) ──
        if bool(self.is_hard_macro[macro_idx]):
            new_macro_cong = self._macro_cong_contrib(nx, ny, w_size, h_size)
        else:
            new_macro_cong = {}
        self.macro_cong_contrib[macro_idx] = new_macro_cong
        for (orient, cell), val in new_macro_cong.items():
            if orient == 0:
                self.H_macro_cong[cell] += val
            else:
                self.V_macro_cong[cell] += val

        # ── Recompute affected nets: bbox + hpwl + congestion contrib ──
        # Get all pin positions (cheap; we only need affected nets' pins).
        x_all, y_all = self._all_pin_positions()
        for n in affected_nets:
            pins = self.net_pins[n]
            xs = x_all[pins]
            ys = y_all[pins]
            mnx, mxx = float(xs.min()), float(xs.max())
            mny, mxy = float(ys.min()), float(ys.max())
            self.net_min_x[n] = mnx
            self.net_max_x[n] = mxx
            self.net_min_y[n] = mny
            self.net_max_y[n] = mxy
            wnet = float(self.net_weight[n])
            self.net_hpwl[n] = wnet * ((mxx - mnx) + (mxy - mny))
            self.total_hpwl += float(self.net_hpwl[n])
            new_contrib = self._net_cong_contrib(n)
            self.net_cong_contrib[n] = new_contrib
            for (orient, cell), val in new_contrib.items():
                if orient == 0:
                    self.H_net_cong[cell] += val
                else:
                    self.V_net_cong[cell] += val

        return self.current_cost()

    def revert(self) -> Dict[str, float]:
        """Undo the most recent ``move``. Single-step undo only."""
        if self._snapshot is None:
            raise RuntimeError("No move to revert.")
        snap = self._snapshot
        bench = self.benchmark
        macro_idx = snap.macro_idx

        # ── Revert per-net congestion (subtract new, add old) ──
        for n in snap.affected_net_indices:
            cur = self.net_cong_contrib[n]
            for (orient, cell), val in cur.items():
                if orient == 0:
                    self.H_net_cong[cell] -= val
                else:
                    self.V_net_cong[cell] -= val
            old = snap.old_net_cong_contrib[n]
            self.net_cong_contrib[n] = old
            for (orient, cell), val in old.items():
                if orient == 0:
                    self.H_net_cong[cell] += val
                else:
                    self.V_net_cong[cell] += val

        # ── Revert macro routing ──
        cur_mc = self.macro_cong_contrib[macro_idx]
        for (orient, cell), val in cur_mc.items():
            if orient == 0:
                self.H_macro_cong[cell] -= val
            else:
                self.V_macro_cong[cell] -= val
        self.macro_cong_contrib[macro_idx] = snap.old_macro_cong_contrib
        for (orient, cell), val in snap.old_macro_cong_contrib.items():
            if orient == 0:
                self.H_macro_cong[cell] += val
            else:
                self.V_macro_cong[cell] += val

        # ── Revert density ──
        cur_d = self.macro_density_contrib[macro_idx]
        for cell, area in cur_d.items():
            self.grid_occupied[cell] -= area
        self.macro_density_contrib[macro_idx] = snap.old_density_contrib
        for cell, area in snap.old_density_contrib.items():
            self.grid_occupied[cell] += area

        # ── Revert hpwl + bboxes ──
        for i, n in enumerate(snap.affected_net_indices):
            self.total_hpwl -= float(self.net_hpwl[n])
            self.net_min_x[n] = snap.old_net_min_x[i]
            self.net_max_x[n] = snap.old_net_max_x[i]
            self.net_min_y[n] = snap.old_net_min_y[i]
            self.net_max_y[n] = snap.old_net_max_y[i]
            self.net_hpwl[n] = snap.old_net_hpwl[i]
            self.total_hpwl += float(self.net_hpwl[n])

        # ── Revert placement + cached pin positions ──
        self.placement[macro_idx, 0] = snap.old_xy[0]
        self.placement[macro_idx, 1] = snap.old_xy[1]
        self._update_macro_pin_positions(macro_idx)

        self._snapshot = None
        return self.current_cost()
