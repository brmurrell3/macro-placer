"""
Cluster bounds for polyhedra navigation — the "star properties" screener.

Analogy: NASA eliminates entire stellar systems from habitability searches
by looking at star properties alone (mass, luminosity, age). We eliminate
entire clusters of neighboring polyhedra by computing cheap invariants of
the center polyhedron's LP solution.

    Star property         →  What it screens
    ─────────────────────────────────────────
    Zobrist hash          →  duplicate topologies (ns)
    Displacement floor    →  moves requiring impossible position changes (μs)
    Net-span HPWL bound   →  moves that must worsen wirelength (μs)
    Axis crowding         →  moves that pack macros beyond canvas (μs)

Integration: called in Navigator.navigate() BEFORE projection + surrogate,
saving the O(cascade_repair + overlap_repair + surrogate_eval) cost per
rejected move.

Usage:
    screener = ClusterScreener(sizes, n_hard, canvas_w, canvas_h, macro_to_nets)
    screener.set_state(assignment, positions, duals)

    for move in proposed_moves:
        prune, reason = screener.screen(move.get_flips())
        if prune:
            continue  # skip expensive projection + surrogate
        # ... proceed with projection + surrogate evaluation

    print(screener.report())
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict

from assignment import L, R, B, A


class ClusterScreener:
    """Multi-tier screening for clusters of pair flips.

    Computes cheap bounds to reject moves before expensive projection
    and surrogate evaluation. Each tier is strictly cheaper than
    projection (~0.5ms) so screening is always net-positive.

    All tiers are O(k) or O(k × avg_nets_per_macro) where k = flip count
    (typically 1-5). Total screening cost: ~1-10μs per move.
    """

    def __init__(self, sizes: np.ndarray, n_hard: int,
                 canvas_w: float, canvas_h: float,
                 macro_to_nets: dict = None,
                 nets: list = None):
        """
        Args:
            sizes: [N, 2] macro sizes (width, height)
            n_hard: number of hard macros
            canvas_w, canvas_h: canvas dimensions
            macro_to_nets: {macro_idx: [net_idx, ...]} from surrogate
            nets: list of (macro_pins, fixed_pins) from surrogate
        """
        self.sizes = sizes
        self.n_hard = n_hard
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.macro_to_nets = macro_to_nets or {}
        self.nets = nets or []

        # Precompute net connectivity between macro pairs
        self._shared_net_cache = {}

        # Zobrist hash table (lazy-initialized)
        self._zobrist_rng = np.random.default_rng(98765)
        self._zobrist_table = {}
        self.seen_hashes = set()
        self.current_hash = 0

        # State (set via set_state)
        self.assignment = {}
        self.positions = None
        self.duals = {}

        # Stats
        self.stats = defaultdict(int)
        self.stats_saved_ms = 0.0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, assignment: dict, positions: np.ndarray,
                  duals: dict):
        """Update screener with current navigator state.

        Call once at navigate() start, and again after each accepted move.
        """
        self.assignment = assignment
        self.positions = positions
        self.duals = duals
        self.current_hash = self._compute_hash(assignment)
        self.seen_hashes.add(self.current_hash)

    def _compute_hash(self, assignment: dict) -> int:
        h = 0
        for pair, d in assignment.items():
            h ^= self._zobrist_val(pair, d)
        return h

    def _zobrist_val(self, pair: tuple, direction: int) -> int:
        key = (pair, direction)
        if key not in self._zobrist_table:
            self._zobrist_table[key] = int(
                self._zobrist_rng.integers(0, 2**63)
            )
        return self._zobrist_table[key]

    # ------------------------------------------------------------------
    # Shared net lookup (cached)
    # ------------------------------------------------------------------

    def _shared_nets(self, i: int, k: int) -> int:
        """Count nets shared between macros i and k."""
        key = (min(i, k), max(i, k))
        if key not in self._shared_net_cache:
            nets_i = set(self.macro_to_nets.get(i, []))
            nets_k = set(self.macro_to_nets.get(k, []))
            self._shared_net_cache[key] = len(nets_i & nets_k)
        return self._shared_net_cache[key]

    # ------------------------------------------------------------------
    # Main screening entry point
    # ------------------------------------------------------------------

    def screen(self, flips: list[tuple[tuple, int]]) -> tuple[bool, str]:
        """Screen a move's flips before projection.

        Args:
            flips: list of ((i, k), new_direction) — from Move objects.
                   For SingleFlipMove: [(pair, new_dir)]
                   For ClusterFlipMove: [(pair1, d1), (pair2, d2), ...]

        Returns:
            (should_prune: bool, reason: str)
        """
        self.stats['total'] += 1

        # Tier 0: Zobrist hash — O(k), nanoseconds
        new_hash = self.current_hash
        for pair, new_dir in flips:
            old_dir = self.assignment.get(pair)
            if old_dir is None:
                continue
            if old_dir == new_dir:
                self.stats['pruned_noop'] += 1
                return True, 'noop'
            new_hash ^= self._zobrist_val(pair, old_dir)
            new_hash ^= self._zobrist_val(pair, new_dir)
        if new_hash in self.seen_hashes:
            self.stats['pruned_dup'] += 1
            return True, 'duplicate'

        # Tier 1: Self-contradiction — O(k²), nanoseconds
        if len(flips) > 1:
            pair_dirs = {}
            for pair, new_dir in flips:
                if pair in pair_dirs and pair_dirs[pair] != new_dir:
                    self.stats['pruned_contradiction'] += 1
                    return True, 'contradiction'
                pair_dirs[pair] = new_dir

        # Tier 2: Displacement floor — O(k), microseconds
        #
        # For each flip, compute the minimum position displacement
        # required to satisfy the new constraint. Large displacement
        # means the move is fighting the current layout.
        total_disp_score = 0.0
        for pair, new_dir in flips:
            i, k = pair
            disp = self._displacement(i, k, new_dir)
            # Weight by net connectivity: macros connected to many nets
            # propagate displacement into HPWL more heavily
            net_weight = (len(self.macro_to_nets.get(i, []))
                          + len(self.macro_to_nets.get(k, [])))
            total_disp_score += disp * net_weight

        # Normalize by canvas scale
        canvas_diag = self.canvas_w + self.canvas_h
        norm_disp = total_disp_score / max(canvas_diag, 1.0)

        # Threshold: reject if displacement-weighted score is extreme.
        # Calibrated so ~20-40% of cluster moves are pruned (the obviously bad ones).
        # Single flips rarely hit this because displacement is small.
        if norm_disp > 50.0:
            self.stats['pruned_displacement'] += 1
            return True, f'displacement={norm_disp:.0f}'

        # Tier 3: Net-span HPWL lower bound — O(k × shared_nets), microseconds
        #
        # For flips that REVERSE a same-axis constraint (L↔R, B↔A),
        # macros sharing nets must see their shared-net bounding boxes grow.
        # This gives a provable HPWL floor.
        hpwl_floor = 0.0
        for pair, new_dir in flips:
            i, k = pair
            old_dir = self.assignment.get(pair)
            if old_dir is None:
                continue

            same_axis = (old_dir in (L, R) and new_dir in (L, R)) or \
                        (old_dir in (B, A) and new_dir in (B, A))
            if not same_axis:
                continue

            # Same-axis reversal: macros swap positions along that axis.
            # For every net containing both i and k, the net's span in
            # that axis increases by at least the swap distance.
            shared = self._shared_nets(i, k)
            if shared == 0:
                continue

            disp = self._displacement(i, k, new_dir)
            hpwl_floor += shared * disp

        # Normalize by WL normalization factor (rough: total_nets × canvas_diag)
        total_nets = len(self.nets) if self.nets else 1
        wl_norm = max(canvas_diag * total_nets, 1.0)
        norm_hpwl = hpwl_floor / wl_norm

        # HPWL floor > 3% of total proxy cost is almost certainly bad
        if norm_hpwl > 0.03:
            self.stats['pruned_hpwl'] += 1
            return True, f'hpwl_floor={norm_hpwl:.4f}'

        # Tier 4: Axis crowding — O(k), microseconds
        #
        # Check if flips force a set of macros into a chain whose total
        # size exceeds the canvas dimension. This catches cases where
        # cascade_repair would fail.
        x_chain_macros = set()
        y_chain_macros = set()
        for pair, new_dir in flips:
            i, k = pair
            if new_dir in (L, R):
                x_chain_macros.add(i)
                x_chain_macros.add(k)
            else:
                y_chain_macros.add(i)
                y_chain_macros.add(k)

        if x_chain_macros:
            total_w = sum(float(self.sizes[m, 0]) for m in x_chain_macros)
            if total_w > self.canvas_w * 0.90:
                self.stats['pruned_crowding'] += 1
                return True, f'x-crowd={total_w:.0f}>{self.canvas_w:.0f}'

        if y_chain_macros:
            total_h = sum(float(self.sizes[m, 1]) for m in y_chain_macros)
            if total_h > self.canvas_h * 0.90:
                self.stats['pruned_crowding'] += 1
                return True, f'y-crowd={total_h:.0f}>{self.canvas_h:.0f}'

        # Register the new hash so we don't evaluate this topology again
        self.seen_hashes.add(new_hash)

        self.stats['passed'] += 1
        return False, 'passed'

    # ------------------------------------------------------------------
    # Displacement computation
    # ------------------------------------------------------------------

    def _displacement(self, i: int, k: int, new_dir: int) -> float:
        """Minimum position displacement to satisfy new_dir for pair (i,k).

        Returns the violation amount: how far macros must move to satisfy
        the new constraint. Zero if already satisfied.
        """
        if self.positions is None:
            return 0.0

        wi, hi = float(self.sizes[i, 0]), float(self.sizes[i, 1])
        wk, hk = float(self.sizes[k, 0]), float(self.sizes[k, 1])

        if new_dir == L:
            # Need x_k - x_i >= (wi+wk)/2
            req = (wi + wk) / 2
            actual = self.positions[k, 0] - self.positions[i, 0]
        elif new_dir == R:
            # Need x_i - x_k >= (wi+wk)/2
            req = (wi + wk) / 2
            actual = self.positions[i, 0] - self.positions[k, 0]
        elif new_dir == B:
            # Need y_k - y_i >= (hi+hk)/2
            req = (hi + hk) / 2
            actual = self.positions[k, 1] - self.positions[i, 1]
        elif new_dir == A:
            # Need y_i - y_k >= (hi+hk)/2
            req = (hi + hk) / 2
            actual = self.positions[i, 1] - self.positions[k, 1]
        else:
            return 0.0

        return max(0.0, req - actual)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a summary of screening statistics."""
        total = self.stats['total']
        if total == 0:
            return "ClusterScreener: no moves screened"
        passed = self.stats['passed']
        pruned = total - passed
        pct = 100 * pruned / total

        parts = [f"Screened {total}: {pruned} pruned ({pct:.0f}%)"]
        for key in ['pruned_noop', 'pruned_dup', 'pruned_contradiction',
                     'pruned_displacement', 'pruned_hpwl', 'pruned_crowding']:
            val = self.stats.get(key, 0)
            if val > 0:
                label = key.replace('pruned_', '')
                parts.append(f"{label}={val}")
        return ', '.join(parts)
