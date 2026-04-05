"""
Move types for polyhedra navigation.

Each move proposes a change to the assignment (topology) and projects it
to get new positions. The Navigator doesn't need to know the move internals —
it just calls propose() and project().

To add a new move type:
  1. Implement the MoveProposer protocol (generates candidates)
  2. Each candidate is a Move (can be projected to get new positions)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from typing import Protocol, runtime_checkable

from assignment import L, R, B, A, OPPOSITE
from projection import (
    push_apart, clamp_to_canvas, cascade_repair, overlap_repair
)


# ---------------------------------------------------------------------------
# Move protocol
# ---------------------------------------------------------------------------

class Move:
    """A proposed change to the assignment that can be projected to positions."""

    def describe(self) -> str:
        """Human-readable description for logging."""
        raise NotImplementedError

    def apply_to_assignment(self, assignment: dict) -> dict:
        """Return a new assignment with this move applied."""
        raise NotImplementedError

    def project(self, positions: np.ndarray, assignment: dict,
                sizes: np.ndarray, fixed_mask: np.ndarray,
                n_hard: int, canvas_w: float, canvas_h: float,
                macro_to_pairs: dict) -> tuple[np.ndarray, set] | None:
        """Project this move onto positions, returning (new_pos, moved_set) or None."""
        raise NotImplementedError


class SingleFlipMove(Move):
    """Flip a single pair's direction."""

    def __init__(self, pair: tuple, new_dir: int, old_dir: int, dual_mag: float = 0.0):
        self.pair = pair
        self.new_dir = new_dir
        self.old_dir = old_dir
        self.dual_mag = dual_mag

    def describe(self) -> str:
        from assignment import DIR_NAMES
        return (f"pair={self.pair}, "
                f"{DIR_NAMES[self.old_dir]}->{DIR_NAMES[self.new_dir]}")

    def apply_to_assignment(self, assignment: dict) -> dict:
        new_assign = dict(assignment)
        new_assign[self.pair] = self.new_dir
        return new_assign

    def project(self, positions, assignment, sizes, fixed_mask,
                n_hard, canvas_w, canvas_h, macro_to_pairs):
        i, k = self.pair
        pos = positions.copy()
        movable_i = i < n_hard and not fixed_mask[i]
        movable_k = k < n_hard and not fixed_mask[k]

        # Apply the new separation constraint
        success = push_apart(pos, sizes, i, k, self.new_dir,
                             movable_i, movable_k)
        if not success:
            return None

        clamp_to_canvas(pos, sizes, [i, k], canvas_w, canvas_h)

        # Temporarily apply the flip in assignment for cascade
        old_dir = assignment.get(self.pair)
        assignment[self.pair] = self.new_dir

        moved = {i, k}
        cascade_repair(pos, sizes, assignment, macro_to_pairs,
                       moved, fixed_mask, n_hard, canvas_w, canvas_h)
        overlap_repair(pos, sizes, moved, fixed_mask, n_hard,
                       canvas_w, canvas_h)

        # Restore assignment
        if old_dir is not None:
            assignment[self.pair] = old_dir

        return pos, moved


class ClusterFlipMove(Move):
    """Flip multiple pairs simultaneously."""

    def __init__(self, flips: list[tuple[tuple, int]]):
        """flips: list of (pair, new_direction) tuples."""
        self.flips = flips

    def describe(self) -> str:
        return f"cluster({len(self.flips)} pairs)"

    def apply_to_assignment(self, assignment: dict) -> dict:
        new_assign = dict(assignment)
        for pair, new_dir in self.flips:
            new_assign[pair] = new_dir
        return new_assign

    def project(self, positions, assignment, sizes, fixed_mask,
                n_hard, canvas_w, canvas_h, macro_to_pairs):
        pos = positions.copy()
        # Save old dirs for restoration
        old_dirs = {}
        for pair, new_dir in self.flips:
            old_dirs[pair] = assignment.get(pair)
            assignment[pair] = new_dir

        # Sequential projection for each flip
        all_moved = set()
        for pair, new_dir in self.flips:
            i, k = pair
            movable_i = i < n_hard and not fixed_mask[i]
            movable_k = k < n_hard and not fixed_mask[k]

            success = push_apart(pos, sizes, i, k, new_dir,
                                 movable_i, movable_k)
            if not success:
                # Restore assignment
                for p, od in old_dirs.items():
                    if od is not None:
                        assignment[p] = od
                return None

            clamp_to_canvas(pos, sizes, [i, k], canvas_w, canvas_h)
            all_moved.add(i)
            all_moved.add(k)

        # Cascade repair after all flips
        cascade_repair(pos, sizes, assignment, macro_to_pairs,
                       all_moved, fixed_mask, n_hard, canvas_w, canvas_h)
        overlap_repair(pos, sizes, all_moved, fixed_mask, n_hard,
                       canvas_w, canvas_h)

        # Restore assignment
        for p, od in old_dirs.items():
            if od is not None:
                assignment[p] = od

        return pos, all_moved


# ---------------------------------------------------------------------------
# MoveProposer protocol and implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class MoveProposer(Protocol):
    """Generates candidate moves for the navigator to evaluate."""

    def propose(self, duals: dict, assignment: dict, **kwargs) -> list[Move]:
        """Return a list of candidate moves, roughly ordered by promise."""
        ...


class DualGuidedProposer:
    """Proposes single-pair flips ranked by LP dual variable magnitude.

    Large dual = expensive constraint = most promising to flip.
    """

    def __init__(self, alpha: float = 1.0, top_k: int = 50):
        self.alpha = alpha
        self.top_k = top_k

    def propose(self, duals: dict, assignment: dict, **kwargs) -> list[Move]:
        candidates = []
        for (i, k), dual_val in duals.items():
            mag = abs(dual_val)
            if mag < 1e-10:
                continue
            current_dir = assignment[(i, k)]
            for new_dir in range(4):
                if new_dir != current_dir:
                    candidates.append(SingleFlipMove(
                        (i, k), new_dir, current_dir, mag
                    ))

        candidates.sort(key=lambda m: -m.dual_mag)
        return candidates[:self.top_k * 3]


class ClusterProposer:
    """Proposes multi-pair cluster flips using three strategies:

    1. Net-correlated flips: flip pairs sharing high-fanout nets
    2. Macro-centered flips: flip all pairs involving a high-dual macro
    3. Connected component flips: BFS on constraint graph
    """

    def __init__(self, n_proposals: int = 20):
        self.n_proposals = n_proposals

    def propose(self, duals: dict, assignment: dict,
                macro_to_nets: dict = None,
                rng: np.random.Generator = None, **kwargs) -> list[Move]:
        if rng is None:
            rng = np.random.default_rng()
        if macro_to_nets is None:
            macro_to_nets = {}

        dual_pairs = [(pair, abs(d)) for pair, d in duals.items() if abs(d) > 1e-10]
        if not dual_pairs:
            return []
        dual_pairs.sort(key=lambda x: -x[1])

        clusters = []

        # Strategy 1: Net-correlated flips
        if macro_to_nets:
            for seed_pair, seed_mag in dual_pairs[:min(10, len(dual_pairs))]:
                si, sk = seed_pair
                shared_nets = set(macro_to_nets.get(si, [])) | set(macro_to_nets.get(sk, []))

                cluster = [(seed_pair, rng.choice([d for d in range(4)
                            if d != assignment[seed_pair]]))]

                for other_pair, other_mag in dual_pairs:
                    if other_pair == seed_pair:
                        continue
                    oi, ok = other_pair
                    if (set(macro_to_nets.get(oi, [])) | set(macro_to_nets.get(ok, []))) & shared_nets:
                        new_dir = rng.choice([d for d in range(4)
                                              if d != assignment[other_pair]])
                        cluster.append((other_pair, new_dir))
                        if len(cluster) >= 4:
                            break

                if len(cluster) >= 2:
                    clusters.append(cluster)

        # Strategy 2: Macro-centered flips
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

        # Strategy 3: Connected component flips
        top_pairs = dual_pairs[:min(100, len(dual_pairs))]
        macro_to_high_pairs = defaultdict(list)
        top_pair_set = set()
        for pair, mag in top_pairs:
            top_pair_set.add(pair)
            pi, pk = pair
            macro_to_high_pairs[pi].append(pair)
            macro_to_high_pairs[pk].append(pair)

        visited = set()
        for seed_pair, _ in top_pairs[:20]:
            if seed_pair in visited:
                continue
            component = []
            queue = deque([seed_pair])
            while queue and len(component) < 30:
                p = queue.popleft()
                if p in visited or p not in top_pair_set:
                    continue
                visited.add(p)
                component.append(p)
                pi, pk = p
                for neighbor in macro_to_high_pairs.get(pi, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
                for neighbor in macro_to_high_pairs.get(pk, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if 5 <= len(component) <= 30:
                cluster = [(p, OPPOSITE[assignment[p]]) for p in component]
                clusters.append(cluster)

        # Deduplicate and convert to ClusterFlipMove
        seen = set()
        moves = []
        for c in clusters:
            key = tuple(sorted((p, d) for p, d in c))
            if key not in seen:
                seen.add(key)
                moves.append(ClusterFlipMove(c))
            if len(moves) >= self.n_proposals:
                break

        return moves
