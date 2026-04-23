"""
Navigation search loop for polyhedra placement.

The Navigator orchestrates: propose moves → project → evaluate → accept/reject.
The acceptance policy is pluggable via AcceptanceStrategy.
Move generation is pluggable via MoveProposer (see moves.py).
"""

from __future__ import annotations

import math
import time
import numpy as np
import torch
from collections import defaultdict
from typing import Protocol, runtime_checkable

from assignment import extract_assignment, DIR_NAMES
from projection import check_overlaps
from moves import Move, MoveProposer, DualGuidedProposer, ClusterProposer
from surrogate import Surrogate
from cluster_bounds import ClusterScreener

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Acceptance strategies
# ---------------------------------------------------------------------------

@runtime_checkable
class AcceptanceStrategy(Protocol):
    """Decides whether to accept a move given the cost delta."""

    def accept(self, delta: float, elapsed: float, time_budget: float) -> bool: ...
    def update(self, delta: float): ...


class GreedyAcceptance:
    """Only accept strictly improving moves."""

    def accept(self, delta: float, elapsed: float, time_budget: float) -> bool:
        return delta < 0

    def update(self, delta: float):
        pass


class SimulatedAnnealingAcceptance:
    """Accept worse moves with Boltzmann probability, temperature cools linearly.

    Auto-calibrates T0 from the first few iterations' delta distribution.
    """

    def __init__(self, rng: np.random.Generator = None, min_t0: float = 5e-3):
        self.rng = rng or np.random.default_rng(42)
        self.t0 = 0.0
        self.min_t0 = min_t0
        self.calibration_deltas = []
        self.calibrated = False

    def accept(self, delta: float, elapsed: float, time_budget: float) -> bool:
        if delta < 0:
            return True
        temperature = self.t0 * max(0.0, 1.0 - elapsed / time_budget)
        if temperature < 1e-12:
            return False
        return self.rng.random() < math.exp(-delta / temperature)

    def update(self, delta: float):
        """Feed observed deltas for calibration."""
        if self.calibrated:
            return
        if delta > 1e-8:
            self.calibration_deltas.append(delta)
        if len(self.calibration_deltas) >= 10 or (
            len(self.calibration_deltas) >= 3
        ):
            median_delta = float(np.median(self.calibration_deltas))
            raw_t0 = median_delta / 0.693 if median_delta > 1e-8 else 0.01
            self.t0 = max(raw_t0, self.min_t0)
            self.calibrated = True

    @property
    def temperature_at(self):
        """Return a function that computes T at a given (elapsed, budget)."""
        def _t(elapsed, budget):
            return self.t0 * max(0.0, 1.0 - elapsed / budget)
        return _t


# ---------------------------------------------------------------------------
# Navigator
# ---------------------------------------------------------------------------

class Navigator:
    """
    Search loop: propose moves, evaluate via surrogate, accept/reject.

    Pluggable components:
    - move_proposers: list of MoveProposers (DualGuidedProposer, ClusterProposer, ...)
    - acceptance: AcceptanceStrategy (greedy, SA, ...)
    - surrogate: Surrogate for fast cost estimation
    """

    def __init__(self, lp_solver, benchmark: Benchmark, plc,
                 surrogate: Surrogate = None,
                 move_proposers: list[MoveProposer] = None,
                 acceptance: AcceptanceStrategy = None):
        self.lp = lp_solver
        self.benchmark = benchmark
        self.plc = plc
        self.surrogate = surrogate

        self.move_proposers = move_proposers or [
            DualGuidedProposer(alpha=1.0, top_k=50),
            ClusterProposer(n_proposals=20),
        ]
        self.acceptance = acceptance or SimulatedAnnealingAcceptance()

        self.rng = np.random.default_rng(42)
        self._macro_to_pairs = None

        self.screener = ClusterScreener(
            sizes=benchmark.macro_sizes.numpy(),
            n_hard=benchmark.num_hard_macros,
            canvas_w=benchmark.canvas_width,
            canvas_h=benchmark.canvas_height,
            macro_to_nets=(surrogate.macro_to_nets if surrogate else {}),
            nets=(surrogate.nets if surrogate else []),
        )

    def _get_macro_to_pairs(self, assignment):
        """Build macro -> list of pairs index."""
        if self._macro_to_pairs is not None:
            return self._macro_to_pairs
        m2p = defaultdict(list)
        for (a, b) in assignment:
            m2p[a].append((a, b))
            m2p[b].append((a, b))
        self._macro_to_pairs = m2p
        return m2p

    @staticmethod
    def _extract_flips(move, assignment):
        """Extract (pair, new_dir) list from a Move for screening."""
        from moves import SingleFlipMove, ClusterFlipMove
        if isinstance(move, SingleFlipMove):
            return [(move.pair, move.new_dir)]
        elif isinstance(move, ClusterFlipMove):
            return move.flips
        return []

    def navigate(self, assignment: dict, lp_result: dict,
                 ref_positions: np.ndarray,
                 max_iters: int = 500, time_budget: float = 300.0,
                 top_k_verify: int = 5, verbose: bool = True,
                 lp_resolve_cap: int = 20) -> dict:
        """
        Main search loop.

        Returns dict with: assignment, positions, proxy_cost, improvements, evaluations.
        """
        t0 = time.time()

        sizes = self.benchmark.macro_sizes.numpy()
        fixed_mask = self.benchmark.macro_fixed.numpy()
        n_hard = self.benchmark.num_hard_macros
        canvas_w = self.benchmark.canvas_width
        canvas_h = self.benchmark.canvas_height
        movable_hard = np.arange(n_hard)[~fixed_mask[:n_hard]]

        best_assignment = dict(assignment)
        best_positions = ref_positions.copy()
        current_assignment = dict(assignment)
        current_positions = ref_positions.copy()

        hpwl_result = lp_result

        # Initialize surrogate
        if self.surrogate is not None:
            self.surrogate.init_from_placement(current_positions)
            best_proxy = self.surrogate.get_proxy_cost()
            if verbose:
                print(f"  Surrogate init: proxy={best_proxy:.4f} "
                      f"(wl={self.surrogate.get_wirelength_cost():.4f}, "
                      f"den={self.surrogate.get_density_cost():.4f}, "
                      f"cong={self.surrogate.get_congestion_cost():.4f})")
        else:
            from macro_place.objective import compute_proxy_cost
            placement = torch.tensor(ref_positions, dtype=torch.float32)
            best_proxy = compute_proxy_cost(placement, self.benchmark, self.plc)["proxy_cost"]

        current_proxy = best_proxy

        improvements = 0
        accepted_count = 0
        surrogate_evals = 0
        best_stale_iters = 0
        lp_resolves = 0

        self.screener.set_state(current_assignment, current_positions,
                                hpwl_result["duals"])

        if verbose:
            print(f"  Starting navigation: proxy={best_proxy:.4f}, "
                  f"acceptance={type(self.acceptance).__name__}")

        for iteration in range(max_iters):
            elapsed = time.time() - t0
            if elapsed > time_budget:
                if verbose:
                    print(f"  Time budget exhausted at iter {iteration}")
                break

            if best_stale_iters > 30:
                if verbose:
                    print(f"  Best stale for {best_stale_iters} iters, stopping")
                break

            macro_to_pairs = self._get_macro_to_pairs(current_assignment)

            # --- Propose moves from all proposers ---
            all_moves: list[Move] = []
            m2n = self.surrogate.macro_to_nets if self.surrogate else {}
            for proposer in self.move_proposers:
                moves = proposer.propose(
                    hpwl_result["duals"], current_assignment,
                    macro_to_nets=m2n, rng=self.rng,
                )
                all_moves.extend(moves)

            if not all_moves:
                best_stale_iters += 1
                continue

            # --- Project and evaluate all moves with surrogate ---
            projected = []  # (surr_proxy, new_pos, move, moved_set)

            for move in all_moves:
                if time.time() - t0 > time_budget:
                    break

                # --- Screen before expensive projection ---
                flips = self._extract_flips(move, current_assignment)
                if flips:
                    prune, reason = self.screener.screen(flips)
                    if prune:
                        continue

                result = move.project(
                    current_positions, current_assignment, sizes, fixed_mask,
                    n_hard, canvas_w, canvas_h, macro_to_pairs
                )
                if result is None:
                    continue

                new_pos, moved_set = result

                if self.surrogate is not None:
                    surr_proxy = self.surrogate.evaluate_move(list(moved_set), new_pos)
                    surrogate_evals += 1
                else:
                    surr_proxy = current_proxy - 0.001  # assume slight improvement

                # Feed delta to acceptance for calibration
                delta = surr_proxy - current_proxy
                self.acceptance.update(delta)

                projected.append((surr_proxy, new_pos, move, moved_set))

            if not projected:
                best_stale_iters += 1
                continue

            # --- Sort by surrogate, verify top-k ---
            projected.sort(key=lambda x: x[0])
            verify_list = projected[:top_k_verify]

            accepted_this_iter = False
            improved_best = False

            for surr_proxy, new_pos, move, moved_set in verify_list:
                if time.time() - t0 > time_budget:
                    break

                # Fast overlap check
                if check_overlaps(new_pos, sizes, n_hard):
                    continue

                accepted_count += 1

                # Acceptance decision
                delta = surr_proxy - current_proxy
                elapsed_now = time.time() - t0
                if not self.acceptance.accept(delta, elapsed_now, time_budget):
                    continue

                # Accept — update current state
                current_proxy = surr_proxy
                current_positions = new_pos.copy()

                # Re-extract assignment from actual positions
                current_assignment = extract_assignment(
                    current_positions, sizes, movable_hard
                )
                self._macro_to_pairs = None

                # Update surrogate
                if self.surrogate is not None:
                    self.surrogate.init_from_placement(current_positions)

                # Update screener state
                self.screener.set_state(
                    current_assignment, current_positions,
                    hpwl_result["duals"]
                )

                accepted_this_iter = True

                # Track global best
                if surr_proxy < best_proxy:
                    improvement = best_proxy - surr_proxy
                    best_proxy = surr_proxy
                    best_positions = new_pos.copy()
                    best_assignment = dict(current_assignment)
                    improvements += 1
                    improved_best = True
                    best_stale_iters = 0

                    if verbose:
                        sa_info = ""
                        if hasattr(self.acceptance, 'temperature_at'):
                            T = self.acceptance.temperature_at(elapsed_now, time_budget)
                            sa_info = f", T={T:.6f}"
                        print(f"  iter {iteration}: best={best_proxy:.4f} "
                              f"(delta={-improvement:.4f}, {move.describe()}"
                              f"{sa_info}, {elapsed_now:.1f}s)")

                # Refresh duals periodically
                remaining = time_budget - (time.time() - t0)
                if (improvements > 0 and improvements % 5 == 0
                        and lp_resolves < lp_resolve_cap and remaining > 15):
                    hpwl_result = self.lp.solve(
                        current_assignment, time_limit=min(10.0, remaining - 5),
                        ref_positions=current_positions, sparse_margin=10.0
                    )
                    lp_resolves += 1
                break

            if not improved_best:
                best_stale_iters += 1

            if not accepted_this_iter:
                remaining = time_budget - (time.time() - t0)
                if (best_stale_iters % 15 == 0 and best_stale_iters > 0
                        and lp_resolves < lp_resolve_cap and remaining > 15):
                    hpwl_result = self.lp.solve(
                        current_assignment, time_limit=min(10.0, remaining - 5),
                        ref_positions=current_positions, sparse_margin=10.0
                    )
                    lp_resolves += 1

        if verbose:
            elapsed = time.time() - t0
            print(f"  Navigation done: {accepted_count} accepted, "
                  f"{surrogate_evals} surrogate evals, "
                  f"{improvements} improvements, {lp_resolves} LP resolves, "
                  f"{elapsed:.1f}s")
            print(f"  Screener: {self.screener.report()}")

        return {
            "assignment": best_assignment,
            "positions": best_positions,
            "proxy_cost": best_proxy,
            "improvements": improvements,
            "evaluations": accepted_count,
            "surrogate_evals": surrogate_evals,
        }
