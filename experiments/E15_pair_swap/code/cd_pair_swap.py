"""E15 — CDAdaptive + 2-macro pair swap polish phase.

Hypothesis: CD plateaus at the joint per-axis fixed point. Single-macro moves
(both CD and grid-bin LNS) can't change the topology. Swapping the positions
of two strongly-coupled macros (sharing nets) is a coordinated joint move
explicitly outside CD's reachable set — could find escapes LNS misses.

Algorithm (per benchmark):
  1. Run CDAdaptive (E16 settings: threshold=0.001, cap=3000s)
  2. Pair-swap phase: until budget exhausted OR no-improvement pass:
       a. Build candidate pairs: macros that share ≥ min_shared_nets nets,
          ranked by (net adjacency × distance) — close, well-connected
          pairs are unlikely to escape via swap; far, well-connected pairs
          are the ones where swap is most likely to reduce HPWL
       b. Take top-K pairs (K = adaptive cap, % of total pairs)
       c. For each pair (a, b):
            - Check legality: A at B's pos, B at A's pos (excluding each other)
            - 2-step move: A → pos_b, then B → pos_a (intermediate state has
              overlap which the evaluator handles transparently)
            - If proxy improves, commit. Else: manual revert via 2 inverse moves.

All hyperparameters global. No per-benchmark tuning. Note: incremental
evaluator's revert() is single-step only, so we revert manually.

Ablation expectation: if no swap improves the converged baseline, swap-style
moves are dead and we should focus on LNS variants instead.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import torch

# This file lives at experiments/E15_pair_swap/code/, so repo root is 4 levels up.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from macro_place.bench_paths import find_benchmark_dir
from submissions.cd_adaptive.placer import run_cd_adaptive


_DIAGNOSTIC_PATH = _ROOT / "scripts" / "cd_ibm10_diagnostic.py"


def _import_diagnostic():
    if "cd_ibm10_diagnostic" in sys.modules:
        return sys.modules["cd_ibm10_diagnostic"]
    spec = importlib.util.spec_from_file_location(
        "cd_ibm10_diagnostic", str(_DIAGNOSTIC_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cd_ibm10_diagnostic"] = mod
    spec.loader.exec_module(mod)
    return mod


_diag = _import_diagnostic()
sdf_init = _diag.sdf_init
project_overlaps = _diag.project_overlaps


# ── Pair selection ──────────────────────────────────────────────────────────


def _build_candidate_pairs(
    benchmark: Benchmark,
    plc,
    placement: torch.Tensor,
    n_hard: int,
    fixed: np.ndarray,
    min_shared_nets: int = 1,
    max_pairs: int = 5000,
) -> List[Tuple[int, int]]:
    """Enumerate candidate macro pairs ranked by (shared_nets × distance).

    Net connectivity is read from ``plc.nets`` (the TILOS PlacementCost
    object) because ``benchmark.net_nodes`` is empty after the loader (a
    known limitation — see GitHub issue #70 in the contest repo). We
    extract macro→macro adjacency through pin parent names, the same way
    ``macro_place/sdf_init.py:_extract_edges`` does.

    A pair (a, b) is a candidate if both are hard movable macros that share
    at least ``min_shared_nets`` nets. We prioritize pairs that are *far
    apart* in the current placement among well-connected pairs — those are
    where a swap could most reduce HPWL.
    """
    # Build PlacementCost-module-name → benchmark hard-macro index map
    name_to_bidx: dict = {}
    for bidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        if bidx >= n_hard:
            break
        if not bool(fixed[bidx]):
            name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx

    pair_counts: dict = {}
    # plc.nets: dict driver_pin_name -> [sink_pin_names]. Pin names are
    # "module_name/pin_name" — module name is the parent.
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + list(sinks):
            parent = pin.split("/")[0]
            bidx = name_to_bidx.get(parent)
            if bidx is not None:
                macros.add(bidx)
        if len(macros) < 2 or len(macros) > 50:
            # Skip nets touching too many movables (clock/power-like).
            continue
        ml = sorted(macros)
        for i in range(len(ml)):
            for j in range(i + 1, len(ml)):
                pair_counts[(ml[i], ml[j])] = pair_counts.get((ml[i], ml[j]), 0) + 1

    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    scored: List[Tuple[float, int, int]] = []
    for (a, b), cnt in pair_counts.items():
        if cnt < min_shared_nets:
            continue
        d = math.hypot(pos[a, 0] - pos[b, 0], pos[a, 1] - pos[b, 1])
        score = cnt * d
        scored.append((score, a, b))

    scored.sort(key=lambda t: -t[0])
    return [(a, b) for _, a, b in scored[:max_pairs]]


def _is_legal_2d_excluding(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    exclude: Set[int],
    eps: float = 1e-4,
) -> bool:
    """True iff placing macro `idx` at (x, y) doesn't overlap any hard macro
    (other than members of `exclude`) at its current position."""
    if idx >= n_hard:
        return True
    half_w = float(macro_sizes_np[idx, 0]) / 2.0
    half_h = float(macro_sizes_np[idx, 1]) / 2.0
    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]
    dx = np.abs(pos[:, 0] - x)
    dy = np.abs(pos[:, 1] - y)
    min_dx = half_w + sz[:, 0] / 2.0
    min_dy = half_h + sz[:, 1] / 2.0
    blockers = (dx < min_dx - eps) & (dy < min_dy - eps)
    for j in exclude:
        if 0 <= j < n_hard:
            blockers[j] = False
    blockers[idx] = False
    return not bool(np.any(blockers))


# ── Swap primitive ──────────────────────────────────────────────────────────


def _try_swap(
    evaluator: IncrementalProxyEvaluator,
    a: int,
    b: int,
    macro_sizes_np: np.ndarray,
    n_hard: int,
) -> float:
    """Attempt to swap positions of macros a and b. If proxy improves, commit
    (state stays at swapped positions). Else manually revert. Returns the
    proxy delta (negative = improvement, 0.0 if rejected/illegal)."""
    cur_cost = evaluator.current_cost()["proxy"]
    pos_a = (
        float(evaluator.placement[a, 0]),
        float(evaluator.placement[a, 1]),
    )
    pos_b = (
        float(evaluator.placement[b, 0]),
        float(evaluator.placement[b, 1]),
    )

    # Legality: A at B's old position (excluding A and B from blockers),
    # B at A's old position (same).
    if not _is_legal_2d_excluding(a, pos_b[0], pos_b[1], evaluator.placement,
                                   macro_sizes_np, n_hard, exclude={a, b}):
        return 0.0
    if not _is_legal_2d_excluding(b, pos_a[0], pos_a[1], evaluator.placement,
                                   macro_sizes_np, n_hard, exclude={a, b}):
        return 0.0

    # Move A → pos_b. Intermediate state has overlap (A and B both at pos_b)
    # but the evaluator just tracks per-net min/max; no constraint check.
    evaluator.move(a, pos_b)
    # Move B → pos_a (old). Now state is: A at pos_b, B at pos_a.
    evaluator.move(b, pos_a)

    new_cost = evaluator.current_cost()["proxy"]

    if new_cost < cur_cost - 1e-9:
        # Accept: leave evaluator at the swapped state.
        return new_cost - cur_cost

    # Reject: manual revert via two inverse moves.
    evaluator.move(b, pos_b)  # B back to its old position
    evaluator.move(a, pos_a)  # A back to its old position
    return 0.0


# ── Main loop ────────────────────────────────────────────────────────────────


def run_pair_swap(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    n_hard: int,
    fixed: np.ndarray,
    time_budget_s: float,
    min_shared_nets: int = 1,
    max_pairs: int = 5000,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Pair-swap phase. Iterates passes through candidate pairs until budget
    exhausted or a full pass produces no improvement."""
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    pairs = _build_candidate_pairs(
        benchmark,
        plc,
        evaluator.placement,
        n_hard,
        fixed,
        min_shared_nets=min_shared_nets,
        max_pairs=max_pairs,
    )
    if log_fn is not None:
        log_fn(
            f"  pair-swap budget={time_budget_s:.0f}s, "
            f"|candidate pairs|={len(pairs)} "
            f"(min_shared_nets={min_shared_nets}, capped at {max_pairs})"
        )

    if not pairs:
        if log_fn is not None:
            log_fn("  no candidate pairs — pair-swap skipped")
        return {"passes": 0, "total_improvement": 0.0, "wall_total_s": 0.0,
                "pairs_considered": 0, "swaps_accepted": 0}

    pass_idx = 0
    total_improvement = 0.0
    total_swaps = 0
    pairs_considered = 0
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        pass_idx += 1
        pass_t0 = time.perf_counter()
        pass_improvement = 0.0
        pass_swaps = 0

        for a, b in pairs:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            pairs_considered += 1
            delta = _try_swap(evaluator, a, b, macro_sizes_np, n_hard)
            if delta < 0:
                pass_improvement += delta
                pass_swaps += 1

        total_improvement += pass_improvement
        total_swaps += pass_swaps
        elapsed = time.perf_counter() - t_start
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  pair-swap pass {pass_idx}: swaps={pass_swaps}/{len(pairs)}, "
                f"Δ={pass_improvement:+.5f}, proxy={cur_proxy:.5f}, "
                f"pass_wall={time.perf_counter()-pass_t0:.1f}s, "
                f"total elapsed={elapsed:.1f}s"
            )
        if pass_swaps == 0:
            if log_fn is not None:
                log_fn(f"  pair-swap converged at pass {pass_idx} (no accepts)")
            break

    return {
        "passes": pass_idx,
        "total_improvement": total_improvement,
        "swaps_accepted": total_swaps,
        "pairs_considered": pairs_considered,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Placer ──────────────────────────────────────────────────────────────────


class CDPairSwapPlacer:
    """E15: CDAdaptive (E16 settings) + post-CD pair-swap polish phase."""

    def __init__(
        self,
        cd_hard_cap_s: float = 3000.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        swap_budget_s: float = 600.0,
        swap_min_shared_nets: int = 1,
        swap_max_pairs: int = 5000,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.swap_budget_s = float(swap_budget_s)
        self.swap_min_shared_nets = int(swap_min_shared_nets)
        self.swap_max_pairs = int(swap_max_pairs)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDPairSwapPlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, swap budget={self.swap_budget_s:.0f}s ==="
        )

        # 1. SDF init
        placement = sdf_init(benchmark)

        # 2. Project overlaps
        placement, proj_iters = project_overlaps(placement, benchmark)
        self._log(f"  overlap projection: {proj_iters} iters")

        # 3. Build evaluator
        _, plc = self._load_plc_for(benchmark)
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(f"  init proxy={init_cost['proxy']:.5f}")

        # 4. CD phase
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_hard_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, wall={cd_stats['wall_total_s']:.1f}s, "
            f"proxy={cd_proxy:.5f}"
        )

        # 5. Pair-swap phase
        n_hard = benchmark.num_hard_macros
        fixed_np = benchmark.macro_fixed.cpu().numpy()
        self._log(f"  starting pair-swap phase (budget={self.swap_budget_s:.0f}s)")
        swap_stats = run_pair_swap(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            n_hard=n_hard,
            fixed=fixed_np,
            time_budget_s=self.swap_budget_s,
            min_shared_nets=self.swap_min_shared_nets,
            max_pairs=self.swap_max_pairs,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  pair-swap done: passes={swap_stats['passes']}, "
            f"swaps={swap_stats['swaps_accepted']}/{swap_stats['pairs_considered']}, "
            f"Δ={swap_stats['total_improvement']:+.5f}, "
            f"final proxy={final_cost['proxy']:.5f}"
        )

        # 6. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDPairSwapPlacer produced {overlaps['overlap_count']} overlaps "
                f"on '{benchmark.name}'"
            )

        self._log(f"  total wall: {time.perf_counter() - t_total0:.1f} s")
        return final_placement
