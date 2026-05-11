"""E67 — K=50 Hungarian re-pack as a non-local feasibility-respecting move.

Drop-in replacement for the E41 K-joint K=3 brute-force move generator. Where
E41 enumerates N^K combos (N=5, K=3 → 125 per K-tuple), this module solves a
K × n_slots Hungarian assignment per step (K=50, n_slots=100 → polynomial
O(K^2 * n_slots) ≈ 250 k cells, ~100 ms per solve once the cost matrix is
built).

Public API:
  - select_cluster(benchmark, placement, evaluator, k=50, mode='adjacency')
  - generate_slots(benchmark, placement, plc, cluster_idx, n_slots=100)
  - build_cost_matrix(evaluator, cluster_idx, slots, n_hard, macro_sizes_np)
  - solve_hungarian(cost_matrix)
  - commit_if_better(evaluator, benchmark, cluster_idx, assignment, slots,
                     baseline_proxy)
  - kjoint_hungarian_step(benchmark, placement, plc, evaluator=None,
                          k=50, n_slots=100, mode='adjacency')

The cost matrix is built via incremental-evaluator deltas (E1, 4657× speedup;
load-bearing here — without it the K × n_slots cell evaluations would take
minutes instead of seconds). Cells where macro_i cannot legally sit at slot_j
(overlap with the *non-cluster* background, ignoring fixed cluster siblings
since they're "destroyed" for the assignment) get cost = +inf.

Defensive revert is mandatory: after applying all K moves, we recompute
`compute_overlap_metrics` over the full placement; any non-zero overlap or
proxy regression triggers full revert to the pre-step state.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Repo root on sys.path (eval harness uses spec_from_file_location).
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics


# ── Cluster selection ──────────────────────────────────────────────────────


def _adjacency_scores(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: Sequence[int],
) -> np.ndarray:
    """Per-macro adjacency = sum_{n in macro_to_nets[m]} 1/max(1, |net_n|-1).

    Reused verbatim from E39 (`experiments/E39_kmacro_joint_lns/code/
    cd_lns_sa_kjoint.py:447`). Higher score = more couplings to small nets,
    which are the dominant proxy contributors. Aligned with the input
    `hard_movable` order.
    """
    out = np.zeros(len(hard_movable), dtype=np.float64)
    for i, m in enumerate(hard_movable):
        nets = evaluator.macro_to_nets[m].tolist()
        s = 0.0
        for n in nets:
            sz = int(evaluator.net_pins[n].shape[0])
            denom = max(1, sz - 1)
            s += 1.0 / denom
        out[i] = s
    return out


def select_cluster(
    benchmark: Benchmark,
    evaluator: IncrementalProxyEvaluator,
    k: int = 50,
    mode: str = "adjacency",
    cluster_seed: Optional[int] = None,
    pool_multiplier: int = 3,
) -> List[int]:
    """Return indices of `k` hard movable macros forming a coupled cluster.

    `mode='adjacency'`:
        - cluster_seed=None: deterministic top-k by `_adjacency_scores`
          (the original behavior).
        - cluster_seed=int: random sample of k from the top-(pool_multiplier·k)
          by adjacency, RNG seeded by cluster_seed. This is the E41 K-joint
          diversification pattern — once the deterministic top-k is exhausted
          by repeated calls, sampling from the top pool finds new clusters
          while still biased toward high-coupling macros.

    `mode='lp_dual'`: not implemented (LP infra deleted in commit 44efd16).

    Excludes fixed macros (`benchmark.macro_fixed[i] == True`).
    """
    if mode != "adjacency":
        raise NotImplementedError(
            f"select_cluster mode={mode!r} not implemented; only 'adjacency' "
            "is available in this scaffold."
        )
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]
    if len(hard_movable) < k:
        k = len(hard_movable)
    adj = _adjacency_scores(evaluator, hard_movable)
    sorted_idx = np.argsort(-adj)

    if cluster_seed is None:
        return [hard_movable[i] for i in sorted_idx[:k]]

    pool_size = min(len(hard_movable), max(k, k * pool_multiplier))
    pool = sorted_idx[:pool_size]
    rng = np.random.default_rng(cluster_seed)
    chosen = rng.choice(pool, size=k, replace=False)
    return [hard_movable[i] for i in chosen]


# ── Slot generation ────────────────────────────────────────────────────────


def generate_slots(
    benchmark: Benchmark,
    placement: torch.Tensor,
    plc,
    cluster_idx: Sequence[int],
    n_slots: int = 100,
    bbox_pad_frac: float = 0.10,
) -> torch.Tensor:
    """Generate `n_slots` candidate (cx, cy) centers around the cluster bbox.

    Strategy (E12 grid-bin pattern; see `submissions/cd_lns_gridbin/placer.py:
    134-180` and `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py:
    147-200` for the reference):

      1. Compute cluster axis-aligned bounding box from current centers.
      2. Pad by `bbox_pad_frac` on each side, clamp to canvas.
      3. Walk plc grid bin centers (col, row) inside the padded bbox, in
         row-major order. Skip bins where ANY canvas-bound check fails for
         the median cluster macro (rough viability filter).
      4. Always include the current center of every cluster macro (k slots).
      5. Trim or pad to exactly `n_slots`. If the bbox produces fewer than
         `n_slots` viable bins, expand to the full canvas as a fallback.

    Fixed macros are not in `cluster_idx` (per `select_cluster`), so we don't
    special-case them here. Per-cell fixed-macro overlap is enforced later in
    `build_cost_matrix` via `_is_legal_2d_excluded`.

    Returns a `[n_slots_actual, 2]` float64 tensor of (cx, cy) centers.
    """
    cluster_idx = list(cluster_idx)
    cw = float(plc.width)
    ch = float(plc.height)
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = cw / grid_col
    grid_h = ch / grid_row

    pos = placement[cluster_idx].cpu().numpy().astype(np.float64)
    cx_min, cy_min = pos.min(axis=0)
    cx_max, cy_max = pos.max(axis=0)
    span_x = cx_max - cx_min
    span_y = cy_max - cy_min
    pad_x = max(grid_w, span_x * bbox_pad_frac)
    pad_y = max(grid_h, span_y * bbox_pad_frac)
    bx_lo = max(0.0, cx_min - pad_x)
    bx_hi = min(cw, cx_max + pad_x)
    by_lo = max(0.0, cy_min - pad_y)
    by_hi = min(ch, cy_max + pad_y)

    # Use median cluster size for the viability filter (per-macro half-extents
    # are checked in build_cost_matrix; this is just a coarse "is this bin
    # near the cluster?" gate).
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    cluster_sizes = sizes[cluster_idx]
    med_w_half = float(np.median(cluster_sizes[:, 0])) / 2.0
    med_h_half = float(np.median(cluster_sizes[:, 1])) / 2.0

    candidates: List[Tuple[float, float]] = []

    # Always include current centers (one per cluster macro) as fallbacks.
    for c_pos in pos:
        candidates.append((float(c_pos[0]), float(c_pos[1])))

    def _walk_grid(lo_x: float, hi_x: float, lo_y: float, hi_y: float):
        for col in range(grid_col):
            cx = (col + 0.5) * grid_w
            if cx < lo_x or cx > hi_x:
                continue
            if cx < med_w_half or cx > cw - med_w_half:
                continue
            for row in range(grid_row):
                cy = (row + 0.5) * grid_h
                if cy < lo_y or cy > hi_y:
                    continue
                if cy < med_h_half or cy > ch - med_h_half:
                    continue
                yield (cx, cy)

    seen = {(round(x, 6), round(y, 6)) for x, y in candidates}
    for cx, cy in _walk_grid(bx_lo, bx_hi, by_lo, by_hi):
        key = (round(cx, 6), round(cy, 6))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((cx, cy))
        if len(candidates) >= n_slots:
            break

    # Fallback: expand to full canvas if bbox produced too few.
    if len(candidates) < n_slots:
        for cx, cy in _walk_grid(0.0, cw, 0.0, ch):
            key = (round(cx, 6), round(cy, 6))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((cx, cy))
            if len(candidates) >= n_slots:
                break

    return torch.tensor(candidates[:n_slots], dtype=torch.float64)


# ── Legality (reused from E39) ─────────────────────────────────────────────


def _is_legal_2d_excluded(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    excluded: Sequence[int],
    eps: float = 1e-9,
) -> bool:
    """Macro `idx` at (x, y) overlaps no hard macro outside `excluded`.

    Reused from E39 (`experiments/E39_kmacro_joint_lns/code/
    cd_lns_sa_kjoint.py:103`). `eps > 0` demands STRICT separation in at
    least one axis — keeps the check at-least-as-strict as
    `compute_overlap_metrics` (E39 ibm07 wrong-direction-eps gotcha;
    `docs/gotchas.md`, memory `kjoint_overlap_eps_gotcha.md`).
    """
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
    blockers = (dx < min_dx + eps) & (dy < min_dy + eps)
    blockers[idx] = False
    for e in excluded:
        if 0 <= e < n_hard:
            blockers[e] = False
    return not bool(np.any(blockers))


# ── Cost matrix (incremental deltas) ───────────────────────────────────────


def build_cost_matrix(
    evaluator: IncrementalProxyEvaluator,
    cluster_idx: Sequence[int],
    slots: torch.Tensor,
    n_hard: int,
    macro_sizes_np: np.ndarray,
) -> np.ndarray:
    """Return [k, n_slots] cost matrix of proxy DELTAS for assignment.

    cost[i, j] = proxy_after_moving_macro_i_to_slot_j  -  baseline_proxy

    Cells where the assignment would overlap a non-cluster hard macro get
    +inf. Cells where the macro is already at the slot (within float eps)
    get 0.0 directly (skip the move/revert round trip).

    Uses the incremental evaluator (E1) for O(degree) per cell. Side-
    effect-neutral: every move is reverted before the next probe, and the
    evaluator state is identical at function entry and exit.
    """
    cluster_idx = list(cluster_idx)
    k = len(cluster_idx)
    n_slots = int(slots.shape[0])
    cost = np.full((k, n_slots), np.inf, dtype=np.float64)
    baseline = evaluator.current_cost()["proxy"]
    slots_np = slots.cpu().numpy().astype(np.float64)

    for i, m in enumerate(cluster_idx):
        cur_x = float(evaluator.placement[m, 0])
        cur_y = float(evaluator.placement[m, 1])
        for j in range(n_slots):
            sx = float(slots_np[j, 0])
            sy = float(slots_np[j, 1])
            # No-op: macro already here.
            if abs(sx - cur_x) < 1e-9 and abs(sy - cur_y) < 1e-9:
                cost[i, j] = 0.0
                continue
            # Legality vs the non-cluster background (cluster siblings are
            # excluded — they're being reassigned in this step).
            if not _is_legal_2d_excluded(
                m, sx, sy, evaluator.placement, macro_sizes_np,
                n_hard, excluded=cluster_idx,
            ):
                continue
            evaluator.move(m, (sx, sy))
            new_proxy = evaluator.current_cost()["proxy"]
            evaluator.revert()
            cost[i, j] = new_proxy - baseline
    return cost


# ── Hungarian solve ────────────────────────────────────────────────────────


def solve_hungarian(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve the rectangular assignment problem.

    Returns a length-k array where `assignment[i]` is the slot index for
    cluster macro i. `scipy.optimize.linear_sum_assignment` handles
    rectangular matrices (k <= n_slots) by selecting k of the n_slots
    columns; +inf rows raise.

    If a row is all-+inf, that row's macro has no legal slot in the pool.
    We replace +inf rows with all zeros so the Hungarian doesn't crash;
    the post-commit overlap check will then either revert or accept.
    """
    cost = cost_matrix.copy()
    inf_rows = np.all(~np.isfinite(cost), axis=1)
    if np.any(inf_rows):
        cost[inf_rows, :] = 0.0  # sentinel: no improvement, will be filtered
    # Replace remaining +inf with a large finite penalty so scipy can solve.
    finite_max = np.max(cost[np.isfinite(cost)]) if np.any(np.isfinite(cost)) else 1.0
    big = max(1.0, finite_max) * 1e6
    cost = np.where(np.isfinite(cost), cost, big)
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind is [0, 1, ..., k-1] in order; col_ind is the assigned slot per row.
    assignment = np.full(cost.shape[0], -1, dtype=np.int64)
    assignment[row_ind] = col_ind
    return assignment


# ── Commit + defensive revert ──────────────────────────────────────────────


def commit_if_better(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    cluster_idx: Sequence[int],
    assignment: np.ndarray,
    slots: torch.Tensor,
    baseline_proxy: float,
    proxy_eps: float = 1e-7,
) -> Tuple[bool, Dict[str, float]]:
    """Apply joint move per `assignment`; revert if any overlap or proxy worsens.

    Returns (accepted, info). info has:
      - 'proxy_before', 'proxy_after'
      - 'overlap_count', 'overlap_area' (post-commit, before any revert)
      - 'reverted' (True if defensive revert fired)
      - 'n_moved' (number of macros whose slot != current pre-commit position)

    NOTE: revert is single-step in the evaluator (`docs/gotchas.md`), so we
    capture pre-commit positions explicitly and walk back via `move()` calls
    if the commit fails.
    """
    cluster_idx = list(cluster_idx)
    slots_np = slots.cpu().numpy().astype(np.float64)
    saved: List[Tuple[int, float, float]] = []
    moved: List[int] = []

    # Snapshot pre-commit positions so we can walk back on revert.
    for m in cluster_idx:
        saved.append(
            (m, float(evaluator.placement[m, 0]), float(evaluator.placement[m, 1]))
        )

    for i, m in enumerate(cluster_idx):
        slot_j = int(assignment[i])
        if slot_j < 0:
            continue
        sx = float(slots_np[slot_j, 0])
        sy = float(slots_np[slot_j, 1])
        cx = float(evaluator.placement[m, 0])
        cy = float(evaluator.placement[m, 1])
        if abs(sx - cx) < 1e-9 and abs(sy - cy) < 1e-9:
            continue
        try:
            evaluator.move(m, (sx, sy))
            moved.append(m)
        except Exception:
            # Walk back any moves applied so far and bail.
            for mm in reversed(moved):
                _, ox, oy = next(s for s in saved if s[0] == mm)
                evaluator.move(mm, (ox, oy))
            return False, {
                "proxy_before": baseline_proxy,
                "proxy_after": float("nan"),
                "overlap_count": -1,
                "overlap_area": float("nan"),
                "reverted": True,
                "n_moved": 0,
                "reason": "move_exception",
            }

    proxy_after = evaluator.current_cost()["proxy"]
    ov = compute_overlap_metrics(evaluator.placement, benchmark)
    overlap_count = int(ov["overlap_count"])
    overlap_area = float(ov["total_overlap_area"])

    accept = (
        overlap_count == 0
        and proxy_after < baseline_proxy - proxy_eps
    )

    if not accept:
        # Walk back every moved macro to its saved position.
        for mm in reversed(moved):
            _, ox, oy = next(s for s in saved if s[0] == mm)
            cx = float(evaluator.placement[mm, 0])
            cy = float(evaluator.placement[mm, 1])
            if abs(ox - cx) > 1e-9 or abs(oy - cy) > 1e-9:
                evaluator.move(mm, (ox, oy))

    return accept, {
        "proxy_before": baseline_proxy,
        "proxy_after": proxy_after,
        "overlap_count": overlap_count,
        "overlap_area": overlap_area,
        "reverted": not accept,
        "n_moved": len(moved),
        "reason": (
            "ok" if accept
            else ("overlap" if overlap_count > 0 else "no_proxy_improvement")
        ),
    }


# ── V2: sequential commit (pairwise-blindness fix) ─────────────────────────


def commit_if_better_sequential(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    cluster_idx: Sequence[int],
    assignment: np.ndarray,
    slots: torch.Tensor,
    baseline_proxy: float,
    cost_matrix: Optional[np.ndarray] = None,
    proxy_eps: float = 1e-7,
) -> Tuple[bool, Dict[str, float]]:
    """V2: apply Hungarian moves one at a time with per-move legality check.

    Fixes V1's pairwise-blindness (Risk b in the manifest): the Hungarian
    cost matrix is built per-macro independently, so the optimal assignment
    can have multiple macros target overlapping slots. V1's commit applied
    all moves jointly then reverted the whole batch on the first overlap.
    V2 orders moves by predicted improvement (most-negative cost first) and
    tests legality against the *committed* state before each move — already-
    placed cluster siblings are at their new positions in `evaluator.placement`,
    so `_is_legal_2d_excluded(..., excluded=[])` rejects collisions
    automatically.

    Returns (accepted, info) with the same shape as `commit_if_better` plus:
      - `n_skipped_illegal`: moves dropped because the slot was already
        occupied by a previously-committed cluster sibling (or any other
        hard macro that the cost matrix's per-cell legality check missed).
      - `n_pending`: moves considered (excludes no-ops).

    Cumulative-proxy revert path is preserved: even though per-move legality
    prevents overlap, the sum of accepted moves may still fail to improve
    over baseline (rare but possible due to per-cell vs. global proxy drift
    in the incremental evaluator). In that case the entire batch reverts.
    """
    cluster_idx = list(cluster_idx)
    slots_np = slots.cpu().numpy().astype(np.float64)
    n_hard = int(benchmark.num_hard_macros)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()

    saved: List[Tuple[int, float, float]] = []
    for m in cluster_idx:
        saved.append(
            (m, float(evaluator.placement[m, 0]), float(evaluator.placement[m, 1]))
        )

    pending: List[Tuple[float, int, int, float, float]] = []
    n_no_op = 0
    n_no_assignment = 0
    for i, m in enumerate(cluster_idx):
        slot_j = int(assignment[i])
        if slot_j < 0:
            n_no_assignment += 1
            continue
        sx = float(slots_np[slot_j, 0])
        sy = float(slots_np[slot_j, 1])
        cx = float(evaluator.placement[m, 0])
        cy = float(evaluator.placement[m, 1])
        if abs(sx - cx) < 1e-9 and abs(sy - cy) < 1e-9:
            n_no_op += 1
            continue
        if cost_matrix is not None:
            predicted = float(cost_matrix[i, slot_j])
            if not np.isfinite(predicted):
                continue
        else:
            predicted = 0.0
        pending.append((predicted, i, m, sx, sy))
    pending.sort(key=lambda x: x[0])

    moved: List[int] = []
    skipped_illegal = 0
    skipped_exception = 0
    for _predicted, _i, m, sx, sy in pending:
        if not _is_legal_2d_excluded(
            m, sx, sy, evaluator.placement, macro_sizes_np,
            n_hard, excluded=[],
        ):
            skipped_illegal += 1
            continue
        try:
            evaluator.move(m, (sx, sy))
            moved.append(m)
        except Exception:
            skipped_exception += 1
            continue

    proxy_after = evaluator.current_cost()["proxy"]
    ov = compute_overlap_metrics(evaluator.placement, benchmark)
    overlap_count = int(ov["overlap_count"])
    overlap_area = float(ov["total_overlap_area"])

    accept = (
        overlap_count == 0
        and proxy_after < baseline_proxy - proxy_eps
        and len(moved) > 0
    )

    if not accept:
        for mm in reversed(moved):
            _, ox, oy = next(s for s in saved if s[0] == mm)
            cx = float(evaluator.placement[mm, 0])
            cy = float(evaluator.placement[mm, 1])
            if abs(ox - cx) > 1e-9 or abs(oy - cy) > 1e-9:
                evaluator.move(mm, (ox, oy))

    return accept, {
        "proxy_before": baseline_proxy,
        "proxy_after": proxy_after,
        "overlap_count": overlap_count,
        "overlap_area": overlap_area,
        "reverted": not accept,
        "n_moved": len(moved),
        "n_pending": len(pending),
        "n_skipped_illegal": skipped_illegal,
        "n_skipped_exception": skipped_exception,
        "n_no_op": n_no_op,
        "n_no_assignment": n_no_assignment,
        "reason": (
            "ok" if accept
            else ("overlap" if overlap_count > 0
                  else ("zero_moved" if len(moved) == 0
                        else "no_proxy_improvement"))
        ),
    }


# ── End-to-end one-shot step ───────────────────────────────────────────────


def kjoint_hungarian_step(
    benchmark: Benchmark,
    placement: torch.Tensor,
    plc,
    evaluator: Optional[IncrementalProxyEvaluator] = None,
    k: int = 50,
    n_slots: int = 100,
    mode: str = "adjacency",
    bbox_pad_frac: float = 0.10,
    commit_mode: str = "sequential",
    cluster_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one K=50 Hungarian re-pack step.

    If `evaluator` is None, builds a fresh `IncrementalProxyEvaluator` from
    `(benchmark, plc, placement)`. Otherwise mutates the provided evaluator
    in place (caller owns lifetime).

    Returns (new_placement, info_dict). `info_dict` includes per-substep
    wall times and the commit info.

    Side-effect contract:
      - On accept: evaluator state reflects the committed move; new_placement
        is the post-commit `evaluator.placement.clone()`.
      - On reject: evaluator state is restored to entry; new_placement is
        the same tensor as the input (passed through, not a clone).
    """
    info: Dict[str, float] = {}
    t0 = time.perf_counter()

    if evaluator is None:
        plc_arg = plc
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc_arg, placement_f64)
    info["t_eval_init_s"] = time.perf_counter() - t0

    n_hard = int(benchmark.num_hard_macros)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    baseline_proxy = evaluator.current_cost()["proxy"]
    info["proxy_baseline"] = baseline_proxy

    t1 = time.perf_counter()
    cluster_idx = select_cluster(
        benchmark, evaluator, k=k, mode=mode, cluster_seed=cluster_seed,
    )
    info["t_cluster_select_s"] = time.perf_counter() - t1
    info["k_actual"] = len(cluster_idx)
    info["cluster_seed"] = cluster_seed

    t2 = time.perf_counter()
    slots = generate_slots(
        benchmark, evaluator.placement, plc, cluster_idx,
        n_slots=n_slots, bbox_pad_frac=bbox_pad_frac,
    )
    info["t_slots_gen_s"] = time.perf_counter() - t2
    info["n_slots_actual"] = int(slots.shape[0])

    t3 = time.perf_counter()
    cost = build_cost_matrix(
        evaluator, cluster_idx, slots, n_hard, macro_sizes_np
    )
    info["t_cost_matrix_s"] = time.perf_counter() - t3
    n_finite = int(np.sum(np.isfinite(cost)))
    info["cost_finite_cells"] = n_finite
    info["cost_finite_frac"] = n_finite / float(cost.size) if cost.size else 0.0
    if n_finite > 0:
        info["cost_min"] = float(np.min(cost[np.isfinite(cost)]))
        info["cost_max"] = float(np.max(cost[np.isfinite(cost)]))
    else:
        info["cost_min"] = float("nan")
        info["cost_max"] = float("nan")

    t4 = time.perf_counter()
    assignment = solve_hungarian(cost)
    info["t_hungarian_s"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    if commit_mode == "sequential":
        accepted, commit_info = commit_if_better_sequential(
            evaluator, benchmark, cluster_idx, assignment, slots,
            baseline_proxy, cost_matrix=cost,
        )
    elif commit_mode == "joint":
        accepted, commit_info = commit_if_better(
            evaluator, benchmark, cluster_idx, assignment, slots, baseline_proxy,
        )
    else:
        raise ValueError(
            f"commit_mode={commit_mode!r} not in {{'sequential', 'joint'}}"
        )
    info["t_commit_s"] = time.perf_counter() - t5
    info["commit_mode"] = commit_mode
    info.update({f"commit_{k_}": v for k_, v in commit_info.items()})
    info["accepted"] = bool(accepted)

    if accepted:
        new_placement = evaluator.placement.detach().clone().to(placement.dtype)
    else:
        new_placement = placement
    info["t_total_s"] = time.perf_counter() - t0

    info["cluster_idx_head"] = cluster_idx[: min(10, len(cluster_idx))]
    info["assignment_head"] = assignment[: min(10, len(assignment))].tolist()
    return new_placement, info
