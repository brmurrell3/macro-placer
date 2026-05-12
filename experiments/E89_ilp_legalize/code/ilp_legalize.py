"""E89 — ILP detailed legalize (PATH C2).

Take a zero-overlap placement, find small spatial clusters of hard macros,
build a candidate-grid per macro, solve a small assignment ILP per cluster
that minimizes proxy delta at fixed zero-overlap, apply moves sequentially.

Uses PATH A's `IncrementalProxyEvaluator.delta_cost` (read-only peek) for
proxy-delta evaluations, and `move(macro, new_xy)` to commit. Does NOT edit
PATH A code — only consumes the API.

Separable-objective approximation: ILP assumes other cluster macros stay at
their current positions when scoring each candidate. Real post-application
delta differs slightly; we accept that for the spike and verify outcome via
the canonical PlacementCost eval.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import highspy

from macro_place.incremental_evaluator import IncrementalProxyEvaluator


@dataclass
class Cluster:
    member_idx: List[int]           # macro indices in this cluster (hard macros only)
    candidates: List[List[Tuple[float, float]]]  # candidates[i] = list of (x,y) for member i
    delta_proxy: List[List[float]]  # delta_proxy[i][j] = proxy delta if member i moves to cand j


def _aabb_overlap(
    ax_min: float, ax_max: float, ay_min: float, ay_max: float,
    bx_min: float, bx_max: float, by_min: float, by_max: float,
    eps: float = 1e-4,
) -> bool:
    return (
        ax_max - eps > bx_min and bx_max - eps > ax_min and
        ay_max - eps > by_min and by_max - eps > ay_min
    )


def _candidate_grid(
    cx: float, cy: float, hw: float, hh: float,
    canvas_w: float, canvas_h: float,
    grid: int, span_frac: float,
) -> List[Tuple[float, float]]:
    """Generate a grid×grid candidate set around (cx, cy).

    Span = ±span_frac * macro_dim. Includes the original position when grid is odd.
    Clipped to keep the macro fully inside the canvas.
    """
    span_x = span_frac * 2 * hw
    span_y = span_frac * 2 * hh
    if grid <= 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-1.0, 1.0, grid).tolist()
    cands = []
    for ox in offsets:
        for oy in offsets:
            nx = cx + ox * span_x
            ny = cy + oy * span_y
            nx = min(max(nx, hw), canvas_w - hw)
            ny = min(max(ny, hh), canvas_h - hh)
            cands.append((nx, ny))
    # Dedupe (clip can collapse two grid cells to the same boundary point)
    seen = set()
    out = []
    for nx, ny in cands:
        key = (round(nx, 6), round(ny, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((nx, ny))
    return out


def _filter_candidates_against_fixed(
    candidates: List[Tuple[float, float]],
    macro_idx: int,
    sizes_np: np.ndarray,
    placement_np: np.ndarray,
    cluster_set: set,
    fixed_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-3,
) -> List[Tuple[float, float]]:
    """Drop any candidate that overlaps with a non-cluster HARD macro at its current
    position. Cluster members are handled later via ILP pairwise constraints."""
    hw = sizes_np[macro_idx, 0] / 2.0
    hh = sizes_np[macro_idx, 1] / 2.0
    kept = []
    for (cx, cy) in candidates:
        ax_min, ax_max = cx - hw, cx + hw
        ay_min, ay_max = cy - hh, cy + hh
        conflict = False
        for j in range(n_hard):
            if j == macro_idx or j in cluster_set:
                continue
            jx, jy = placement_np[j]
            jhw, jhh = sizes_np[j, 0] / 2.0, sizes_np[j, 1] / 2.0
            if _aabb_overlap(
                ax_min, ax_max, ay_min, ay_max,
                jx - jhw, jx + jhw, jy - jhh, jy + jhh,
                eps=eps,
            ):
                conflict = True
                break
        if not conflict:
            kept.append((cx, cy))
    return kept


def build_kmeans_clusters_by_knn(
    placement: torch.Tensor,
    fixed: torch.Tensor,
    n_hard: int,
    cluster_size: int,
    rng_seed: int = 0,
) -> List[List[int]]:
    """Greedy k-NN partition of hard macros into cluster_size-sized groups.

    Skips fixed macros (they're constants, not decision vars). Returns a list
    of clusters; each is a list of MOVABLE hard-macro indices.
    """
    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    movable = ~fixed[:n_hard].cpu().numpy()
    unclaimed = set(int(i) for i in np.where(movable)[0])
    clusters: List[List[int]] = []
    rng = np.random.default_rng(rng_seed)
    order = list(unclaimed)
    rng.shuffle(order)
    seed_iter = iter(order)
    while unclaimed:
        try:
            seed = next(seed_iter)
        except StopIteration:
            seed = next(iter(unclaimed))
        if seed not in unclaimed:
            continue
        # Find cluster_size-1 nearest movable unclaimed neighbors
        seed_pos = pos[seed]
        candidates = list(unclaimed - {seed})
        if not candidates:
            clusters.append([seed])
            unclaimed.discard(seed)
            continue
        cand_pos = pos[candidates]
        d = np.linalg.norm(cand_pos - seed_pos, axis=1)
        order_idx = np.argsort(d)
        take = min(cluster_size - 1, len(candidates))
        neighbors = [candidates[i] for i in order_idx[:take]]
        cluster = [seed] + neighbors
        clusters.append(cluster)
        for m in cluster:
            unclaimed.discard(m)
    return clusters


def build_cluster_ilp_data(
    cluster_members: List[int],
    evaluator: IncrementalProxyEvaluator,
    benchmark,
    n_hard: int,
    grid: int,
    span_frac: float,
) -> Optional[Cluster]:
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    placement_np = evaluator.placement.cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cluster_set = set(int(i) for i in cluster_members)

    candidates: List[List[Tuple[float, float]]] = []
    delta_proxy: List[List[float]] = []
    for m in cluster_members:
        cx, cy = placement_np[m]
        hw = sizes_np[m, 0] / 2.0
        hh = sizes_np[m, 1] / 2.0
        raw = _candidate_grid(cx, cy, hw, hh, cw, ch, grid=grid, span_frac=span_frac)
        # Always include the current position so the ILP can choose "no move".
        current = (cx, cy)
        if current not in raw:
            raw.insert(0, current)
        kept = _filter_candidates_against_fixed(
            raw, m, sizes_np, placement_np, cluster_set, fixed_np, n_hard,
        )
        if not kept:
            # No legal moves for this macro — pin it.
            kept = [current]
        # Score each candidate via delta_cost (read-only peek)
        deltas = []
        for (nx, ny) in kept:
            d = evaluator.delta_cost(m, (nx, ny))
            deltas.append(float(d["proxy"]) - float(evaluator.current_cost()["proxy"]))
        candidates.append(kept)
        delta_proxy.append(deltas)
    return Cluster(member_idx=list(cluster_members), candidates=candidates, delta_proxy=delta_proxy)


def solve_cluster_ilp(
    cluster: Cluster,
    sizes_np: np.ndarray,
    time_limit_s: float = 5.0,
    eps: float = 1e-3,
    verbose: bool = False,
) -> Optional[List[int]]:
    """Solve the small assignment ILP for one cluster.

    Returns a list (one per cluster member) of the chosen candidate index, or
    None if the solver fails / times out.
    """
    K = len(cluster.member_idx)
    if K == 0:
        return []
    var_idx: dict[Tuple[int, int], int] = {}
    n_var = 0
    for i in range(K):
        for j in range(len(cluster.candidates[i])):
            var_idx[(i, j)] = n_var
            n_var += 1
    if n_var == 0:
        return [0] * K

    h = highspy.Highs()
    if not verbose:
        h.silent()
    lp = highspy.HighsLp()
    lp.num_col_ = n_var
    lp.col_lower_ = [0.0] * n_var
    lp.col_upper_ = [1.0] * n_var
    obj = [0.0] * n_var
    for i in range(K):
        for j in range(len(cluster.candidates[i])):
            obj[var_idx[(i, j)]] = cluster.delta_proxy[i][j]
    lp.col_cost_ = obj
    lp.sense_ = highspy.ObjSense.kMinimize
    integrality = [highspy.HighsVarType.kInteger] * n_var
    lp.integrality_ = integrality

    # Constraints: row by row, sparse CSR
    rows_start = [0]
    cols_idx: List[int] = []
    cols_val: List[float] = []
    row_lower: List[float] = []
    row_upper: List[float] = []

    def end_row(lo: float, up: float) -> None:
        rows_start.append(len(cols_idx))
        row_lower.append(lo)
        row_upper.append(up)

    # Each cluster member picks exactly one candidate
    for i in range(K):
        for j in range(len(cluster.candidates[i])):
            cols_idx.append(var_idx[(i, j)])
            cols_val.append(1.0)
        end_row(1.0, 1.0)

    # Pairwise non-overlap among cluster members
    for i1 in range(K):
        m1 = cluster.member_idx[i1]
        hw1 = sizes_np[m1, 0] / 2.0
        hh1 = sizes_np[m1, 1] / 2.0
        for i2 in range(i1 + 1, K):
            m2 = cluster.member_idx[i2]
            hw2 = sizes_np[m2, 0] / 2.0
            hh2 = sizes_np[m2, 1] / 2.0
            for j1, (x1, y1) in enumerate(cluster.candidates[i1]):
                for j2, (x2, y2) in enumerate(cluster.candidates[i2]):
                    if _aabb_overlap(
                        x1 - hw1, x1 + hw1, y1 - hh1, y1 + hh1,
                        x2 - hw2, x2 + hw2, y2 - hh2, y2 + hh2,
                        eps=eps,
                    ):
                        cols_idx.append(var_idx[(i1, j1)])
                        cols_val.append(1.0)
                        cols_idx.append(var_idx[(i2, j2)])
                        cols_val.append(1.0)
                        end_row(0.0, 1.0)

    n_row = len(row_lower)
    lp.num_row_ = n_row
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = rows_start
    lp.a_matrix_.index_ = cols_idx
    lp.a_matrix_.value_ = cols_val

    h.passModel(lp)
    h.setOptionValue("time_limit", float(time_limit_s))
    status = h.run()
    if status != highspy.HighsStatus.kOk:
        return None
    sol = h.getSolution()
    if sol is None or len(sol.col_value) != n_var:
        return None
    picks = [0] * K
    for i in range(K):
        best_j = 0
        best_val = -1.0
        for j in range(len(cluster.candidates[i])):
            v = sol.col_value[var_idx[(i, j)]]
            if v > best_val:
                best_val = v
                best_j = j
        picks[i] = best_j
    return picks


def apply_picks_sequentially(
    cluster: Cluster,
    picks: List[int],
    evaluator: IncrementalProxyEvaluator,
) -> dict:
    """Apply each pick via evaluator.move. Returns aggregate before/after proxy
    + count of moves that actually changed position.
    """
    proxy_before = float(evaluator.current_cost()["proxy"])
    n_moved = 0
    for i, m in enumerate(cluster.member_idx):
        chosen = cluster.candidates[i][picks[i]]
        cur = (float(evaluator.placement[m, 0]), float(evaluator.placement[m, 1]))
        if abs(chosen[0] - cur[0]) < 1e-6 and abs(chosen[1] - cur[1]) < 1e-6:
            continue
        evaluator.move(m, chosen)
        n_moved += 1
    proxy_after = float(evaluator.current_cost()["proxy"])
    return {
        "proxy_before": proxy_before,
        "proxy_after": proxy_after,
        "delta": proxy_after - proxy_before,
        "n_moved": n_moved,
    }
