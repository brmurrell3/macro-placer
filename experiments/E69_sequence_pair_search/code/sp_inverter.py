"""Murata-Fujiyoshi sequence-pair encoder, decoder, and distance.

Cartesian (x,y) per macro <-> SP = (Gamma+, Gamma-) where each Gamma is a
permutation of macro indices. Encoding rule:

  i.right <= j.left            -> 'left':  i before j in BOTH G+ and G-
  i.left  >= j.right           -> 'right': j before i in BOTH G+ and G-
  i.top   <= j.bottom          -> 'below': i before j in G+, j before i in G-
  i.bottom >= j.top            -> 'above': j before i in G+, i before j in G-

Convention: prefer 'left'/'right' (horizontal relation) when both H and V
separations are possible; this gives a canonical encoding for non-overlapping
placements with mixed H/V slack.

Decoding (compaction): for each axis, build a longest-path DAG over the
relation graph (H-relations -> x DAG, V-relations -> y DAG); each macro's
coordinate = longest path from source. Result is bottom-left compact packing.

Distance: pair-relation Hamming distance. Number of pairs (i, j) where SP_a
and SP_b yield different one-of-{left, right, below, above}.

For ~500 hard macros the encoder is O(N^2 + N log N) ~ 250k ops <100 ms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


# Pair-relation codes
LEFT = 0   # i strictly left of j (i.right <= j.left)
RIGHT = 1  # i strictly right of j (i.left >= j.right)
BELOW = 2  # i strictly below j (i.top <= j.bottom)
ABOVE = 3  # i strictly above j (i.bottom >= j.top)


@dataclass
class SequencePair:
    """SP encoding of a placement."""
    gamma_plus: np.ndarray   # [n_hard] int64 — permutation of hard macro indices
    gamma_minus: np.ndarray  # [n_hard] int64 — permutation of hard macro indices

    def rank_plus(self) -> np.ndarray:
        r = np.empty_like(self.gamma_plus)
        r[self.gamma_plus] = np.arange(len(self.gamma_plus))
        return r

    def rank_minus(self) -> np.ndarray:
        r = np.empty_like(self.gamma_minus)
        r[self.gamma_minus] = np.arange(len(self.gamma_minus))
        return r


def _pair_relation(
    pos: np.ndarray, sizes: np.ndarray, i: int, j: int, eps: float = 1e-3
) -> int:
    """Decide LEFT / RIGHT / BELOW / ABOVE for pair (i, j) by Cartesian.

    Prefer horizontal when both apply (this guarantees an acyclic G+/G- by
    a partial-order-on-x argument). eps ~ 1nm absorbs float32 boundary
    noise (positions drift ~1e-6 from intended at canvas scales 1-23μm;
    1e-3 is well above float noise yet below physical macro scales of
    0.1-7 microns).

    Direction conventions:
      i.right ≤ j.left + eps    →  i to the LEFT of j  (j is east of i)
      i.left + eps ≥ j.right    →  i to the RIGHT of j (j is west of i)
      i.top ≤ j.bottom + eps    →  i BELOW j           (j is north of i)
      i.bottom + eps ≥ j.top    →  i ABOVE j           (j is south of i)
    """
    hi_w = sizes[i, 0] / 2.0
    hi_h = sizes[i, 1] / 2.0
    hj_w = sizes[j, 0] / 2.0
    hj_h = sizes[j, 1] / 2.0
    i_left = pos[i, 0] - hi_w
    i_right = pos[i, 0] + hi_w
    i_bot = pos[i, 1] - hi_h
    i_top = pos[i, 1] + hi_h
    j_left = pos[j, 0] - hj_w
    j_right = pos[j, 0] + hj_w
    j_bot = pos[j, 1] - hj_h
    j_top = pos[j, 1] + hj_h

    if i_right <= j_left + eps:
        return LEFT
    if i_left + eps >= j_right:
        return RIGHT
    if i_top <= j_bot + eps:
        return BELOW
    if i_bot + eps >= j_top:
        return ABOVE
    # Overlap (fall back to dominant axis).
    cx_diff = pos[i, 0] - pos[j, 0]
    cy_diff = pos[i, 1] - pos[j, 1]
    if abs(cx_diff) >= abs(cy_diff):
        return LEFT if cx_diff < 0 else RIGHT
    return BELOW if cy_diff < 0 else ABOVE


def encode(positions: torch.Tensor, sizes: torch.Tensor, n_hard: int,
           eps: float = 1e-3) -> SequencePair:
    """Encode Cartesian positions to SP via Murata-Fujiyoshi adjacency rule.

    This is the canonical inversion: only "directly adjacent" pairs add
    constraint edges. A pair has an H-edge iff their y-projections overlap
    AND they are strictly H-separated. A pair has a V-edge iff their
    x-projections overlap AND they are strictly V-separated. Diagonal
    pairs (neither projection overlaps) impose NO edge — their relative
    order is determined by topological tie-breaking using a canonical
    spatial key.

    This rule is provably acyclic for non-overlapping placements (Murata
    et al. 1995). My earlier pair-by-pair "H-first-or-V" rule
    over-constrained the graph by adding LEFT edges to diagonal pairs,
    producing false cycles on dense outputs.

    Args:
      positions: [N, 2] tensor of macro centers (only first n_hard used).
      sizes:     [N, 2] tensor of macro (w, h) (only first n_hard used).
      n_hard:    number of hard macros to encode.
      eps:       boundary slack for adjacency tests.

    Returns:
      SequencePair with gamma_plus, gamma_minus as np.int64 arrays.
    """
    pos = positions[:n_hard].detach().cpu().numpy().astype(np.float64)
    siz = sizes[:n_hard].detach().cpu().numpy().astype(np.float64)

    # G_plus  = (H-LEFT edges) ∪ (V-BELOW edges)  — i precedes j in Γ+
    # G_minus = (H-LEFT edges) ∪ (V-ABOVE edges, recoded as i→j when i is
    #           BELOW j, i.e. j precedes i in Γ-) — i precedes j in Γ-
    g_plus_succ: List[List[int]] = [[] for _ in range(n_hard)]
    g_minus_succ: List[List[int]] = [[] for _ in range(n_hard)]
    in_plus = np.zeros(n_hard, dtype=np.int32)
    in_minus = np.zeros(n_hard, dtype=np.int32)

    # Precompute bbox edges.
    x_left = pos[:, 0] - siz[:, 0] / 2.0
    x_right = pos[:, 0] + siz[:, 0] / 2.0
    y_bot = pos[:, 1] - siz[:, 1] / 2.0
    y_top = pos[:, 1] + siz[:, 1] / 2.0

    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            # Y-projections overlap if [y_bot_i, y_top_i] ∩ [y_bot_j, y_top_j] != ∅.
            y_overlap = (y_top[i] > y_bot[j] + eps) and (y_top[j] > y_bot[i] + eps)
            x_overlap = (x_right[i] > x_left[j] + eps) and (x_right[j] > x_left[i] + eps)

            # H-edge: y-projections overlap AND strict H-separation.
            if y_overlap:
                if x_right[i] <= x_left[j] + eps:
                    # i strictly LEFT of j. Both Γ+ and Γ-: i precedes j.
                    g_plus_succ[i].append(j); in_plus[j] += 1
                    g_minus_succ[i].append(j); in_minus[j] += 1
                elif x_right[j] <= x_left[i] + eps:
                    # i strictly RIGHT of j. j precedes i in both.
                    g_plus_succ[j].append(i); in_plus[i] += 1
                    g_minus_succ[j].append(i); in_minus[i] += 1
                # else: x-overlapping AND y-overlapping → real overlap.
                # Caller should have legalized; we silently skip.

            # V-edge: x-projections overlap AND strict V-separation.
            if x_overlap:
                if y_top[i] <= y_bot[j] + eps:
                    # i strictly BELOW j. Γ+: i precedes j; Γ-: j precedes i.
                    g_plus_succ[i].append(j); in_plus[j] += 1
                    g_minus_succ[j].append(i); in_minus[i] += 1
                elif y_top[j] <= y_bot[i] + eps:
                    # i strictly ABOVE j. Γ+: j precedes i; Γ-: i precedes j.
                    g_plus_succ[j].append(i); in_plus[i] += 1
                    g_minus_succ[i].append(j); in_minus[j] += 1

    # Topo-sort with deterministic tie-breaking. For Γ+, the canonical key
    # is (x_left + y_bot) — places macros with smaller x+y first. For Γ-,
    # use (x_left - y_top) — orders by NW-to-SE diagonal.
    plus_key = x_left + y_bot
    minus_key = x_left - y_top
    gamma_plus = _kahn_topo_sort(g_plus_succ, in_plus.copy(), plus_key)
    gamma_minus = _kahn_topo_sort(g_minus_succ, in_minus.copy(), minus_key)

    return SequencePair(
        gamma_plus=np.asarray(gamma_plus, dtype=np.int64),
        gamma_minus=np.asarray(gamma_minus, dtype=np.int64),
    )


def _kahn_topo_sort(
    succ: List[List[int]], indeg: np.ndarray, key: np.ndarray
) -> List[int]:
    """Kahn's algorithm with deterministic tie-breaking by `key[i]`.

    `key` is a 1D array of floats; smaller key values come first in the topo
    order whenever multiple nodes are ready.
    """
    import heapq
    n = len(succ)
    out: List[int] = []
    heap: List[Tuple[float, int]] = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(heap, (float(key[i]), i))
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (float(key[v]), v))
    if len(out) != n:
        stuck = [i for i in range(n) if indeg[i] > 0]
        sample = stuck[:8]
        sample_info = [f"macro {m}: indeg={int(indeg[m])} key={key[m]:.4f}" for m in sample]
        raise RuntimeError(
            f"Topo-sort failed: cycle detected (consumed {len(out)} of {n} nodes; "
            f"{len(stuck)} stuck). First {len(sample)} cycle members:\n  "
            + "\n  ".join(sample_info)
            + "\nThis means the input placement has inconsistent pair-relations — "
            f"either overlapping macros or float-boundary issue with eps."
        )
    return out


def relation_for_pair(sp: SequencePair, i: int, j: int) -> int:
    """Recover one-of-{LEFT, RIGHT, BELOW, ABOVE} for pair (i, j) from SP.

    LEFT  iff i before j in both gamma_plus and gamma_minus.
    RIGHT iff j before i in both.
    BELOW iff i before j in gamma_plus, j before i in gamma_minus.
    ABOVE iff j before i in gamma_plus, i before j in gamma_minus.
    """
    rp = sp.rank_plus()
    rm = sp.rank_minus()
    p = rp[i] < rp[j]
    m = rm[i] < rm[j]
    if p and m:
        return LEFT
    if not p and not m:
        return RIGHT
    if p and not m:
        return BELOW
    return ABOVE


def pair_relation_distance(sp_a: SequencePair, sp_b: SequencePair) -> Tuple[int, dict]:
    """Number of pairs where sp_a and sp_b have different one-of-4 relations.

    Returns (total_diff_count, breakdown_dict). The breakdown is keyed by
    (rel_a, rel_b) pair so we can see HOW the SPs differ — e.g.,
    'LEFT in A, BELOW in B' is a vertical-vs-horizontal disagreement.
    """
    n = len(sp_a.gamma_plus)
    if len(sp_b.gamma_plus) != n:
        raise ValueError(f"SP size mismatch: {n} vs {len(sp_b.gamma_plus)}")

    rp_a, rm_a = sp_a.rank_plus(), sp_a.rank_minus()
    rp_b, rm_b = sp_b.rank_plus(), sp_b.rank_minus()

    # Vectorized: compute relations for all pairs at once.
    # rel(i, j) coded as 2*p + m where p = (rp[i] < rp[j]), m = (rm[i] < rm[j]).
    # 11 -> LEFT (0), 00 -> RIGHT (1), 10 -> BELOW (2), 01 -> ABOVE (3).
    name = ["LEFT", "RIGHT", "BELOW", "ABOVE"]

    diff = 0
    breakdown: dict = {}
    # Iterate upper triangle; skip vectorize for clarity (N^2 ~ 250k for N=500).
    for i in range(n):
        for j in range(i + 1, n):
            p_a = rp_a[i] < rp_a[j]; m_a = rm_a[i] < rm_a[j]
            r_a = (LEFT if (p_a and m_a) else RIGHT if (not p_a and not m_a)
                   else BELOW if (p_a and not m_a) else ABOVE)
            p_b = rp_b[i] < rp_b[j]; m_b = rm_b[i] < rm_b[j]
            r_b = (LEFT if (p_b and m_b) else RIGHT if (not p_b and not m_b)
                   else BELOW if (p_b and not m_b) else ABOVE)
            if r_a != r_b:
                diff += 1
                key = f"{name[r_a]}->{name[r_b]}"
                breakdown[key] = breakdown.get(key, 0) + 1
    return diff, breakdown


def kendall_tau(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """Number of pair inversions between two permutations of the same items.

    Naive O(N^2). For our sizes this is sub-second.
    """
    n = len(seq_a)
    if len(seq_b) != n:
        raise ValueError("Sequences must be same length")
    pos_a = np.empty(n, dtype=np.int64)
    pos_a[seq_a] = np.arange(n)
    pos_b = np.empty(n, dtype=np.int64)
    pos_b[seq_b] = np.arange(n)
    diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = pos_a[i] - pos_a[j]
            sign_b = pos_b[i] - pos_b[j]
            if (sign_a > 0) != (sign_b > 0):
                diff += 1
    return diff


def murata_decode(
    sp: SequencePair, sizes: torch.Tensor, n_hard: int
) -> torch.Tensor:
    """Murata-Fujiyoshi compaction decoder: SP -> Cartesian positions.

    For each macro m:
      x[m] = longest path in horizontal DAG (edges weighted by source widths)
      y[m] = longest path in vertical DAG

    Result: bottom-left-compact non-overlapping packing. Positions are CENTERS.
    """
    siz = sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    # Build successor lists for x (H) and y (V) DAGs from SP relations.
    # We can derive relations directly from SP without re-running pair logic:
    # relation_for_pair(sp, i, j) tells us which DAG gets the edge.
    rp = sp.rank_plus()
    rm = sp.rank_minus()

    h_succ: List[List[int]] = [[] for _ in range(n_hard)]  # i -> j: i precedes j in x
    v_succ: List[List[int]] = [[] for _ in range(n_hard)]  # i -> j: i precedes j in y
    h_pred: List[List[int]] = [[] for _ in range(n_hard)]
    v_pred: List[List[int]] = [[] for _ in range(n_hard)]

    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            p = rp[i] < rp[j]
            m = rm[i] < rm[j]
            if p and m:
                # i LEFT of j: x DAG i->j
                h_succ[i].append(j); h_pred[j].append(i)
            elif (not p) and (not m):
                # i RIGHT of j: x DAG j->i
                h_succ[j].append(i); h_pred[i].append(j)
            elif p and (not m):
                # i BELOW j: y DAG i->j
                v_succ[i].append(j); v_pred[j].append(i)
            else:
                # i ABOVE j: y DAG j->i
                v_succ[j].append(i); v_pred[i].append(j)

    x_pos = _longest_path(n_hard, h_succ, h_pred, siz[:, 0])
    y_pos = _longest_path(n_hard, v_succ, v_pred, siz[:, 1])

    # Convert from "left edge" coords to centers (decode places macros at
    # bottom-left = predecessor's right edge; we'll add half-width to centers).
    centers = np.zeros((n_hard, 2), dtype=np.float64)
    centers[:, 0] = x_pos + siz[:, 0] / 2.0
    centers[:, 1] = y_pos + siz[:, 1] / 2.0
    return torch.tensor(centers, dtype=torch.float32)


def _longest_path(n: int, succ: List[List[int]], pred: List[List[int]],
                  weight: np.ndarray) -> np.ndarray:
    """Longest path from source to each node; node weight = own dimension.

    Each macro's left-edge coordinate = max over predecessors of (pred.left + pred.weight).
    """
    # Topological order via Kahn (in-degree from predecessors).
    indeg = np.array([len(pred[i]) for i in range(n)], dtype=np.int32)
    order: List[int] = []
    queue: List[int] = [i for i in range(n) if indeg[i] == 0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != n:
        raise RuntimeError("Cycle in SP DAG (should not happen for valid SP)")

    left_edge = np.zeros(n, dtype=np.float64)
    for u in order:
        best = 0.0
        for p in pred[u]:
            cand = left_edge[p] + weight[p]
            if cand > best:
                best = cand
        left_edge[u] = best
    return left_edge
