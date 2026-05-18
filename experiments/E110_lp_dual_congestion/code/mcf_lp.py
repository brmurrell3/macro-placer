"""Subset multi-commodity flow LP → per-gcell capacity dual prices.

The canonical proxy's congestion is built from per-net L-shaped trace-routes
on the gcell grid (see IncrementalProxyEvaluator._net_cong_contrib_flat).
Smooth RUDY approximates this with bbox-uniform demand spread, which has
been shown (E88, E95, E98) to have a gradient direction misaligned with
canonical on hard benches.

This module solves a small multi-commodity flow LP that *exactly* uses
canonical routing topology but lets each 2-pin net choose between
patterns L1 (source-row + sink-col) and L2 (source-col + sink-row).
The LP capacity duals π are valid first-order subgradients of canonical
congestion at the current placement — Magnanti & Wong (1981), Bertsekas
network flow.

Returns LPDuals(pi_h, pi_v) gridded over [grid_row, grid_col]; non-binding
gcells are zero. Falls back to status="timeout"/"infeasible"/"fallback"
on solver failure — callers must handle gracefully.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass
class LPDuals:
    pi_h: np.ndarray   # [grid_row, grid_col]; H capacity dual price per gcell
    pi_v: np.ndarray   # [grid_row, grid_col]; V capacity dual price per gcell
    solver_wall_s: float
    status: str        # "ok" | "timeout" | "infeasible" | "no_binding" | "error"
    n_vars: int
    n_constraints: int
    n_subset_nets: int = 0
    info: str = ""


# ── Per-net pattern demand extraction ────────────────────────────────────────


def _two_pin_patterns(
    r_s: int, c_s: int, r_t: int, c_t: int, weight: float
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
           List[Tuple[int, int]], List[Tuple[int, int]]]:
    """For a 2-pin net (r_s, c_s) → (r_t, c_t), return four lists of
    (gcell_r, gcell_c) pairs:
        L1_h: H gcells along source row r_s
        L1_v: V gcells along sink col c_t
        L2_h: H gcells along sink row r_t
        L2_v: V gcells along source col c_s
    Each gcell contributes `weight` demand. Used to build the LP coefficient
    matrix and to compute fixed demand from non-subset nets.
    """
    cmin = min(c_s, c_t)
    cmax = max(c_s, c_t)
    rmin = min(r_s, r_t)
    rmax = max(r_s, r_t)

    L1_h = [(r_s, c) for c in range(cmin, cmax)]
    L1_v = [(r, c_t) for r in range(rmin, rmax)]
    L2_h = [(r_t, c) for c in range(cmin, cmax)]
    L2_v = [(r, c_s) for r in range(rmin, rmax)]
    return L1_h, L1_v, L2_h, L2_v


def _net_source_sink_gcells(evaluator, net_idx: int) -> Optional[Tuple[int, int, int, int, float]]:
    """Return (r_s, c_s, r_t, c_t, weight) for a 2-pin net (after gcell
    deduplication), or None if the net has fewer than 2 distinct gcells
    OR more than 2 distinct gcells (multi-pin nets are handled as fixed
    demand in the LP).
    """
    pins = evaluator.net_pins[net_idx]
    if len(pins) < 2:
        return None
    xs = evaluator.pin_x[pins]
    ys = evaluator.pin_y[pins]
    cols = torch.floor(xs / evaluator.grid_width).clamp_(0, evaluator.grid_col - 1).long().tolist()
    rows = torch.floor(ys / evaluator.grid_height).clamp_(0, evaluator.grid_row - 1).long().tolist()

    # Deduplicate gcells, preserving order (source = first pin's gcell).
    unique: List[Tuple[int, int]] = []
    seen = set()
    for g in zip(rows, cols):
        if g not in seen:
            seen.add(g)
            unique.append(g)
    if len(unique) != 2:
        return None  # not 2-distinct-gcell; let it contribute fixed demand
    weight = float(evaluator.net_weight[net_idx])
    r_s, c_s = unique[0]
    r_t, c_t = unique[1]
    return r_s, c_s, r_t, c_t, weight


# ── Full demand snapshot (canonical, from current placement) ─────────────────


def _snapshot_canonical_demand(evaluator) -> Tuple[np.ndarray, np.ndarray]:
    """Build full per-gcell H/V demand grids using canonical routing
    (`_net_cong_contrib_flat`) over all nets at the current placement.
    Returns (demand_h, demand_v) of shape [grid_row, grid_col] each.
    """
    R, C = evaluator.grid_row, evaluator.grid_col
    demand_h = np.zeros((R, C), dtype=np.float64)
    demand_v = np.zeros((R, C), dtype=np.float64)
    n_nets = len(evaluator.net_pins)
    for net_idx in range(n_nets):
        h_cells, h_vals, v_cells, v_vals = evaluator._net_cong_contrib_flat(net_idx)
        for cell, w in zip(h_cells, h_vals):
            r, c = divmod(cell, C)
            demand_h[r, c] += w
        for cell, w in zip(v_cells, v_vals):
            r, c = divmod(cell, C)
            demand_v[r, c] += w
    return demand_h, demand_v


# ── LP build + solve ─────────────────────────────────────────────────────────


def build_and_solve_subset_mcf(
    evaluator,
    *,
    top_edge_frac: float = 0.20,
    top_k_nets: int = 2000,
    wall_budget_s: float = 5.0,
) -> LPDuals:
    """Subset multi-commodity flow LP on canonical-aligned routing topology.

    1. Snapshot per-gcell demand from canonical routes (all nets).
    2. Select top-frac binding gcells (those with highest utilization).
    3. Select top-K 2-pin nets crossing those binding gcells; rest contribute
       fixed demand.
    4. Build LP:
         min Σ s_e
         s.t. f_n_L1 + f_n_L2 = 1   ∀ subset net n     [n_subset_nets eq]
              demand(f) + c_fixed ≤ cap + s   ∀ binding gcell e   [n_binding ineq]
              f ≥ 0,  s ≥ 0
       where demand(f) = Σ_n Σ_p f_n_p · w_n_p_e.
    5. Solve via scipy linprog HiGHS dual simplex. Extract π from
       res.ineqlin.marginals (HiGHS returns negative for ≤ constraints).
    """
    t0 = time.perf_counter()
    R, C = evaluator.grid_row, evaluator.grid_col
    cap_h = float(evaluator.grid_h_routes)
    cap_v = float(evaluator.grid_v_routes)

    # 1. Canonical demand snapshot.
    demand_h_canon, demand_v_canon = _snapshot_canonical_demand(evaluator)

    # 2. Binding gcell selection by utilization.
    util_h = demand_h_canon / max(cap_h, 1e-12)
    util_v = demand_v_canon / max(cap_v, 1e-12)
    util_all = np.concatenate([util_h.ravel(), util_v.ravel()])
    if util_all.max() <= 0:
        return LPDuals(
            pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
            solver_wall_s=time.perf_counter() - t0,
            status="no_binding", n_vars=0, n_constraints=0,
            info="all gcells have zero demand",
        )
    thresh = float(np.quantile(util_all, 1.0 - top_edge_frac))
    binding_h_mask = util_h >= thresh
    binding_v_mask = util_v >= thresh

    binding_h_idx = np.argwhere(binding_h_mask)  # [(r, c), ...]
    binding_v_idx = np.argwhere(binding_v_mask)
    n_binding_h = len(binding_h_idx)
    n_binding_v = len(binding_v_idx)
    if n_binding_h + n_binding_v == 0:
        return LPDuals(
            pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
            solver_wall_s=time.perf_counter() - t0,
            status="no_binding", n_vars=0, n_constraints=0,
            info=f"thresh={thresh:.3f} but no binding cells",
        )

    # Build per-gcell index in the constraint vector.
    # H constraints first, then V constraints. Index by (r, c) -> ineq row.
    h_cell_to_ineq = {(int(r), int(c)): i for i, (r, c) in enumerate(binding_h_idx)}
    v_cell_to_ineq = {(int(r), int(c)): n_binding_h + i for i, (r, c) in enumerate(binding_v_idx)}
    n_ineq = n_binding_h + n_binding_v

    # 3. Candidate 2-pin nets (filter: must cross at least one binding gcell).
    n_nets = len(evaluator.net_pins)
    candidate_nets: List[Tuple[int, float, int, int, int, int, float]] = []
    # (net_idx, util_score, r_s, c_s, r_t, c_t, weight)
    for net_idx in range(n_nets):
        spec = _net_source_sink_gcells(evaluator, net_idx)
        if spec is None:
            continue
        r_s, c_s, r_t, c_t, weight = spec
        # Net crosses a binding gcell if any of its L1 or L2 gcells are binding.
        L1_h, L1_v, L2_h, L2_v = _two_pin_patterns(r_s, c_s, r_t, c_t, weight)
        crosses = False
        score = 0.0
        for r, c in L1_h:
            if (r, c) in h_cell_to_ineq:
                crosses = True
                score += util_h[r, c]
        for r, c in L1_v:
            if (r, c) in v_cell_to_ineq:
                crosses = True
                score += util_v[r, c]
        for r, c in L2_h:
            if (r, c) in h_cell_to_ineq:
                crosses = True
                score += util_h[r, c]
        for r, c in L2_v:
            if (r, c) in v_cell_to_ineq:
                crosses = True
                score += util_v[r, c]
        if not crosses:
            continue
        candidate_nets.append((net_idx, score, r_s, c_s, r_t, c_t, weight))
        # Wall budget check during candidate scan (can be slow for 50k nets).
        if time.perf_counter() - t0 > wall_budget_s:
            return LPDuals(
                pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
                solver_wall_s=time.perf_counter() - t0,
                status="timeout", n_vars=0, n_constraints=0,
                info=f"candidate scan timeout after {len(candidate_nets)} nets",
            )

    candidate_nets.sort(key=lambda t: -t[1])
    subset_nets = candidate_nets[:top_k_nets]
    n_subset = len(subset_nets)
    if n_subset == 0:
        return LPDuals(
            pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
            solver_wall_s=time.perf_counter() - t0,
            status="no_binding", n_vars=0, n_constraints=0,
            info="no 2-pin nets cross binding gcells",
        )

    subset_net_ids = {nid for (nid, *_rest) in subset_nets}

    # 4a. Fixed-demand contribution from non-subset nets at binding gcells.
    # For non-subset 2-pin nets and all multi-pin nets, use canonical (L1)
    # demand from the canonical snapshot, MINUS the L1 contribution of
    # subset nets (which is variable in the LP).
    c_fixed_h = np.zeros(n_binding_h, dtype=np.float64)
    c_fixed_v = np.zeros(n_binding_v, dtype=np.float64)
    # Start with full canonical demand, then subtract subset nets' canonical
    # contribution (their flow becomes a variable).
    for i, (r, c) in enumerate(binding_h_idx):
        c_fixed_h[i] = demand_h_canon[r, c]
    for i, (r, c) in enumerate(binding_v_idx):
        c_fixed_v[i] = demand_v_canon[r, c]
    # Subtract subset nets' L1 contributions (their canonical pattern).
    for (net_idx, _score, r_s, c_s, r_t, c_t, weight) in subset_nets:
        L1_h, L1_v, _L2_h, _L2_v = _two_pin_patterns(r_s, c_s, r_t, c_t, weight)
        for (r, c) in L1_h:
            if (r, c) in h_cell_to_ineq:
                c_fixed_h[h_cell_to_ineq[(r, c)]] -= weight
        for (r, c) in L1_v:
            if (r, c) in v_cell_to_ineq:
                c_fixed_v[v_cell_to_ineq[(r, c)] - n_binding_h] -= weight

    # 4b. LP variables.
    # First n_f = 2 * n_subset flow vars; then n_ineq slack vars.
    n_f = 2 * n_subset
    n_vars = n_f + n_ineq

    # 4c. Equality constraints: f_n_L1 + f_n_L2 = 1 per net.
    eq_rows = []
    eq_cols = []
    eq_vals = []
    for k in range(n_subset):
        eq_rows.append(k); eq_cols.append(2 * k);     eq_vals.append(1.0)
        eq_rows.append(k); eq_cols.append(2 * k + 1); eq_vals.append(1.0)
    A_eq = coo_matrix(
        (eq_vals, (eq_rows, eq_cols)), shape=(n_subset, n_vars)
    ).tocsr()
    b_eq = np.ones(n_subset, dtype=np.float64)

    # 4d. Inequality constraints: demand_e - s_e ≤ cap_e - c_fixed_e.
    # For each binding gcell e, accumulate coefficients on f variables
    # from each subset net's L1 and L2 contributions to e.
    ub_rows: List[int] = []
    ub_cols: List[int] = []
    ub_vals: List[float] = []
    for k, (net_idx, _score, r_s, c_s, r_t, c_t, weight) in enumerate(subset_nets):
        L1_h, L1_v, L2_h, L2_v = _two_pin_patterns(r_s, c_s, r_t, c_t, weight)
        # L1 contributions go onto column 2k.
        for (r, c) in L1_h:
            if (r, c) in h_cell_to_ineq:
                ub_rows.append(h_cell_to_ineq[(r, c)])
                ub_cols.append(2 * k)
                ub_vals.append(weight)
        for (r, c) in L1_v:
            if (r, c) in v_cell_to_ineq:
                ub_rows.append(v_cell_to_ineq[(r, c)])
                ub_cols.append(2 * k)
                ub_vals.append(weight)
        # L2 contributions go onto column 2k + 1.
        for (r, c) in L2_h:
            if (r, c) in h_cell_to_ineq:
                ub_rows.append(h_cell_to_ineq[(r, c)])
                ub_cols.append(2 * k + 1)
                ub_vals.append(weight)
        for (r, c) in L2_v:
            if (r, c) in v_cell_to_ineq:
                ub_rows.append(v_cell_to_ineq[(r, c)])
                ub_cols.append(2 * k + 1)
                ub_vals.append(weight)
    # Slack: coefficient -1 on the corresponding ineq row.
    for e in range(n_ineq):
        ub_rows.append(e)
        ub_cols.append(n_f + e)
        ub_vals.append(-1.0)
    A_ub = coo_matrix(
        (ub_vals, (ub_rows, ub_cols)), shape=(n_ineq, n_vars)
    ).tocsr()
    b_ub = np.empty(n_ineq, dtype=np.float64)
    for i in range(n_binding_h):
        b_ub[i] = cap_h - c_fixed_h[i]
    for j in range(n_binding_v):
        b_ub[n_binding_h + j] = cap_v - c_fixed_v[j]

    # 4e. Objective: minimize Σ s_e (unit-weight slack).
    # Calibration showed unit weights give Spearman +0.32; u^2 weights
    # over-amplify a few extreme gcells and drop Spearman to +0.14.
    # Linear is empirically better — duals are binary (1 on binding cells)
    # but the macro score aggregates net weights over many binding cells,
    # which produces a useful ranking despite the binary π.
    cost = np.zeros(n_vars, dtype=np.float64)
    cost[n_f:] = 1.0

    # 4f. Bounds: flow f_n_p ∈ [0, 1], slack s_e ∈ [0, ∞).
    bounds = [(0.0, 1.0)] * n_f + [(0.0, None)] * n_ineq

    elapsed = time.perf_counter() - t0
    remaining = wall_budget_s - elapsed
    if remaining <= 0:
        return LPDuals(
            pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
            solver_wall_s=elapsed,
            status="timeout", n_vars=n_vars, n_constraints=n_subset + n_ineq,
            info="ran out of time before solve",
        )

    # 5. Solve.
    try:
        res = linprog(
            cost,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs-ds",
            options={"time_limit": float(remaining), "presolve": True},
        )
    except Exception as exc:
        return LPDuals(
            pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
            solver_wall_s=time.perf_counter() - t0,
            status="error",
            n_vars=n_vars, n_constraints=n_subset + n_ineq,
            info=f"linprog raised: {exc!r}",
            n_subset_nets=n_subset,
        )

    if not res.success:
        return LPDuals(
            pi_h=np.zeros((R, C)), pi_v=np.zeros((R, C)),
            solver_wall_s=time.perf_counter() - t0,
            status="infeasible",
            n_vars=n_vars, n_constraints=n_subset + n_ineq,
            info=f"status={res.status}: {res.message}",
            n_subset_nets=n_subset,
        )

    # 6. Extract duals. HiGHS gives marginals; for ≤ constraints these are
    # ≤ 0 (a binding capacity constraint has a non-positive marginal indicating
    # cost decrease if RHS increases). We negate so π_e ≥ 0 = shadow price.
    marginals = res.ineqlin.marginals  # length n_ineq
    pi_e = -marginals  # ≥ 0
    pi_h = np.zeros((R, C), dtype=np.float64)
    pi_v = np.zeros((R, C), dtype=np.float64)
    for i, (r, c) in enumerate(binding_h_idx):
        pi_h[r, c] = max(float(pi_e[i]), 0.0)
    for j, (r, c) in enumerate(binding_v_idx):
        pi_v[r, c] = max(float(pi_e[n_binding_h + j]), 0.0)

    return LPDuals(
        pi_h=pi_h, pi_v=pi_v,
        solver_wall_s=time.perf_counter() - t0,
        status="ok",
        n_vars=n_vars, n_constraints=n_subset + n_ineq,
        n_subset_nets=n_subset,
        info=f"obj={res.fun:.4f} binding_h={n_binding_h} binding_v={n_binding_v}",
    )


# ── Macro scoring ────────────────────────────────────────────────────────────


def score_macros_by_duals(
    evaluator,
    hard_movable: List[int],
    duals: LPDuals,
) -> List[Tuple[int, float]]:
    """For each movable macro, sum LP-dual prices over every gcell every net
    touching that macro contributes to under canonical routing. High score =
    macro is incident on expensive (capacity-binding) gcells; high-priority
    LNS destroy target.
    """
    C = evaluator.grid_col
    scores: List[Tuple[int, float]] = []
    for m in hard_movable:
        # macro_to_nets[m] is a LongTensor of unique net indices touching macro m.
        nets = evaluator.macro_to_nets[m]
        if isinstance(nets, torch.Tensor):
            nets_iter = nets.tolist()
        else:
            nets_iter = list(nets)
        s = 0.0
        for net_idx in nets_iter:
            h_cells, h_vals, v_cells, v_vals = evaluator._net_cong_contrib_flat(int(net_idx))
            for cell, w in zip(h_cells, h_vals):
                r, c = divmod(cell, C)
                s += w * float(duals.pi_h[r, c])
            for cell, w in zip(v_cells, v_vals):
                r, c = divmod(cell, C)
                s += w * float(duals.pi_v[r, c])
        scores.append((int(m), s))
    scores.sort(key=lambda t: -t[1])
    return scores
