# SP4: Congestion-Aware LP — Improvement Avenues

Last updated: 2026-04-14

## Current State

- LP minimizes HPWL only: `min sum_j (u_max_j - u_min_j + v_max_j - v_min_j)`
- Congestion is **66.5% of proxy cost** and the ENTIRE gap to RePlAce
- Proxy formula: `1.0*WL + 0.5*density + 0.5*congestion`
- LP is blind to 2/3 of the objective it's supposed to optimize
- Real congestion uses L-shaped routing + ABU5 (top 5% of H+V routing arrays)
- Surrogate congestion uses RUDY (bounding box demand spread uniformly)

**The core opportunity:** Nobody else has an LP solver inside a feasible-topology framework. Adding congestion awareness to the LP lets us optimize congestion WITHIN the polyhedron — a capability no other placer has.

---

## Approach 1: Iterative Net Weighting (Brenner/Vygen)

**Source:** Brenner & Vygen, "Congestion Driven Placement Framework," ISPD 2002 / IEEE TCAD 2004. Also used in CRISP, Ripple, DREAMPlace.

**Idea:** After each LP solve, compute RUDY congestion from the solution. Nets that contribute most to congestion in hot cells get higher HPWL weights. Re-solve. Over iterations, the LP preferentially shortens nets passing through congested regions.

**Algorithm:**
```
weights = [1.0] * n_nets
for round in range(n_rounds):
    result = lp.solve(assignment, net_weights=weights)
    positions = result['positions']
    
    # Compute RUDY congestion map from LP positions
    cong_map = compute_rudy(positions, nets, grid)
    
    # Identify top-5% congested cells
    threshold = np.percentile(cong_map, 95)
    hot_cells = set(np.where(cong_map > threshold)[0])
    
    # For each net, compute congestion contribution
    for j, net in enumerate(nets):
        bbox_cells = get_cells_in_bbox(net, positions)
        hot_overlap = len(bbox_cells & hot_cells)
        if hot_overlap > 0:
            weights[j] *= (1.0 + alpha * hot_overlap / len(bbox_cells))
    
    # Normalize to preserve scale
    weights = [w / np.mean(weights) for w in weights]
```

**Implementation in codebase:**
- Modify `lp.py:solve()` (line 227-234) to accept `net_weights: list[float]`
- Objective becomes: `col_cost[base+0] = w_j; col_cost[base+1] = -w_j` etc.
- Add reweighting loop in `placer.py` after initial LP solve (line 232)
- Use surrogate's RUDY computation for congestion map (already implemented)

**Parameters to tune:**
- `alpha`: weight increase factor (start with 0.5)
- `n_rounds`: number of reweighting iterations (2-3, limited by time budget)
- Whether to reweight during navigation or only at initialization

**Expected outcome:** LP positions that avoid congestion hotspots at the cost of slightly longer wirelength. Since WL is only 4.2% of cost and congestion is 66.5%, this tradeoff is hugely favorable.

**Effort:** LOW (~50-100 lines). Fully LP-compatible — only changes objective coefficients.

**Risk:** Over-weighting can destabilize the LP or push all nets to one side. Mitigation: cap individual weights at 5x, normalize mean to 1.0.

---

## Approach 2: Congestion-Proportional Separation Margins (CRISP-Inspired)

**Source:** Roy et al., "CRISP: Congestion Reduction by Iterated Spreading," ICCAD 2009. Also Ripple (Chow et al., ICCAD 2011).

**Idea:** In congested regions, increase the minimum separation between macros. This forces the LP to spread macros apart where routing is tight, creating more routing space.

**Current constraint** (`lp.py:108-162`):
```python
# Separation for pair (i,k) in direction L:
# x_k - x_i >= (w_i + w_k)/2 + eps
```

**Proposed modification:**
```python
# x_k - x_i >= (w_i + w_k)/2 + eps + margin_ik
# where margin_ik depends on congestion at the pair's midpoint
```

**Algorithm:**
1. Solve LP with standard margins → get positions
2. Compute RUDY congestion map
3. For each pair (i,k), look up congestion at midpoint
4. Set `margin_ik = beta * max(0, congestion - capacity)` for congested pairs
5. Re-solve LP with increased margins

**Implementation:**
- Modify `_add_separation_constraints()` (line 122-162) to accept per-pair margins
- Compute margins from surrogate congestion map after first LP solve
- Re-solve with updated margins

**Expected outcome:** More whitespace in congested regions → lower peak congestion. May slightly increase total wirelength (macros pushed further apart).

**Effort:** LOW (~30-50 lines). Only modifies constraint RHS values.

**Risk:** Too-large margins can make LP infeasible (macros don't fit on canvas). Mitigation: cap margins at 10% of minimum canvas dimension.

---

## Approach 3: McCormick Bounding Box Area Penalty

**Source:** Standard optimization technique for linearizing bilinear terms.

**Idea:** Congestion concentration is proportional to net bounding box AREA (not just perimeter). Area = x_span * y_span, which is bilinear. McCormick envelopes provide a linear relaxation that can be added to the LP.

**Formulation:**
For each net j, introduce variable A_j representing bbox area:
```
A_j >= x_L * y_span + x_span * y_L - x_L * y_L    (McCormick lower 1)
A_j >= x_U * y_span + x_span * y_U - x_U * y_U    (McCormick lower 2)
A_j <= x_U * y_span + x_span * y_L - x_U * y_L    (McCormick upper 1)
A_j <= x_span * y_U + x_L * y_span - x_L * y_U    (McCormick upper 2)
```
where x_span = u_max_j - u_min_j, y_span = v_max_j - v_min_j, and x_L, x_U, y_L, y_U are bounds.

**LP objective becomes:**
```
min sum_j [w_j * HPWL_j + lambda * A_j]
```

**Implementation:**
- Add n_nets auxiliary variables A_j
- Add 4 * n_nets McCormick constraints
- Add `lambda * A_j` to objective for each net
- Bounds: x_L = 0, x_U = canvas_width, y_L = 0, y_U = canvas_height

**Expected outcome:** LP penalizes nets with large bounding box area, which directly reduces routing demand concentration. Nets are encouraged to be either wide-and-short or tall-and-narrow, not sprawling.

**Effort:** MEDIUM (~100 lines). Adds variables and constraints to LP.

**Risk:** McCormick relaxation may be loose (poor approximation of true area). For nets spanning most of the canvas, the bounds are wide and the linearization is weak.

---

## Approach 4: Net-Span Penalty (Simplified Congestion Proxy)

**Idea:** A simpler alternative to McCormick: add a penalty on total net span (x_span + y_span) with higher weight for nets that are already congestion-heavy.

**Key insight:** Total RUDY demand per net IS proportional to HPWL. But what matters for ABU5 congestion is demand CONCENTRATION in hot tiles. Nets that are short but in congested areas contribute more to peak congestion than long nets in uncongested areas. So the penalty should be congestion-WEIGHTED span.

**Formulation:**
```
min sum_j [1.0 * HPWL_j + lambda * c_j * HPWL_j]
  = min sum_j [(1.0 + lambda * c_j) * HPWL_j]
```
where c_j is net j's congestion contribution (from RUDY map).

This is just Approach 1 (net weighting) with a specific formula. The distinction: here we motivate it as a linearization of the congestion objective, not just a heuristic reweighting.

**Why this IS the right linear proxy for congestion:**
- True congestion ≈ ABU5 of RUDY demand map
- RUDY demand per tile ≈ sum of (1/bbox_area) for nets covering that tile
- Minimizing span reduces bbox area, reducing per-tile demand
- Weighting by current congestion targets the nets that matter most

**Effort:** Same as Approach 1.

---

## Approach 5: Dual-Informed Congestion Targeting

**Source:** Novel combination of existing LP duals with congestion information.

**Idea:** LP duals tell us which separation constraints are expensive for HPWL. We want to know which are expensive for CONGESTION. By cross-referencing dual values with congestion map data, we can identify pairs where relaxing the constraint would reduce congestion.

**Algorithm:**
1. Solve LP, get duals and positions
2. Compute congestion map from positions
3. For each pair (i,k) with dual d_ik:
   a. Identify nets connecting macros i and k
   b. Compute those nets' contribution to congested cells
   c. Compute "congestion dual" = congestion_contribution * sign(d_ik)
4. Navigator uses congestion duals (not HPWL duals) for move selection

**Implementation:**
- Compute congestion duals after each LP solve
- Modify `DualGuidedProposer` in `moves.py:170-194` to use congestion duals
- Or blend: `effective_dual = alpha * hpwl_dual + (1-alpha) * congestion_dual`

**Expected outcome:** Navigator proposes moves that target congestion, not just wirelength. This directly addresses the finding that "local pair flips cannot reach congestion" — maybe they can, with the right pair selection.

**Effort:** MEDIUM (~100 lines). Requires congestion attribution per pair.

**Risk:** Congestion attribution is approximate (RUDY-based). May misidentify target pairs.

---

## Approach 6: LP with Congestion Feedback from Real Proxy

**Source:** DREAMPlace's iterative congestion feedback loop.

**Idea:** Use the REAL proxy cost evaluator (not just RUDY surrogate) to compute congestion maps at key checkpoints, then feed those back into LP weights.

**Algorithm:**
```
# Initial LP solve
result = lp.solve(assignment)
positions = result['positions']

# Compute REAL congestion breakdown
plc.set_positions(positions)
real_cong = get_real_congestion_map(plc)  # From PlacementCost

# Reweight based on real congestion
for j, net in enumerate(nets):
    contribution = net_congestion_contribution(net, positions, real_cong)
    weights[j] = 1.0 + gamma * contribution

# Re-solve with real-congestion-informed weights
result2 = lp.solve(assignment, net_weights=weights)
```

**Difference from Approach 1:** Uses the ACTUAL congestion evaluator (PlacementCost's L-shaped routing) instead of RUDY approximation. Much more accurate but more expensive (~50ms per evaluation).

**When to use:** At initialization (before navigation starts). The 50-100ms cost of 2-3 real congestion evaluations is negligible compared to the 50s navigation budget.

**Effort:** MEDIUM. Need to extract per-cell congestion from PlacementCost (may require modifying `objective.py`).

**Risk:** PlacementCost may not expose per-cell congestion arrays directly.

---

## Approach 7: Iterative LP-Navigate-Reweight Loop

**Idea:** Combine congestion-aware LP with navigation in an outer loop.

**Algorithm:**
```
for outer_round in range(3):
    # 1. Solve congestion-aware LP
    result = lp.solve(assignment, net_weights=weights)
    
    # 2. Navigate for density improvement (15s budget)
    nav_result = navigate(assignment, result, time_budget=15)
    
    # 3. Recompute congestion from navigated positions
    cong_map = compute_rudy(nav_result['positions'])
    
    # 4. Update weights for next round
    weights = reweight_from_congestion(cong_map, nets)
    
    # 5. Re-extract assignment (navigation may have changed topology)
    assignment = extract_assignment(nav_result['positions'])
```

**Why this helps:** Each round, the LP gets a better congestion signal from the previous round's navigated positions. The navigation gets a better starting point from the congestion-aware LP. They bootstrap each other.

**Time budget:** 3 rounds * (LP: 5s + navigate: 15s) = 60s total. Fits within the 60s/benchmark limit.

**Effort:** MEDIUM. Restructures the main `place()` method in `placer.py`.

**Risk:** Outer loop may not converge — congestion weights could oscillate. Mitigation: dampen weight updates (exponential moving average).

---

## Recommended Priority

| Priority | Approach | Effort | Impact | Rationale |
|----------|----------|--------|--------|-----------|
| 1 | Iterative net weighting | 50-100 LOC | HIGH | Strongest literature support; fully LP-native |
| 2 | Separation margins | 30-50 LOC | MEDIUM | Complementary to net weighting; easy to combine |
| 3 | Dual-informed targeting | ~100 LOC | MEDIUM | Changes navigator to target congestion directly |
| 4 | LP-Navigate-Reweight loop | ~150 LOC | HIGH | Multiplies the effect of approaches 1+2 |
| 5 | Real-proxy feedback | ~100 LOC | MEDIUM | More accurate than RUDY; use at init only |
| 6 | McCormick area penalty | ~100 LOC | LOW-MED | Theoretically sound but relaxation may be loose |
| 7 | Net-span penalty | Same as 1 | Same | Alternative motivation for the same implementation |

**Suggested sequence:** Implement 1 (net weighting) and test on ibm01/ibm06. If congestion moves >3%, add 2 (margins) and combine. Then wrap in 4 (outer loop) for the full pipeline.

---

## Implementation Entry Points

| What to modify | File | Lines | Change |
|---------------|------|-------|--------|
| LP objective weights | `lp.py` | 227-234 | Per-net weight in col_cost |
| Separation margins | `lp.py` | 122-162 | Per-pair margin in RHS |
| Weight computation | New function in `lp.py` or `surrogate.py` | — | RUDY → net weights |
| Outer loop | `placer.py` | 232-277 | Wrap LP+navigate in reweighting loop |
| Navigator dual source | `moves.py` | 170-194 | Congestion duals instead of HPWL duals |

---

## See Also

- [approach.md](approach.md) — congestion barrier analysis (section 5)
- [sp1_topology_selection.md](sp1_topology_selection.md) — better initial basin
- [sp3_surrogate_accuracy.md](sp3_surrogate_accuracy.md) — better navigation within basin
