# Polyhedra Navigation Phase 2: Closing the Gap

Last updated: 2026-04-04

**Goal:** Close the 2.9% gap between polyhedra navigation (1.4997) and RePlAce (1.4578).
**Timeline:** 1-2 weeks. Three workstreams, prioritized by expected impact.

---

## Current bottleneck diagnosis

| Bottleneck | Evidence | Impact |
|---|---|---|
| Search budget (~20 candidates/benchmark) | `compute_proxy_cost` takes ~1s; 30s navigation budget | Cannot explore enough of the discrete space |
| Single-pair flips too local | 0.1% improvement after 5 experiments | Trapped in SDF's topology basin |
| LP optimizes HPWL only | 30% WL slack exists but exploiting it worsens density/congestion | Duals guide toward wirelength, not proxy cost |

All three compound: slow evaluation + small moves + wrong ranking signal = stalled search.

---

## Workstream 1: Fast proxy surrogate (highest priority)

**Why:** 1000x speedup in candidate evaluation unlocks the entire search. This is the single
highest-leverage change. Going from 20 to 20,000 candidate evaluations per benchmark
transforms the algorithm from "barely exploring" to "properly searching."

### Design

Incremental grid-based surrogate that estimates proxy cost delta for a candidate flip:

1. **Density surrogate**
   - Maintain a density grid (32x32 or 64x64) over the canvas
   - Each cell stores total macro area overlapping that cell
   - On a candidate flip (2 macros move), update only affected cells
   - Density penalty = mean of top-10% cell values (matches `objective.py` metric)

2. **Congestion surrogate (RUDY-style)**
   - For each net, compute bounding box and distribute routing demand uniformly across crossed grid edges
   - On a candidate flip, recompute only nets connected to the moved macros
   - Congestion = mean of top-10% horizontal + vertical edge utilizations

3. **Wirelength surrogate**
   - HPWL is cheap to compute incrementally — just recompute bounding boxes of affected nets

4. **Composite surrogate**
   - `proxy_est = wl + 0.5 * density + 0.5 * congestion` (matches official weights)
   - **Filter-then-verify**: evaluate all candidates with surrogate, verify top-5 with real `compute_proxy_cost`

### Target performance

| Metric | Current | Target |
|---|---|---|
| Candidate evaluation time | ~1s (full proxy) | ~0.1ms (surrogate) |
| Candidates evaluated per benchmark | ~20 | ~20,000+ |
| Surrogate correlation with true proxy | N/A | >0.8 rank correlation |

### Implementation plan

1. Build `GridSurrogate` class with `init_from_placement()` and `evaluate_delta(old_pos, new_pos, affected_macros)` methods
2. Validate surrogate accuracy: compare surrogate ranking vs true proxy ranking on 100 random flips per benchmark
3. Integrate into Navigator: surrogate filters candidates, top-k verified with real proxy
4. Tune grid resolution (32x32 vs 64x64) and top-k threshold

### Acceptance criteria

- Surrogate evaluates in <1ms per candidate
- Rank correlation with true proxy >0.7 on random flips
- Navigator evaluates 1000+ candidates per benchmark within time budget

---

## Workstream 2: Cluster moves (medium priority)

**Why:** Single-pair flips yield 0.1% improvement. To escape local optima, we need moves
that rearrange multiple macros simultaneously. This is the difference between hill-climbing
and actually exploring the topology space.

### Design

Three move types, increasing in complexity:

1. **Net-correlated flips**
   - Identify pairs sharing high-fanout nets (duals often cluster around bottleneck nets)
   - Flip 2-5 related pairs simultaneously
   - If the resulting polyhedron is feasible, evaluate with surrogate

2. **Row/column block swaps**
   - Identify macros in a spatial row (similar y-coordinates, sequential x-order)
   - Reverse the left-right ordering of a contiguous block (2-4 macros)
   - This flips O(k^2) pair relations but preserves above/below structure

3. **Swendsen-Wang cluster moves** (if time permits)
   - Build a graph where edges connect pairs with |dual| above threshold
   - Identify connected components
   - Flip all pairs in a component with some probability
   - Motivated by statistical physics: breaks out of metastable states

### Implementation plan

1. Add `propose_cluster_flip()` to NeighborGenerator using net-correlated strategy
2. Add feasibility check: after flipping multiple pairs, verify polyhedron is non-empty via LP
3. Evaluate cluster moves with surrogate (from Workstream 1)
4. Track acceptance rates: if cluster moves have <1% feasibility, fall back to row/column swaps

### Acceptance criteria

- Cluster moves find improvements that single-pair flips miss
- At least one cluster move type has >5% feasibility rate
- Measurable proxy cost reduction beyond single-pair navigation

---

## Workstream 3: Density-aware candidate ranking (lower priority)

**Why:** LP duals only reflect wirelength pressure. Congestion is 50% of proxy cost and
the main gap to RePlAce. Ranking candidates by wirelength duals alone misses moves that
would reduce congestion.

### Design

Replace pure dual-magnitude ranking with a blended score:

```
candidate_score = alpha * |lp_dual| + beta * congestion_delta_est + gamma * density_delta_est
```

Where:
- `|lp_dual|` = LP dual variable magnitude (current signal)
- `congestion_delta_est` = estimated congestion change from surrogate (Workstream 1)
- `density_delta_est` = estimated density change from surrogate

This requires the surrogate from Workstream 1, so it's naturally sequenced after it.

### Implementation plan

1. After surrogate is built, compute surrogate proxy delta for top-100 dual-ranked candidates
2. Re-rank by surrogate proxy delta instead of pure dual magnitude
3. Compare: does surrogate-ranked navigation find better moves than dual-ranked?

### Acceptance criteria

- Surrogate-ranked candidates yield better proxy improvements than dual-ranked
- Navigation finds moves that reduce congestion, not just wirelength

---

## Sequencing

```
Week 1:  [Workstream 1: Surrogate] ──────────────────────────────►
         Build surrogate → validate accuracy → integrate into navigator

Week 2:  [Workstream 2: Cluster moves] ──────►  [Workstream 3: Ranking] ──►
         Net-correlated flips → row swaps       Blend duals + surrogate
```

Workstream 1 is a prerequisite for Workstream 3 and strongly benefits Workstream 2
(feasibility checking is much faster with surrogate).

---

## Success criteria

| Milestone | Target | Measurement |
|---|---|---|
| Surrogate working | Correlation >0.7 | Compare rankings on 100 random flips |
| Search budget unlocked | 1000+ candidates/benchmark | Count evaluations in navigator log |
| Cluster moves working | >5% feasibility rate | Track accept/reject in navigator |
| Proxy cost reduction | avg < 1.48 on --fast | `uv run evaluate ... --fast --json` |
| Full validation | avg < 1.46 on --all | `uv run evaluate ... --all --json` |
| **Champion** | **avg < 1.4578 on --all** | **Beats RePlAce** |

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Surrogate too inaccurate (rank corr <0.5) | Medium | Increase grid resolution; use surrogate only for filtering, not final decision |
| Cluster moves have <1% feasibility | Medium-high | Fall back to row/column swaps which preserve more structure |
| Congestion reduction requires fundamentally different topology | Medium | Try random restarts from different initializations (not just SDF) |
| LP solve too slow on ibm18 for warm-start iteration | Low | Already measured at ~9s cold; warm-start should be <1s |
| Time budget insufficient even with surrogate | Low | Reduce nav_iters, focus budget on most promising benchmarks |
