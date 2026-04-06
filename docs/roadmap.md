# Roadmap

Last updated: 2026-04-05
Competition deadline: May 21, 2026 (~6.5 weeks)

---

## Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | — |
| **Polyhedra (50s nav)** | **1.4921** | **-2.4%** |
| SDF v5 (init) | 1.5002 | -2.9% |

- Beat RePlAce on 3/17 benchmarks (ibm02, ibm10, ibm12)
- Congestion is 66.5% of proxy, density 29.3%, WL 4.2%
- Local navigation is at ceiling — can improve density but not congestion
- Tunneling theory developed, not yet validated experimentally

---

## Phase 1–3: DONE

- [x] SDF init + polyhedra navigation framework
- [x] Surrogate-guided search (1000+ candidates/benchmark)
- [x] Sparse LP for large benchmarks
- [x] Navigation speedup (2x), SA acceptance, cluster moves
- [x] Profiling + component breakdown + benchmark triage
- [x] Congestion diagnostic experiments (proved local moves can't reach congestion)

---

## Phase 4: Tunneling Validation (Apr 6–14)

**Goal:** Test whether the tunneling theory produces actionable results. Three experiments in order of cost.

### 4a. Soft-pair flipping (1 day)

The core hypothesis: flipping pairs with SMALL duals (soft modes) moves congestion, while flipping pairs with LARGE duals (current approach) only moves density/WL.

1. Solve LP, get duals
2. Sort pairs by |dual| ASCENDING
3. Flip softest 5-20 pairs, accept cost increases
4. Re-solve LP, evaluate real proxy
5. Compare congestion change vs hard-pair flips

**Kill gate:** If soft-pair flips don't move congestion more than hard-pair flips → theory is wrong, skip 4b/4c.

### 4b. Parametric LP continuation (3 days)

If 4a validates, trace the mountain pass:

1. Pick softest pair, set up parametric LP rotating constraint from direction A → B
2. Solve LP at t = 0.0, 0.1, ..., 1.0
3. Track proxy cost and congestion along the path
4. Look for saddle structure (cost rises then falls)

**Kill gate:** No saddle found in any parametric path → barrier is flat, not structured.

### 4c. Saddle search integration (5 days)

If 4b finds saddles:

1. Implement dimer method: uphill on softest mode, downhill on all others
2. Cross the saddle, descend into new basin
3. Run local navigation in new basin to recover density
4. Evaluate on --all

### Verification

- [ ] Soft-pair flips move congestion >1% on ibm06
- [ ] Parametric LP shows saddle structure
- [ ] At least one benchmark improves proxy cost via saddle crossing
- [ ] Avg proxy --all < 1.48

---

## Phase 5: Innovation Prize (Apr 14–May 7)

**Goal:** Document the polyhedra framework for the innovation prize ($4K), regardless of placement score.

### The narrative

1. **The decomposition theorem** — union of convex polyhedra, LP inside combinatorics
2. **What we proved empirically** — local search ceiling, congestion barrier, failed approaches
3. **The tunneling theory** — complexification, mountain pass, Potts model connection
4. **Connections to deep math** — tropical geometry, cavity method, stratified Morse theory

### Deliverables

- [ ] Competition report (PDF)
- [ ] Visualizations of polyhedra structure + tunneling paths
- [ ] Code documentation

---

## Phase 6: Polish & Submit (May 7–21)

1. Final validation — all 17 benchmarks + NG45, zero overlaps, <60s/benchmark
2. Best of: tunneling result (if Phase 4 works) or current 1.4921
3. Robustness check — 3 seeds, stable results
4. Submit placement + innovation prize report

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Soft-pair flips don't move congestion | Medium | Accept 1.49, focus on innovation prize |
| Parametric LP too slow for N=500+ | Medium | Focus on smaller benchmarks; extrapolate |
| Tunneling finds better congestion but density regresses | High | Combine: tunnel for congestion, navigate for density |
| Not enough time for both placement and innovation prize | Low | Innovation prize is parallelizable with experiments |
