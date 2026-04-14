# Roadmap

Last updated: 2026-04-13
Competition deadline: May 21, 2026 (~5.5 weeks)

---

## Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | --- |
| **Polyhedra (50s nav)** | **1.4867** | **-2.0%** |
| SDF v5 (init) | 1.5002 | -2.9% |

- Beat RePlAce on 3/17 benchmarks (ibm02, ibm10, ibm12)
- Congestion is 66.5% of proxy, density 29.3%, WL 4.2%
- Local navigation is at ceiling --- can improve density but not congestion
- Tunneling theory developed, not yet validated experimentally

---

Phases 1-3 complete. See results.md for history.

---

## Phase 4: Tunneling Validation (Apr 6-14)

**Window closing Apr 14.** Status: 4a (soft-pair flipping) has NOT been tested yet.

### 4a. Soft-pair flipping (1 day) --- NOT STARTED

The core hypothesis: flipping pairs with SMALL duals (soft modes) moves congestion, while flipping pairs with LARGE duals (current approach) only moves density/WL.

1. Solve LP, get duals
2. Sort pairs by |dual| ASCENDING
3. Flip softest 5-20 pairs, accept cost increases
4. Re-solve LP, evaluate real proxy
5. Compare congestion change vs hard-pair flips

**Kill gate:** If soft-pair flips don't move congestion more than hard-pair flips, theory is wrong, skip 4b/4c.

### 4b. Parametric LP continuation (3 days)

If 4a validates, trace the mountain pass:

1. Pick softest pair, set up parametric LP rotating constraint from direction A to B
2. Solve LP at t = 0.0, 0.1, ..., 1.0
3. Track proxy cost and congestion along the path
4. Look for saddle structure (cost rises then falls)

**Kill gate:** No saddle found in any parametric path, barrier is flat, not structured.

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

## Phase 4+: Actionable Implementation Ideas

Key ideas from multi-fidelity evaluation analysis, ordered by estimated impact and implementation effort.

### Miftari LP sensitivity bounds (3-5 days)

Solve one "center" LP per cluster, extract duals. For nearby LPs differing in k separation directions, check dual feasibility --- if valid, get a lower bound in microseconds via matrix-vector product. Prunes entire assignment clusters without LP solves. Expected failure mode: basis instability when switching affects active constraints.

**Minimal experiment:** Solve one LP, record duals. For 1000 nearby assignments (1-10 pair flips), compute Lagrangian bound and compare to actual LP. Characterizes whether cluster pruning is viable.

### Ejection chains (1 week)

Solves the group topology infeasibility problem that killed the cascade approach. Instead of flipping 200+ pairs simultaneously and hoping for feasibility, build new assignments incrementally --- propagate from a pivot pair through dependent pairs via BFS, maintaining a feasibility-restoring escape at every step. Evaluate LP only at promising trial depths (1, 5, 10, 20, 50).

**Minimal experiment:** From current local minimum, select highest-sensitivity pair, flip and propagate through 5-50 dependent pairs, repair via constraint propagation, evaluate at each depth.

### Spectral decomposition of constraint graph (2-3 days)

Compute weighted Laplacian using LP dual values as edge weights. Top-20 eigenvectors reveal which groups of macro pairs can be flipped quasi-independently. Spectral clustering identifies natural "flip neighborhoods" that minimize inter-group coupling --- directly addresses the circular constraint chain problem.

**Minimal experiment:** Compute weighted Laplacian, cluster, test whether cluster LPs approximate full LP cost.

### Multi-fidelity evaluation cascade (1-2 weeks)

Four-level screening architecture:
- **Level 0** (nanoseconds): Zobrist hashing for duplicate detection
- **Level 1** (microseconds): Structural feasibility filters (cycle detection, constraint propagation)
- **Level 2** (microseconds): Dual-based cluster bounds + surrogate screening
- **Level 3** (50-150ms): Partial LP (10-20% of constraints, early termination)
- **Level 4** (seconds): Full LP with warm-starting

Screens 99.9% of candidates at negligible cost, expanding effective evaluation budget from ~30 to thousands.

### Congestion-aware LP reweighting

Iterative LP with net weights adjusted by congestion feedback. Nets contributing most to congestion get higher HPWL weights, biasing the LP toward congestion-reducing placements. Lightweight addition to current LP infrastructure.

### Fix surrogate accuracy

Within-benchmark Spearman rho = 0.17 (global = 0.899). Surrogate ranks across benchmarks but not within-benchmark candidates. Needs online calibration --- fit per-benchmark correction using LP evaluations seen so far. Would directly improve navigation quality since surrogate-guided search is the core loop.

---

## Phase 5: Innovation Prize (Apr 14-May 7)

**Goal:** Document the polyhedra framework for the innovation prize ($4K), regardless of placement score.

### The narrative

1. **The decomposition theorem** --- union of convex polyhedra, LP inside combinatorics
2. **What we proved empirically** --- local search ceiling, congestion barrier, failed approaches
3. **The tunneling theory** --- complexification, mountain pass, Potts model connection
4. **Connections to deep math** --- tropical geometry, cavity method, stratified Morse theory

### Deliverables

- [ ] Competition report (PDF)
- [ ] Visualizations of polyhedra structure + tunneling paths
- [ ] Code documentation

---

## Phase 6: Polish & Submit (May 7-21)

1. Final validation --- all 17 benchmarks + NG45, zero overlaps, <60s/benchmark
2. Best of: tunneling result (if Phase 4 works) or current 1.4867
3. Robustness check --- 3 seeds, stable results
4. Submit placement + innovation prize report

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Soft-pair flips don't move congestion | Medium | Accept ~1.49, focus on innovation prize |
| Ejection chains still hit infeasibility | Medium | Use spectral decomposition to pick quasi-independent neighborhoods |
| Surrogate accuracy too low for effective navigation | High | Online calibration; multi-fidelity cascade bypasses surrogate |
| Parametric LP too slow for N=500+ | Medium | Focus on smaller benchmarks; extrapolate |
| Not enough time for both placement and innovation prize | Low | Innovation prize is parallelizable with experiments |
| Phase 4 tunneling window closes without testing 4a | High | Prioritize soft-pair experiment immediately |

---

## See Also

- [approach.md](approach.md) --- methodology and architecture
- [results.md](results.md) --- experiment history and per-benchmark results
- [theory.md](theory.md) --- theoretical foundations and frameworks
