# Roadmap

Last updated: 2026-04-05
Competition deadline: May 21, 2026 (~6.5 weeks)

---

## Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | — |
| **Polyhedra Navigation (ours)** | **1.4890** (300s nav) / **1.4921** (50s nav) | **-2.1%** / **-2.4%** |
| SDF v5 (init topology) | 1.5002 | -2.9% |
| Will's SA seed | 1.5338 | -5.2% |

- Beat RePlAce on 3 benchmarks (ibm02, ibm10, ibm12)
- **Profiling complete** — congestion is 66.5% of proxy, density 29.3%, WL 4.2%
- Surrogate formula discrepancies were investigated and found to be non-issues
- RUDY vs ground truth routing model mismatch is the real surrogate accuracy limit
- 4 benchmarks get 0 improvements from navigation (stuck in local optima)
- ibm10 LP infeasible (308K pairs) — returns init, already winning vs RePlAce

## Success Criteria

| Metric | Current | Target | Champion |
|--------|---------|--------|----------|
| Avg proxy --all | 1.4921 (50s) / 1.4890 (300s) | <1.46 | <1.4578 |
| Benchmarks beating RePlAce | 3/17 | 6+ | 10+ |
| Zero overlaps | Yes | Yes | Yes |
| Runtime / benchmark | ~60s | <60s | <60s |

**Time budget constraint:** The 1hr competition limit is for the hidden NG45 test case (much larger). IBM benchmarks must run fast (~50-60s) as they are the public evaluation. Do not trade speed for quality on IBM — improvements must come from better algorithms, not more time.

---

## Phase 1: Speedup (DONE)

- [x] Surrogate-only navigation, vectorized overlap check, adaptive time budget
- [x] All 17 benchmarks <60s, zero overlaps, <1% quality loss

## Phase 2: Profiling & Diagnostics (DONE)

- [x] Component breakdown, benchmark triage, surrogate accuracy, convergence, timing
- Key findings in `results/profiling/` — see below for how they drive the remaining phases

### What profiling revealed

| Finding | Implication |
|---------|-------------|
| Congestion = 66.5% of proxy | All improvement must come from congestion reduction |
| Surrogate RUDY ≠ ground truth routing | Within-benchmark candidate ranking is unreliable (rho=0.17) |
| Surrogate formulas match ground truth | Phase 3's planned formula fixes are moot |
| 4 benchmarks get 0 improvements | Single-pair flips can't escape SDF topology basin |
| `num_nets` predicts gap (rho=0.958) | High-connectivity benchmarks are structurally harder |
| ibm10 LP infeasible (308K pairs) | Need sparse LP for large benchmarks |
| Search not plateauing at 300s | Need faster iterations, not more time (budget is ~50s) |

---

## Phase 3: Quick Wins (Apr 5–10)

**Goal:** Harvest easy improvements before the hard architecture work.

### Actions

1. **Sparse LP for large benchmarks** — only include pairs within margin of their min separation distance (skip pairs far apart that are trivially satisfied). Should make ibm10/12/14 LP feasible in <15s, freeing time for navigation within the ~50s budget.
2. **Verification with real proxy** — after surrogate accepts a move, do one real `compute_proxy_cost` as a safety check. Reject moves where surrogate improvement doesn't match reality. Costs ~1s per accepted move but prevents ibm15/ibm17-style regressions.
3. **Stale dual refresh** — when navigation stalls (20+ iterations with no improvement), re-solve LP to get fresh duals instead of just incrementing stale counter.
4. **Navigation speed** — profile the per-iteration cost. Each navigation iteration takes ~6s on average (300s / 50 iterations). Identify and optimize the bottleneck (projection? surrogate eval? candidate generation?) to get more iterations per second.

### Verification

- [x] ibm10 LP feasible in <18s (sparse margin=1.0, was infeasible at 60s)
- [ ] ibm15/ibm17 no longer regress with navigation (DEFERRED — real proxy verification too expensive at ~5s/eval)
- [x] Navigation throughput: ~0.4 iters/second (was ~0.15). 16 improvements in 45s (was 9 in 45s)
- [ ] Avg proxy --all < 1.485 (at 1.4921 with 50s nav; 1.4890 with 300s nav)

### Kill gate

None — each action is independent and low-risk.

---

## Phase 4: Escape Local Optima (Apr 10–21)

**Goal:** Break out of the SDF topology basin. This is the core problem — 4 benchmarks get zero improvements because no single-pair flip helps.

### 4a. Larger moves (Apr 10–14)

The current search only flips 1-5 pairs at a time. For benchmarks where this doesn't help, we need structurally different topologies.

1. **Swendsen-Wang cluster flips** — identify connected components in the constraint graph (pairs sharing macros), flip entire components. Changes 10-50 pairs simultaneously.
2. **Random restarts with perturbation** — perturb the SDF init positions (add Gaussian noise to 10-20% of macros), re-extract assignment, navigate from there. Different init → different topology basin.
3. **Simulated annealing on surrogate** — accept worse moves with Boltzmann probability `exp(-delta/T)`. Temperature schedule over the nav budget. Escapes shallow local optima.

### 4b. Hierarchical search (Apr 14–21)

If 4a is insufficient, build the multi-fidelity cascade.

1. **Spectral clustering** — group macros by netlist connectivity (k=8-16). Identify which inter-group relations matter most.
2. **Group-level topology enumeration** — enumerate L/R/A/B assignments for group pairs. Filter by geometric feasibility + HPWL bound.
3. **Topology expansion** — expand promising group topologies to macro-level, keeping intra-group structure from SDF.
4. **Cascade integration** — group screen → surrogate eval → LP solve → verify. Target: 5000+ topologies explored, 5-10 expensive evaluations.

### Verification

- [ ] At least 2 of {ibm03, ibm06, ibm16} improve with larger moves
- [ ] Avg proxy --all < 1.47
- [ ] Beat RePlAce on >= 5/17 benchmarks

### Kill gate

- 4a fails (no benchmark improves): skip to 4b hierarchical
- 4b feasibility rate <1% at all group counts: try 3D lifting (Phase 5)
- Avg proxy not below 1.47 after full Phase 4: pivot to Phase 5

---

## Phase 5: Congestion-Aware Search (Apr 21 – May 7)

**Goal:** Congestion is 66.5% of proxy. Make the search directly target it.

### Actions

1. **Congestion-delta candidate ranking** — rank candidates by surrogate `delta_congestion` instead of LP dual magnitude. Duals only reflect HPWL (4.2% of proxy). Surrogate congestion deltas directly target the dominant cost component.
2. **Congestion-aware init** — place high-fanout macros (most nets) first, spread them to reduce routing demand. Current SDF init optimizes density, not congestion.
3. **Multi-start** — run 2-3 random perturbations of the init placement, navigate from each, keep the best. General-purpose way to escape local optima without benchmark-specific tuning.

### Verification

- [ ] Avg proxy --all < 1.46 (beats RePlAce)
- [ ] Beat RePlAce on >= 6/17 benchmarks

### Kill gate

Avg proxy not below 1.46 → accept current score, focus on submission quality + innovation prize.

---

## Phase 6: Contingency & Extensions (May 7–14)

**Goal:** If Phases 4-5 insufficient, try structural changes. Prepare innovation prize.

### Contingency actions (pick based on diagnosis)

- **3D lifting** — add synthetic z-dimension during search, anneal to 2D. Mitigates low feasibility between polyhedra.
- **Disjunctive programming** — tighter LP relaxation (Balas/Kronqvist P-split). Better bounds → better duals.
- **Tropical gradient descent** — HPWL is tropical polynomial. May find better descent directions than LP duals.

### Innovation prize

- Write competition report documenting polyhedra framework
- Reference `docs/novel-mathematical-machinery.pdf`
- Prepare visualizations

### Verification

- [ ] Avg proxy --all < 1.45 (champion-level)
- [ ] Innovation prize submission complete

---

## Phase 7: Polish & Submit (May 14–21)

**Goal:** Final tuning, robustness, and submission.

### Actions

1. **Final validation** — all 17 benchmarks + NG45 hidden set, verify zero overlaps, timing <60s per IBM benchmark
2. **Robustness check** — run 3 seeds, verify results are stable
3. **Submit**

### Verification

- [ ] All 17 IBM benchmarks: zero overlaps, <60s each
- [ ] Avg proxy --all is personal best
- [ ] Submission passes validation

---

## Risk Register

| Risk | Phase | Likelihood | Mitigation |
|------|-------|------------|------------|
| Larger moves don't help (topology truly optimal) | 4a | Medium | Move to hierarchical (4b) or per-benchmark (5) |
| Hierarchical feasibility <1% | 4b | Medium-high | Adjust k; try 3D lifting (Phase 6) |
| ibm01 gap irreducible (structural) | 5 | Medium | Accept it; focus on close benchmarks |
| LP infeasible on large benchmarks | 3, 4 | Known | Sparse LP with margin-based pair filtering |
| Surrogate RUDY fundamentally wrong | 4, 5 | Medium | Real proxy verification as safety net |
| Time budget insufficient for large cascade | 4b | Low | Adaptive budgeting; focus on worst benchmarks |
