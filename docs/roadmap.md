# Roadmap

Last updated: 2026-04-23
Competition deadline: May 21, 2026 (~4 weeks)

---

## Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | --- |
| **Polyhedra (50s nav)** | **1.4918** | **-2.3%** |
| SDF v5 (init) | 1.5002 | -2.9% |

- Beat RePlAce on 3/17 benchmarks (ibm02, ibm10, ibm12)
- Congestion is 66.5% of proxy cost, density 29.3%, WL 4.2%
- **Phase 5 complete (overnight sweep): 22 experiments, all within noise of baseline**
- Local navigation at ceiling — neither surrogate fixes, alternative inits, nor LP modifications moved the needle

---

## Completed Phases

**Phases 1-3:** Architecture built. SDF init → LP → surrogate-guided navigation → robust projection. Graduated at 1.4867 (300s nav), 1.4921 (50s nav).

**Phase 4:** Tunneling theory. Soft-pair flips, group cascade, sequence pairs — all failed to cross the congestion barrier.

**Phase 5 (overnight Apr 14-15):** 22 automated experiments across 3 subproblems + combine. Decision gate result: **none of the three levers (SP3/SP1/SP4) moved congestion**. See results.md for full breakdown.

Key Phase 5 findings:
- SP3: Surrogate is already well-calibrated. Only top_k_verify=20 helped marginally (1.4919).
- SP1: SDF init is genuinely hard to beat. Spectral (1.78), HMetis (1.91), greedy (1.75) all far worse.
- SP4: LP modifications are washed out by 50s of navigation. McCormick area penalty best at 1.4918.
- Navigation dominates everything upstream. LP starting conditions don't persist.

---

## What's Left

The overnight sweep eliminated incremental approaches. Remaining options fall into two categories:

### A. Structural Changes (high risk, high reward)

These change the architecture, not parameters.

| Approach | Effort | Potential | Risk |
|----------|--------|-----------|------|
| **Multilevel navigation** | 1-2 weeks | Break congestion barrier via coarse-level topology | Coarse decisions may not survive refinement |
| **TCG closure propagation** | 3-5 days | Large consistent topology moves (50-200 pairs) | O(N^3) may be too slow for N=500 |
| **Hybrid: SA at coarse + LP at fine** | 1 week | SA explores topologies, LP optimizes within each | Integration complexity |
| **Incremental real-proxy evaluator** | 3-5 days | Exact signal replaces ρ=0.17 surrogate; removes confound | SP3 results suggest accuracy isn't binding constraint |

### B. Accept ~1.49 and Focus on Innovation Prize

The polyhedra decomposition framework IS novel and publishable regardless of whether we beat RePlAce. The innovation prize values methodology over raw score.

---

### C. Infrastructure (implemented)

- **ClusterScreener** (implemented): multi-tier pre-projection pruning (Zobrist dedup, displacement floor, HPWL bound, axis crowding). Rejects ~20-40% of obviously bad moves at ~1-10us each, saving projection+surrogate cost. See [evaluation.md](evaluation.md).

---

## Phase 6: Structural Attack (Apr 15 - May 4, ~2.7 weeks)

### Option 6a: Multilevel Navigation (recommended)

The only approach from literature that could cross the congestion barrier. All others failed in Phase 5.

**Week 1 (Apr 15-21):**
1. **Macro clustering** — hMETIS/modularity clustering into K=15-25 super-macros (1-2 days)
2. **Coarse-level polyhedra** — SDF + LP + navigation on super-macro problem (2-3 days)
3. **Uncoarsening** — expand super-macro placements to individual macros (1 day)
4. **Fast validation** — `--fast` on multilevel pipeline

**Week 2 (Apr 21-28):**
5. **Fine-level refinement** — run current navigator within each cluster (2 days)
6. **Iterate** — alternate coarse/fine optimization, 2-3 levels (3 days)
7. **Full --all validation**

**Week 3 (Apr 28 - May 4):**
8. **Parameter tuning** — cluster sizes, time budgets per level, iteration count
9. **Robustness** — test with different random seeds
10. **Final --all + --ng45 validation**

**Kill gate:** If coarse-level navigation on super-macros doesn't produce >5% different congestion patterns than SDF init by end of Week 1, abandon multilevel and move to Phase 7.

### Option 6b: TCG (backup)

Only if multilevel fails. TCG ensures consistency for large topology moves but is O(N^3) — may not scale to ibm18 (537 macros).

### Option 6c: Incremental Real-Proxy Evaluator (infrastructure)

Replace GridSurrogate with an exact incremental evaluator matching `compute_proxy_cost` bit-for-bit. Removes the surrogate as a confound and enables more candidates per iteration at exact signal. 3-5 days. Orthogonal to multilevel — could be combined with 6a. See [evaluation.md](evaluation.md) for full design. Risk: SP3 evidence suggests surrogate accuracy isn't the binding constraint, so payoff may be diagnostic rather than score-improving.

---

## Phase 7: Polish & Report (May 5-14)

1. Merge best findings (McCormick + top_k_20 at minimum)
2. Time budget optimization per benchmark
3. Run --all with 3 seeds for robustness
4. Full --all + --ng45 validation
5. Begin innovation prize report

---

## Phase 8: Innovation Prize & Submit (May 14-21)

1. Write competition report (PDF):
   - The decomposition theorem (union of convex polyhedra)
   - Empirical findings: 22-experiment systematic sweep proving congestion is structural
   - Congestion barrier analysis (the most interesting technical finding)
   - Connections to spectral theory, mountain pass, combinatorial optimization
2. Visualizations of polyhedra structure + congestion maps
3. Final validation: all 17 benchmarks + NG45, zero overlaps, <60s/benchmark
4. Submit placement + innovation prize report

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Multilevel doesn't scale to 500+ macros in 60s | Medium | Limit to 2 levels; coarse K=15-20 |
| Multilevel doesn't break congestion barrier | Medium | Accept ~1.49; focus on innovation prize |
| Not enough time for both multilevel and report | Medium | Start report outline now (Phase 7); parallelize |
| Coarse decisions don't survive refinement | Medium | Iterate coarse/fine; test on small benchmarks first |

---

## See Also

- [approach.md](approach.md) --- methodology and architecture
- [results.md](results.md) --- experiment history (includes overnight sweep)
- [theory.md](theory.md) --- theoretical foundations
- [evaluation.md](evaluation.md) --- navigator eval pipeline (screener + proposed incremental evaluator)
