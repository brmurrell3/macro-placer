# Roadmap

Last updated: 2026-04-14
Competition deadline: May 21, 2026 (~5.3 weeks)

---

## Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | --- |
| **Polyhedra (50s nav)** | **1.4921** | **-2.4%** |
| SDF v5 (init) | 1.5002 | -2.9% |

- Beat RePlAce on 3/17 benchmarks (ibm02, ibm10, ibm12)
- Congestion is 66.5% of proxy, density 29.3%, WL 4.2%
- Local navigation at ceiling --- can improve density but not congestion
- Surrogate within-benchmark rho = 0.17 (nearly blind)

---

Phases 1-3 complete. Phase 4 (tunneling) partially explored, no improvement.
See results.md for full history.

---

## Attack Strategy

Three independent subproblems with room to improve:

| SP | Problem | Share of cost | Current status | Opportunity |
|----|---------|---------------|----------------|-------------|
| SP1 | Initial topology | Controls ~95% of congestion | SDF basin = local minimum | Find a better global basin |
| SP4 | Congestion in LP | 66.5% | LP ignores congestion entirely | Congestion-aware LP objective |
| SP3 | Surrogate accuracy | Enables navigation | rho=0.17 within-benchmark | Fix ranking → better moves |

Key insight from literature: successful placers start **connectivity-optimal** and refine for density. We do the opposite (density-optimal SDF → navigate). Fixing this inversion is the highest-potential change.

SP3 fixes are independent and improve all subsequent work. Do them first.

---

## Phase 5: Quick Validation (Apr 14-17, 3 days)

Goal: Run cheap experiments across all three SPs. Determine which levers actually move the needle before investing deeply.

### 5a. SP3 — Surrogate fixes (Day 1)

Fix known issues and add calibration. These improve navigation quality for ALL subsequent experiments.

1. **Fix density grid bug** — top-10% of occupied cells, not all cells (`surrogate.py:239-244`)
2. **Delta-based ranking** — rank by surrogate cost change, not absolute cost
3. **Per-component online calibration** — fit affine corrections per component using verified evaluations
4. **Increase top_k_verify** to 10 (from 5)

**Test:** Run `--fast` before and after. Measure within-benchmark Spearman rho. If rho > 0.35, surrogate is meaningfully better.

**Kill gate:** If rho doesn't improve, surrogate error is structural (not calibratable). Move on to PinRUDY + macro blockage.

### 5b. SP1 — Two topology experiments (Day 2)

Test two fundamentally different initial topologies. Each takes ~1-2 hours to implement.

1. **Spectral topology** — Fiedler vector of netlist Laplacian → extract assignment → LP → navigate
2. **Congestion-aware assignment extraction** — modify `extract_assignment()` to prefer directions minimizing net bounding box area for congested nets

**Test:** Run `--fast` for each. Compare congestion component to SDF baseline. If EITHER shows >5% lower congestion (even with worse density), the basin hypothesis is confirmed.

**Kill gate:** If neither moves congestion, the topology selection problem requires heavier machinery (hMETIS, multilevel). Move to Phase 6 approaches.

### 5c. SP4 — Net weighting prototype (Day 3)

Test congestion-aware LP within the current SDF topology.

1. **Iterative net weighting** — after LP solve, compute RUDY congestion, increase weights on nets in congested cells, re-solve 2-3 times
2. **Measure:** congestion component before and after reweighting (at LP positions, before navigation)

**Test:** Run `--fast`. Compare congestion at LP positions (before navigation). If congestion drops >2% at LP stage, the LP lever works.

**Kill gate:** If reweighting doesn't move congestion within the topology, the problem is the topology, not the LP objective. Confirms SP1 is the bottleneck.

### Phase 5 Decision Gate (end of Day 3)

| SP3 result | SP1 result | SP4 result | Next action |
|------------|------------|------------|-------------|
| rho improved | Congestion moved | Congestion moved | All levers work — combine in Phase 6 |
| rho improved | Congestion moved | No movement | Topology is bottleneck — invest in SP1 (Phase 6a) |
| rho improved | No movement | Congestion moved | LP lever works — invest in SP4 (Phase 6b) |
| rho improved | No movement | No movement | Surrogate was the bottleneck — test with longer navigation |
| No improvement | Congestion moved | * | SP1 is the key — invest in SP1 |
| No improvement | No movement | No movement | Need heavy machinery — go to Phase 6c |

---

## Phase 6: Deep Investment (Apr 18 - May 4, ~2.5 weeks)

### Phase 6a: Topology (if SP1 validated)

**Week 1:**
1. **hMETIS/KaHyPar partitioning topology** — recursive bisection of netlist, convert partition tree to pairwise assignment (1-2 days)
2. **Combine best init with congestion-aware LP** — test spectral/partition topology with net weighting (1 day)
3. **Run --all** on best combination

**Week 2:**
4. **Multilevel topology navigation** — cluster macros, navigate at coarse level (K=20 super-macros), refine at fine level (5-7 days)
5. **Full --all** validation

### Phase 6b: Congestion LP (if SP4 validated)

**Week 1:**
1. **Congestion-proportional separation margins** — increase separations in congested regions (1 day)
2. **Combine net weighting + margins** — test on --all (1 day)
3. **LP-Navigate-Reweight outer loop** — 3 rounds of (LP → navigate 15s → reweight) (2 days)

**Week 2:**
4. **Dual-informed congestion targeting** — navigator uses congestion duals, not HPWL duals (2 days)
5. **Real-proxy congestion feedback** — use PlacementCost congestion map for LP weights at init (1 day)
6. **Full --all** validation

### Phase 6c: Heavy machinery (if quick tests failed)

If neither topology change nor LP reweighting moved congestion in Phase 5:

1. **TCG with transitive closure** — guaranteed-consistent multi-pair topology changes (3-5 days)
2. **McCormick bounding box area penalty in LP** — penalize net bbox area directly (2 days)
3. **PinRUDY + macro blockage** — improve surrogate congestion model (2 days)

---

## Phase 7: Integration & Polish (May 5-14)

1. Combine best findings from Phase 6 into a single placer
2. Run --all with 3 different seeds for robustness
3. Optimize time budget allocation (LP iters vs navigation time)
4. Profile and tune parameters (alpha, n_rounds, margins)
5. Full --all + --ng45 validation

**Target:** avg proxy < 1.46 (beat RePlAce)

---

## Phase 8: Innovation Prize & Submit (May 14-21)

1. Write competition report (PDF):
   - The decomposition theorem (union of convex polyhedra)
   - Empirical findings (congestion barrier, topology dependence)
   - Congestion-aware LP within polyhedra framework (novel contribution)
   - Connections to spectral theory, mountain pass, etc.
2. Visualizations of polyhedra structure + congestion maps
3. Final validation: all 17 benchmarks + NG45, zero overlaps, <60s/benchmark
4. Submit placement + innovation prize report

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| No quick test moves congestion | Medium | Phase 6c heavy machinery; accept ~1.49 |
| Spectral topology has terrible density | Medium | Navigator is good at density recovery |
| Net weighting destabilizes LP | Low | Cap weights at 5x, normalize mean to 1.0 |
| Multilevel clustering too expensive | Low | Use coarse K=10-20; time budget is 60s |
| Not enough time for both placement and report | Low | Report is parallelizable with experiments |
| Phase 5 takes longer than 3 days | Medium | Each experiment is independent; can parallelize |

---

## See Also

- [sp1_topology_selection.md](sp1_topology_selection.md) --- SP1 detailed approaches
- [sp4_congestion_lp.md](sp4_congestion_lp.md) --- SP4 detailed approaches
- [sp3_surrogate_accuracy.md](sp3_surrogate_accuracy.md) --- SP3 detailed approaches
- [approach.md](approach.md) --- methodology and architecture
- [results.md](results.md) --- experiment history
- [theory.md](theory.md) --- theoretical foundations
