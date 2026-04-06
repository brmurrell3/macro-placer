# Experiment Log

Last updated: 2026-04-05

## Summary

| Hypothesis | Status | Best Avg Proxy | Variants Tried |
|------------|--------|----------------|----------------|
| Polyhedra Navigation | **graduated** | **1.4867** | 2 (phase 2) |
| SDF Density | **stalled** | 1.5002 | 10 |
| Optimal Transport | **killed** | 1.2775* | 12 |
| Diffusion | not recommended | -- | 0 |
| Variational | not recommended | -- | 0 |
| Neural ODE | not recommended | -- | 0 |
| CBO | not recommended | -- | 0 |

*Same as legalize-only baseline.

### The paradigm problem

Four of six original hypotheses share the same paradigm: continuous optimization with overlap
relaxation → legalization. Our experiments prove the optimizer does nothing and legalization
does all the work. Only polyhedra navigation escapes this by staying in feasible space.

## Baselines

| Method | Avg Proxy (--all) | Overlaps | Notes |
|--------|-------------------|----------|-------|
| RePlAce | 1.4578 | 0 | Target to beat for champion |
| SA | 2.1362 | 0 | Weak baseline |
| Will's seed (SA 3000) | 1.5338 | 0 | Current best submission |
| Greedy row | ~2.20 | 0 | Demo placer |

## SDF Density — Detailed Results

Best variant: **sdf_v5** (submissions/sdf_v5.py)
- Avg proxy cost (--all): **1.5002** (0.0002 above graduate threshold)
- Zero overlaps on all 17 benchmarks
- Runtime: ~78s (--all), ~12s (--fast)
- Beats Will's seed by 2.2% (1.5002 vs 1.5338)
- 3.3% behind RePlAce (1.5002 vs 1.4578)

### Per-benchmark breakdown (v5 vs RePlAce)
- Beats RePlAce on 3: ibm02 (+8.1%), ibm10 (+6.0%), ibm12 (+4.4%)
- Closest: ibm09 (-1.3%), ibm18 (-1.2%), ibm07 (-1.8%)
- Worst gap: ibm01 (-19.8%), ibm06 (-5.9%), ibm15 (-6.0%)

### Key findings
- **WL is excellent** (0.05-0.08 range), not the bottleneck
- **Congestion is the main weakness** (1.3-2.5 range, weight 0.5)
- Congestion proxy (v3) helped marginally but was expensive
- SA post-optimization (v7) hurt density/congestion 
- Soft macro optimization (v9) catastrophically increased density
- Force-directed init (v6) destroyed good initial structure
- Results are fully deterministic (seed-independent)
- Two-phase approach (v4) was worse than single-phase

### Variant history
| Variant | Fast Avg | All Avg | Change | Result |
|---------|----------|---------|--------|--------|
| v1 | 1.2253* | - | Initial (slow) | 78s for ibm01 alone |
| v2 | 1.2932 | 1.5062 | Vectorized WL, 32x grid | Baseline |
| v3 | 1.2775 | - | + congestion proxy | Marginal gain, 2x slower |
| v4 | 1.3797 | - | Two-phase spread+refine | Worse |
| v5 | 1.2775 | **1.5002** | Top-10% density, schedule | **Best** |
| v6 | 1.6482 | - | FD initialization | Much worse |
| v7 | 1.2921 | - | + SA post-processing | Worse |
| v8 | 1.2775 | 1.5059 | Tuned v5 | Marginal regression |
| v9 | 2.2396 | - | Soft macro opt | Catastrophic |
| v10 | 1.2775 | - | + density variance | No change |

*v1 only ran ibm01 (too slow for full fast subset)

## Status Legend

- **pending**: Not yet started
- **exploring**: Active implementation, running variants
- **active**: Passed fast_gate, running on --all
- **graduated**: avg proxy < 1.50 on --all, waiting for human review
- **killed**: Failed kill criteria, no further exploration

## Optimal Transport — Detailed Results

**Status: UNLIKELY** — 12 variants tested, no improvement over legalize-only baseline. May revisit if optimizer/legalizer improves.

### Critical finding: SDF v5 optimization loop is inert

The gradient descent optimizer in SDF v5 (and all OT variants) has **zero effect** on any benchmark.
The 1.5002 avg proxy on --all is entirely `greedy_legalize(initial_positions)`:

| Benchmark | Raw init (overlapping) | Legalized init | SDF v5 result | Matches legalized? |
|-----------|----------------------|----------------|---------------|-------------------|
| ibm01     | 1.0385 (69 overlaps) | 1.1953         | 1.1953        | YES               |
| ibm04     | 1.3133 (68 overlaps) | 1.3826         | 1.3826        | YES               |
| ibm09     | 1.1126 (101 overlaps)| 1.1337         | 1.1337        | YES               |
| ibm13     | 1.3854 (181 overlaps)| 1.3986         | 1.3986        | YES               |

### Why the optimizer doesn't work

1. Initial positions have overlaps (69-181 pairs) that trap the gradient descent
2. The overlap penalty prevents movement, WL holds positions in place
3. Best checkpoint selection uses only WL + overlap (ignoring density)
4. Net result: optimizer converges to initial positions; legalization does all work

### OT spreading results

When forced to spread (weight=200, from legalized positions), OT makes things worse:

| Variant | Approach | Fast Avg | vs Baseline | Notes |
|---------|----------|----------|-------------|-------|
| v1-v6   | Weak OT | 1.2775-1.3017 | ±0% | OT gradient negligible |
| v10     | OT from legalized | 1.3863 | -8.5% | Worse: disrupts locality |
| v11     | WL-only refinement | 1.3110 | -2.6% | WL moves hurt congestion |
| v12     | Strong OT spreading | 1.4319 | -12.1% | Much worse: density+congestion up |

### Why OT doesn't help

1. Treats macros as points (ignores rectangular sizes)
2. Uniform target destroys netlist-aware locality from initial positions
3. Congestion depends on routing demand structure, not just density uniformity
4. Initial positions already have near-uniform area coverage

### Implication

The path to improvement is **better legalization**, not better optimization. The greedy legalizer
degrades proxy cost by 10-15% from the (overlapping) initial positions. A legalizer that preserves
more structure (e.g., min-displacement, OT-based assignment) could recover significant quality.

## Polyhedra Navigation — Detailed Results

**Status: GRADUATED** — avg proxy **1.4867** on --all (2.0% behind RePlAce).

Implementation: `submissions/polyhedra/placer.py`

### Architecture (Phase 2)

Decomposes placement into discrete topology (which polyhedron) + continuous solve (LP within it):
1. **SDF v5 as initial placement** — provides good density/congestion structure
2. **Assignment extraction** — extract L/R/A/B pairwise relations from SDF positions
3. **HPWL LP solve** (HiGHS) — get optimal wirelength positions + dual variables
4. **GridSurrogate** — fast incremental proxy estimator (density + RUDY congestion + HPWL)
5. **Surrogate-guided navigation** — evaluate 1000+ candidates with surrogate, verify top-5 with real proxy
6. **Cluster moves** — net-correlated multi-pair flips + macro-centered flips
7. **Robust projection** — cascade constraint repair + direct overlap repair for zero overlaps

### Per-benchmark results (--all, Phase 3, 50s nav)

| Benchmark | Phase 3 | Phase 2 (300s) | SDF v5 | RePlAce | vs RePlAce |
|-----------|---------|----------------|--------|---------|------------|
| ibm01 | 1.1871 | **1.1715** | 1.1953 | 0.9976 | -19.0% |
| ibm02 | 1.6205 | **1.6118** | 1.6888 | 1.8370 | **+11.8%** |
| ibm03 | **1.4058** | 1.4070 | 1.4070 | 1.3222 | -6.3% |
| ibm04 | **1.3652** | 1.3496 | 1.3826 | 1.3024 | -4.8% |
| ibm06 | **1.7003** | 1.6847 | 1.7150 | 1.6187 | -5.0% |
| ibm07 | 1.4867 | **1.4818** | 1.4898 | 1.4633 | -1.6% |
| ibm08 | 1.5080 | **1.5029** | 1.5113 | 1.4285 | -5.6% |
| ibm09 | **1.1245** | 1.1195 | 1.1337 | 1.1194 | -0.5% |
| ibm10 | **1.4067** | 1.4112 | 1.4112 | 1.5009 | **+6.3%** |
| ibm11 | 1.2317 | **1.2248** | 1.2336 | 1.1774 | -4.6% |
| ibm12 | **1.6482** | 1.6478 | 1.6497 | 1.7261 | **+4.5%** |
| ibm13 | 1.3984 | **1.3947** | 1.3986 | 1.3355 | -4.7% |
| ibm14 | 1.6025 | **1.5948** | 1.6003 | 1.5436 | -3.8% |
| ibm15 | 1.6059 | **1.6045** | 1.6073 | 1.5159 | -5.9% |
| ibm16 | 1.5421 | **1.5347** | 1.5424 | 1.4780 | -4.3% |
| ibm17 | 1.7431 | **1.7422** | 1.7431 | 1.6446 | -6.0% |
| ibm18 | **1.7897** | 1.7906 | 1.7927 | 1.7722 | -1.0% |
| **AVG** | **1.4921** | **1.4867** | **1.5002** | **1.4578** | **-2.4%** |

Note: Phase 2 (300s nav) is the best quality result at ~5300s total runtime. Phase 3 (50s nav) is 5x faster at ~1030s total with 0.2% quality loss. Bold indicates per-column best.

### Phase 2 key findings

1. **GridSurrogate enables 1000+ candidate evaluations** per benchmark (vs ~20 before). Surrogate-guided filtering consistently picks candidates that improve real proxy — first verified candidate almost always accepted.

2. **Overlap repair is critical.** Cascade projection alone leaves 1-3 overlaps on many candidates. Adding direct rectangle overlap repair after cascade eliminates all overlaps while preserving quality.

3. **Cluster moves contribute.** Net-correlated multi-pair flips (2-5 pairs) found improvements on ibm09 and ibm13 that single-pair flips missed.

4. **Density is the main improvement vector.** Navigation consistently reduces density cost (0.943→0.921 on ibm01) while holding congestion and wirelength stable.

5. **Remaining gap is congestion-dominated.** RePlAce's advantage is primarily in congestion (1.3 vs 1.78 on ibm13). The surrogate's RUDY congestion approximation doesn't capture L-routing patterns accurately enough for congestion-targeted moves.

### Critical empirical findings (Phase 1, still valid)

1. **30% continuous wirelength slack** exists within SDF's polyhedron but density tradeoff makes it unexploitable directly.

2. **The LP's main value is dual variables**, not positions. Duals identify expensive pairwise constraints to flip.

3. **Constraint projection is fast but approximate.** 10 cascade rounds + direct repair usually suffices.

### Performance

- Runtime: ~74s per benchmark avg (1260s total for --all)
  - SDF init: ~3s
  - HPWL LP solve: 1.7-9s (scales with benchmark size)
  - Navigation (30s budget): 3-20 evals depending on LP solve cost
- Fast subset (4 benchmarks): ~47s avg, 186s total

## Log

- 2026-04-01: SDF hypothesis started. v1 through v10 explored.
- 2026-04-01: Best result: v5 at 1.5002 avg proxy on --all (0.0002 above graduate).
- 2026-04-01: Congestion is the remaining bottleneck. WL is already competitive.
- 2026-04-02: OT hypothesis explored. 12 variants tested.
- 2026-04-02: Critical finding: all optimization loops are inert — results = legalize(init).
- 2026-04-02: OT spreading hurts proxy cost by 8-12% when forced to activate.
- 2026-04-02: OT hypothesis marked UNLIKELY. May revisit with better legalizer/optimizer.
- 2026-04-04: Polyhedra navigation phase 1 implemented (submissions/polyhedra/placer.py).
- 2026-04-04: Avg proxy 1.4997 on --all — GRADUATED (below 1.50 threshold).
- 2026-04-04: Phase 2 implemented: GridSurrogate, cluster moves, overlap repair.
- 2026-04-04: Avg proxy **1.4910** on --all — 0.6% improvement from phase 1, 2.3% gap to RePlAce.
- 2026-04-04: Surrogate evaluates 1000+ candidates/benchmark; cluster moves contribute on ibm09/ibm13.
- 2026-04-04: Density improvements are primary vector; congestion remains main gap to RePlAce.
- 2026-04-04: Critical fix: re-extract assignment from positions after each accepted move.
- 2026-04-04: Without fix, assignment/position inconsistency caused LP infeasibility after ~5 moves.
- 2026-04-04: With fix + 300s nav: ibm01 1.1715 (31 impr), ibm09 1.1195 (matches RePlAce!).
- 2026-04-04: --fast avg 1.2588 (-1.4% from phase 2). Search NOT plateauing.
- 2026-04-04: --all avg **1.4867** (2.0% behind RePlAce, down from 2.9%). ibm09 matches RePlAce.
- 2026-04-04: Key finding: search NOT plateauing. ibm01 found 31 improvements in 300s, still improving at cutoff.
- 2026-04-05: **Phase 2 profiling completed** on all 17 benchmarks (results/profiling/).
- 2026-04-05: Component breakdown: congestion 66.5%, density 29.3%, WL 4.2% of proxy cost.
- 2026-04-05: Benchmark triage: 3 winning, 7 close (<5%), 7 hard-loss (>5%). ibm01 worst at +18.9%.
- 2026-04-05: Strongest gap predictor: num_nets (rho=0.958). More nets → harder for us.
- 2026-04-05: Surrogate global Spearman rho=0.899 (exceeds 0.7 target), BUT within-benchmark rho=0.17.
- 2026-04-05: **Critical: surrogate can rank across benchmarks but NOT within-benchmark candidates.**
- 2026-04-05: Surrogate systematic bias: underestimates proxy by 0.56 on average.
- 2026-04-05: 4 benchmarks get 0 nav improvements (ibm03, ibm06, ibm10, ibm16).
- 2026-04-05: ibm10 LP solve takes 60s, leaving 0 time for navigation.
- 2026-04-05: Phase timing: navigation 62%, LP solve 29%, SDF init 7%.
- 2026-04-05: ibm15/ibm17 slightly worsen with navigation — surrogate accepting bad moves.
- 2026-04-05: Fixed overlap check tolerance (1e-3 → 1e-6) — ibm02 had overlap without this.
- 2026-04-05: Added LP solve time cap (15s small, 60s large benchmarks) — ibm10/12/14 were starving navigation.
- 2026-04-05: ibm10 LP infeasible at 60s (308K pairs) — returns init placement (already winning vs RePlAce).
- 2026-04-05: --all avg **1.4890** (0 overlaps, 2.1% behind RePlAce). Slight regression from 1.4867 due to tighter overlap check.
- 2026-04-05: ibm02 improved 1.6118→1.6009, ibm06 improved 1.6847→1.6761. Most others slightly worse.
- 2026-04-05: Tighter overlap check trades ~0.002 avg proxy for guaranteed zero overlaps.
- 2026-04-05: **Phase 3 quick wins implemented:**
- 2026-04-05: Sparse LP: margin-based pair filtering makes ibm10 LP feasible (was infeasible). margin=1.0 for >200K pairs.
- 2026-04-05: ibm10 now gets navigation (1.4067 vs 1.4112 SDF-only), still winning vs RePlAce.
- 2026-04-05: Navigation 2x faster: top_k=50 candidates (was 150), vectorized congestion update, moved-set tracking.
- 2026-04-05: ~2.7s/iteration (was ~5-6s), 16 improvements in 45s on ibm01 (was 9).
- 2026-04-05: ibm06 improved 1.7150→1.7003 (+0.9%). ibm03 improved 1.4070→1.4058.
- 2026-04-05: Sparse LP re-solve during navigation (margin=10.0) for faster dual refresh.
- 2026-04-05: Stale detection threshold reduced to 50 iterations (was 100).
- 2026-04-05: --all avg **1.4921** at 50s nav/benchmark (0 overlaps, ~60s/benchmark total).
- 2026-04-05: Tradeoff: 5x faster runtime (1030s vs 5300s) with 0.2% quality regression on avg.
- 2026-04-05: ibm03 and ibm06 now get improvements (were stuck with 0 nav improvements).
- 2026-04-05: **Congestion diagnostic experiment** (scripts/congestion_diagnostic.py):
- 2026-04-05: Real-proxy-guided navigation (300s) on ibm01/ibm06/ibm08.
- 2026-04-05: ibm01: cong -0.04%, den -2.75%. ibm06: cong -0.67%, den -5.57%. ibm08: cong -0.36%, den -1.19%.
- 2026-04-05: **Conclusion: local pair flips cannot reach congestion.** Density is the only improvable component.
- 2026-04-05: **Swap diagnostic experiment** (scripts/swap_diagnostic.py):
- 2026-04-05: swap+LP on ibm06: congestion -44% but density +180%. LP destroys density after swap.
- 2026-04-05: swap_only (no LP): congestion <0.5%, density preserved. Single swap doesn't change routing structure.
- 2026-04-05: **Conclusion: congestion IS reachable via different topologies, but needs coordinated multi-macro changes.**
- 2026-04-05: Phases 4a (larger local moves) and 5 (congestion-aware ranking) killed by experimental evidence.
- 2026-04-05: **Group topology cascade attempted and failed:**
- 2026-04-05: Spectral clustering groups macros by connectivity (not space) — bounding boxes cover 60-100% of canvas.
- 2026-04-05: Expanding group topology to macro-level: changing 200-900 pairs creates circular constraint chains → LP infeasible.
- 2026-04-05: Even with gap filtering (only flip plausible pairs) and single group-pair flips: 0/20 LP feasible at sparse_margin=0.
- 2026-04-05: Root cause: netlist-connected macros are spatially interleaved. "Group A left of Group B" is unenforceable at macro level.
- 2026-04-05: **Sequence pair experiments:**
- 2026-04-05: Single transpositions: 3205/3250 LP feasible, 14 overlap-free, 0 improvements. Same local basin.
- 2026-04-05: Multi-step (10-500 transpositions): all feasible, all WORSE on congestion (+1-2%). Random distant topologies are worse than SDF.
- 2026-04-05: **Position blending (SDF ↔ LP):** linear interpolation monotonically worse until pure LP. No smooth path in position space.
- 2026-04-05: **Gaussian homotopy attempts:** noise+legalize fails (can't legalize 100+ overlaps). Surrogate-based ES finds lower surrogate cost but with overlaps.
- 2026-04-05: **Key conclusion:** SDF basin is genuine local minimum in ALL tested directions. Need to tunnel through barrier, not search harder.
- 2026-04-05: Tunneling theory developed: complexification, mountain pass, Potts model. See docs/tunneling-theory.md.
- 2026-04-05: Core hypothesis: flip SOFT pairs (small duals) instead of HARD pairs (large duals) to cross saddle points between basins.
