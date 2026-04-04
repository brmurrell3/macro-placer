# Hypothesis Status Tracker

Last updated: 2026-04-04

## Summary

| Hypothesis | Status | Best Avg Proxy | Variants Tried | Next Step |
|------------|--------|----------------|----------------|-----------|
| Polyhedra Navigation | **graduated** | **1.4997** | 1 | Speed optimization, multi-pair moves |
| SDF Density | **stalled** | 1.5002 | 10 | Superseded by polyhedra; used as init topology |
| Optimal Transport | **killed** | 1.2775* | 12 | *Same as legalize-only baseline |
| Diffusion | not recommended | -- | 0 | Same legalization wall as SDF/OT |
| Variational | not recommended | -- | 0 | Same legalization wall as SDF/OT |
| Neural ODE | not recommended | -- | 0 | Same legalization wall as SDF/OT |
| CBO | not recommended | -- | 0 | Can't encode non-overlap constraint |

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

**Status: GRADUATED** — avg proxy 1.4997 on --all (below 1.50 threshold).

Implementation: `submissions/polyhedra/placer.py`

### Architecture

Decomposes placement into discrete topology (which polyhedron) + continuous solve (LP within it):
1. **SDF v5 as initial placement** — provides good density/congestion structure
2. **Assignment extraction** — extract L/R/A/B pairwise relations from SDF positions
3. **HPWL LP solve** (HiGHS) — get optimal wirelength positions + dual variables
4. **Dual-guided navigation** — flip pairs ranked by |dual|, project positions, eval proxy cost
5. Accept flips that improve full proxy cost (wirelength + 0.5*density + 0.5*congestion)

### Per-benchmark results (--all)

| Benchmark | Polyhedra | SDF v5 | RePlAce | vs RePlAce |
|-----------|-----------|--------|---------|------------|
| ibm01 | 1.1951 | 1.1953 | 0.9976 | -19.8% |
| ibm02 | 1.6888 | 1.6888 | 1.8370 | **+8.1%** |
| ibm03 | 1.4070 | 1.4070 | 1.3222 | -6.4% |
| ibm04 | 1.3826 | 1.3826 | 1.3024 | -6.2% |
| ibm06 | 1.7093 | 1.7150 | 1.6187 | -5.6% |
| ibm07 | 1.4897 | 1.4898 | 1.4633 | -1.8% |
| ibm08 | 1.5113 | 1.5113 | 1.4285 | -5.8% |
| ibm09 | 1.1326 | 1.1337 | 1.1194 | -1.2% |
| ibm10 | 1.4112 | 1.4112 | 1.5009 | **+6.0%** |
| ibm11 | 1.2336 | 1.2336 | 1.1774 | -4.8% |
| ibm12 | 1.6497 | 1.6497 | 1.7261 | **+4.4%** |
| ibm13 | 1.3986 | 1.3986 | 1.3355 | -4.7% |
| ibm14 | 1.6003 | 1.6003 | 1.5436 | -3.7% |
| ibm15 | 1.6073 | 1.6073 | 1.5159 | -6.0% |
| ibm16 | 1.5424 | 1.5424 | 1.4780 | -4.4% |
| ibm17 | 1.7431 | 1.7431 | 1.6446 | -6.0% |
| ibm18 | 1.7927 | 1.7927 | 1.7722 | -1.2% |
| **AVG** | **1.4997** | **1.5002** | **1.4578** | **-2.9%** |

### Critical empirical findings

1. **30% continuous wirelength slack** exists within SDF's polyhedron. LP-optimal HPWL within the same topology is 30% better than SDF's HPWL (0.0512 vs 0.0731 on ibm01). But exploiting it is impossible — density degrades catastrophically. Even 5% interpolation toward LP positions makes proxy cost worse.

2. **Navigation gives ~0.1% proxy improvement** via single-pair flips. The SDF topology is already near a local optimum for proxy cost. Larger moves (multi-pair, cluster) are needed to escape.

3. **The LP's main value is dual variables**, not positions. Duals identify which pairwise constraints are expensive (435 of 30,135 pairs have nonzero duals on ibm01). These guide which pairs to flip.

4. **Proxy cost evaluation is the bottleneck** (~1s per call via PlacementCost). With 30s navigation budget, we evaluate ~10-20 candidate flips per benchmark. Most candidates don't improve proxy cost.

5. **Constraint projection is fast but approximate.** Moving two macros to satisfy a flipped constraint can cascade into violations with neighbors. 3 rounds of cascade projection usually suffices.

### Performance

- Runtime: ~74s per benchmark avg (1260s total for --all)
  - SDF init: ~3s
  - HPWL LP solve: 1.7-9s (scales with benchmark size)
  - Navigation (30s budget): 3-20 evals depending on LP solve cost
- Fast subset (4 benchmarks): ~47s avg, 186s total

### Bottlenecks and next steps

1. **Speed**: HPWL LP solve for ibm13+ (424+ macros) takes 9s. Could skip LP and use heuristic duals, or sparsify the LP by only including nearby-pair constraints.

2. **Stronger moves**: Single-pair flips can't escape local optima. Need:
   - Multi-pair simultaneous flips (e.g., flip all pairs involving macro i)
   - Swendsen-Wang cluster moves (flip connected components)
   - Hierarchical grouping (Level 1 coarse search from the paper)

3. **Faster proxy evaluation**: The PlacementCost evaluator is the true bottleneck. Could use a lightweight surrogate (e.g., grid-based density + HPWL estimate) for filtering, then full eval only for promising candidates.

4. **Better initial topology**: SDF's topology is near-optimal locally. Starting from a different topology (e.g., RePlAce's, or a random perturbation) might find better basins.

## Log

- 2026-04-01: SDF hypothesis started. v1 through v10 explored.
- 2026-04-01: Best result: v5 at 1.5002 avg proxy on --all (0.0002 above graduate).
- 2026-04-01: Congestion is the remaining bottleneck. WL is already competitive.
- 2026-04-02: OT hypothesis explored. 12 variants tested.
- 2026-04-02: Critical finding: all optimization loops are inert — results = legalize(init).
- 2026-04-02: OT spreading hurts proxy cost by 8-12% when forced to activate.
- 2026-04-02: OT hypothesis marked UNLIKELY. May revisit with better legalizer/optimizer.
- 2026-04-04: Polyhedra navigation implemented (submissions/polyhedra/placer.py).
- 2026-04-04: Avg proxy 1.4997 on --all — GRADUATED (below 1.50 threshold).
- 2026-04-04: 30% wirelength slack confirmed but density tradeoff makes it unexploitable directly.
- 2026-04-04: Single-pair navigation adds ~0.1% improvement; need multi-pair moves for more.
