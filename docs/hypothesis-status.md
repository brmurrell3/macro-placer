# Hypothesis Status Tracker

Last updated: 2026-04-02

## Summary

| Hypothesis | Status | Best Avg Proxy | Variants Tried | Next Step |
|------------|--------|----------------|----------------|-----------|
| SDF Density | **exploring** | **1.5002** | 10 | Congestion reduction needed to pass 1.50 |
| Optimal Transport | **unlikely** | 1.2775* | 12 | *Same as legalize-only baseline; may revisit with better legalizer |
| Diffusion | pending | -- | 0 | Energy-guided Langevin baseline |
| Variational | pending | -- | 0 | JKO scheme on ibm01 |
| Neural ODE | pending | -- | 0 | Verify adjoint gradients |
| CBO | pending | -- | 0 | 100-particle swarm on ibm01 |

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

## Log

- 2026-04-01: SDF hypothesis started. v1 through v10 explored.
- 2026-04-01: Best result: v5 at 1.5002 avg proxy on --all (0.0002 above graduate).
- 2026-04-01: Congestion is the remaining bottleneck. WL is already competitive.
- 2026-04-02: OT hypothesis explored. 12 variants tested.
- 2026-04-02: Critical finding: all optimization loops are inert — results = legalize(init).
- 2026-04-02: OT spreading hurts proxy cost by 8-12% when forced to activate.
- 2026-04-02: OT hypothesis marked UNLIKELY. May revisit with better legalizer/optimizer.
