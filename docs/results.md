# Results

Last updated: 2026-04-13

Trimmed results focusing on the active polyhedra navigation hypothesis.
SDF density and optimal transport hypotheses were explored and superseded/killed;
see git history for full variant logs.

## Baselines

| Method | Avg Proxy (--all) | Overlaps | Notes |
|--------|-------------------|----------|-------|
| RePlAce | 1.4578 | 0 | Target to beat for champion |
| SA | 2.1362 | 0 | Weak baseline |
| Will's seed (SA 3000) | 1.5338 | 0 | Current best submission |
| Greedy row | ~2.20 | 0 | Demo placer |

## Summary

| Hypothesis | Status | Best Avg Proxy | Notes |
|------------|--------|----------------|-------|
| Polyhedra Navigation | **graduated** | **1.4867** | Active approach, 2.0% behind RePlAce |
| SDF Density | superseded | 1.5002 | Now used as init for polyhedra navigation |

## Polyhedra Navigation --- Results

**Status: GRADUATED** --- avg proxy **1.4867** on --all (2.0% behind RePlAce).

Implementation: `submissions/polyhedra/placer.py`

### Architecture (Phase 2)

Decomposes placement into discrete topology (which polyhedron) + continuous solve (LP within it):
1. **SDF v5 as initial placement** --- provides good density/congestion structure
2. **Assignment extraction** --- extract L/R/A/B pairwise relations from SDF positions
3. **HPWL LP solve** (HiGHS) --- get optimal wirelength positions + dual variables
4. **GridSurrogate** --- fast incremental proxy estimator (density + RUDY congestion + HPWL)
5. **Surrogate-guided navigation** --- evaluate 1000+ candidates with surrogate, verify top-5 with real proxy
6. **Cluster moves** --- net-correlated multi-pair flips + macro-centered flips
7. **Robust projection** --- cascade constraint repair + direct overlap repair for zero overlaps

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

1. **GridSurrogate enables 1000+ candidate evaluations** per benchmark (vs ~20 before). Surrogate-guided filtering consistently picks candidates that improve real proxy --- first verified candidate almost always accepted.

2. **Overlap repair is critical.** Cascade projection alone leaves 1-3 overlaps on many candidates. Adding direct rectangle overlap repair after cascade eliminates all overlaps while preserving quality.

3. **Cluster moves contribute.** Net-correlated multi-pair flips (2-5 pairs) found improvements on ibm09 and ibm13 that single-pair flips missed.

4. **Density is the main improvement vector.** Navigation consistently reduces density cost (0.943->0.921 on ibm01) while holding congestion and wirelength stable.

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

### Phase 3 quick wins

- Sparse LP: margin-based pair filtering makes ibm10 LP feasible (was infeasible). margin=1.0 for >200K pairs.
- Navigation 2x faster: top_k=50 candidates (was 150), vectorized congestion update, moved-set tracking.
- ~2.7s/iteration (was ~5-6s), 16 improvements in 45s on ibm01 (was 9).
- ibm06 improved 1.7150->1.7003, ibm03 improved 1.4070->1.4058.
- Sparse LP re-solve during navigation (margin=10.0) for faster dual refresh.
- --all avg **1.4921** at 50s nav/benchmark (0 overlaps, ~60s/benchmark total).
- Tradeoff: 5x faster runtime (1030s vs 5300s) with 0.2% quality regression on avg.

### Profiling results (Phase 2)

- Component breakdown: congestion 66.5%, density 29.3%, WL 4.2% of proxy cost.
- Benchmark triage: 3 winning, 7 close (<5%), 7 hard-loss (>5%). ibm01 worst at +18.9%.
- Strongest gap predictor: num_nets (rho=0.958). More nets -> harder for us.
- Surrogate global Spearman rho=0.899 (exceeds 0.7 target), BUT within-benchmark rho=0.17.
- **Critical: surrogate can rank across benchmarks but NOT within-benchmark candidates.**
- 4 benchmarks get 0 nav improvements (ibm03, ibm06, ibm10, ibm16).

### Congestion and topology experiments

- **Local pair flips cannot reach congestion.** Real-proxy-guided navigation (300s) on ibm01/ibm06/ibm08: congestion moves <1%, density moves 2-6%.
- **Congestion IS reachable via different topologies** (swap+LP: congestion -44%) but LP destroys density after swap (+180%).
- **Group topology cascade fails:** spectral clustering groups cover 60-100% of canvas; expanding to macro-level creates circular constraint chains -> LP infeasible.
- **Sequence pair experiments:** single transpositions: 3205/3250 LP feasible, 14 overlap-free, 0 improvements. Multi-step (10-500): all worse on congestion.
- **SDF basin is a genuine local minimum** in all tested directions. Need to tunnel through barrier, not search harder.
- Core hypothesis for next phase: flip SOFT pairs (small duals) instead of HARD pairs (large duals) to cross saddle points between basins.

## Key Findings

- **The paradigm problem:** continuous optimization with overlap relaxation -> legalization is broken. Optimizer converges to initial positions; legalization does all the work. Only polyhedra navigation escapes this by staying in feasible space.
- **Congestion dominates the gap to RePlAce** (66.5% of proxy cost). Density is improvable via local moves; congestion requires topology changes that current methods cannot reach without infeasibilities.
- **Surrogate ranking is the bottleneck:** global correlation is strong (rho=0.899) but within-benchmark candidate ranking is weak (rho=0.17). Improving surrogate fidelity would directly improve navigation quality.
- **Search is NOT plateauing:** ibm01 found 31 improvements in 300s, still improving at cutoff. More time budget -> more improvements.
- **3 benchmarks beat RePlAce:** ibm02 (+11.8%), ibm10 (+6.3%), ibm12 (+4.5%). These have favorable macro/net ratios.

## See Also

- [approach.md](approach.md) --- methodology and architecture
- [theory.md](theory.md) --- theoretical foundations and frameworks
- [roadmap.md](roadmap.md) --- next steps
