# Results

Last updated: 2026-04-23

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

**Overnight run (Apr 14-15):** 22 experiments across SP3/SP1/SP4/combine. Best: 1.4918 (McCormick area penalty). All within noise of baseline 1.4921. See "Overnight Sweep" section below.

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
7. **ClusterScreener** --- multi-tier pre-projection pruning ([evaluation.md](evaluation.md))
8. **Robust projection** --- cascade constraint repair + direct overlap repair for zero overlaps

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

## Overnight Sweep (Apr 14-15): 22 Experiments

Full log: `results/overnight_run.log`.

### Summary

22 items across 4 stages. **Global best: 1.4918** (vs baseline 1.4921). No experiment moved the needle more than noise.

| Stage | Items | Done (--all) | Killed | Skipped | Winner | Avg |
|-------|-------|-------------|--------|---------|--------|-----|
| SP3 (Surrogate) | 8 | 5 | 2 | 1 | sp3_top_k_20 | 1.4919 |
| SP1 (Topology) | 6 | 1 | 5 | 0 | sp1_congestion_aware_extraction | 1.4930 |
| SP4 (LP) | 6 | 3 | 3 | 0 | sp4_mccormick_area | 1.4918 |
| Combine | 2 | 1 | 0 | 1 | combine_stage_winners | 1.4918 |

### SP3 — Surrogate Accuracy (8 items)

| Hypothesis | Result | Avg | Finding |
|------------|--------|-----|---------|
| density_grid_fix | done | 1.4931 | Filtering to occupied cells — no effect |
| delta_ranking | done | 1.4929 | Surrogate re-inits on acceptance, so delta=absolute |
| online_calibration | killed | — | Too few accepted moves (1-6) for OLS fit in 50s |
| **top_k_20** | **done** | **1.4919** | **Best SP3. Better move selection outweighs fewer iterations** |
| rank_aggregation | done | 1.4997 | Borda ranking identical to composite (same weights) |
| adaptive_verification | skipped | — | Depends on killed online_calibration |
| pinrudy_blockage | done | 1.4935 | Pin data available; PinRUDY implemented but no improvement |
| pairwise_ranking | done | 1.4930 | Ranker never activated (needs 10 samples, got 1-7) |

**Lesson:** The surrogate is already well-calibrated for this task. Within-benchmark rho=0.17 is misleading — the real bottleneck isn't surrogate ranking, it's that **few candidates are actually better** (only 1-7 accepted per benchmark). Increasing top_k from 3→20 helps marginally by casting a wider net.

### SP1 — Initial Topology (6 items)

| Hypothesis | Result | Avg | Finding |
|------------|--------|-----|---------|
| spectral_topology | killed | 1.78 | Spectral coords ignore macro sizes → overlapping clusters → legalizer destroys connectivity |
| replace_topology | killed | — | extract_assignment() erases any congestion advantage |
| **congestion_aware_extraction** | **done** | **1.4930** | **Only SP1 finisher. Largest-gap already near-optimal** |
| hmetis_partitioning | killed | 1.91 | kahypar works, but shelf packing within partitions is terrible |
| greedy_construction | killed | 1.75 | Clustering creates dense regions — WL vs density conflict |
| boundary_attraction | killed | — | Penalty inert at safe lambda, harmful at aggressive lambda |

**Lesson:** SDF's analytical spreading produces a topology that is **extremely hard to beat** from alternative inits. Every alternative (spectral, partitioning, greedy, RePlAce extraction) produces either much worse density or identical topology after extract_assignment(). The SDF basin isn't just a local minimum — it's a **good** local minimum.

### SP4 — Congestion-Aware LP (6 items)

| Hypothesis | Result | Avg | Finding |
|------------|--------|-----|---------|
| net_weighting | done | 1.4974 | RUDY too uniform for meaningful hot cells; 3x LP overhead |
| separation_margins | done | 1.4932 | Margins added but navigation overwrites LP positions |
| **mccormick_area** | **done** | **1.4918** | **Best overall. Marginal — nearly inert at lambda=0.001** |
| dual_informed_targeting | killed | — | Uniform per-net congestion → no ranking change |
| real_proxy_feedback | killed | — | Congestion weighting inflates HPWL without reducing congestion |
| lp_navigate_reweight | done | 1.4973 | Splitting nav budget across 3 rounds cancels benefit |

**Lesson:** LP-level congestion modifications are **washed out by navigation**. The navigator re-solves the LP and re-extracts assignment each iteration, so LP starting conditions have minimal lasting effect. The few-second LP solve sets a starting point, but 50s of navigation dominates the final result.

### Combine (2 items)

Three-way combine (SP3+SP1+SP4 winners) scored 1.4918, tying sp4_mccormick_area. Changes don't compound because SP1 and SP3 winners are essentially neutral. Pairwise combine skipped due to uncommitted worktree changes.

### Strategic Implications

1. **The system is at a plateau.** 22 independent experiments spanning surrogate accuracy, initial topology, and LP formulation all produced results within ±0.5% of baseline. The polyhedra navigation architecture has been thoroughly explored at the parameter/surrogate/LP level.

2. **Navigation dominates everything upstream.** LP changes, init changes, and surrogate changes are all erased by 50s of navigation. The only path to meaningful improvement is changing **what navigation can reach** — i.e., making larger topology jumps possible.

3. **SDF init is not the bottleneck.** The literature-motivated hypothesis that "start connectivity-optimal, refine for density" would beat "start density-optimal, refine for connectivity" was decisively falsified. SDF's topology is genuinely good.

4. **Congestion is structural, not parametric.** You can't reduce congestion by tuning LP weights or surrogate parameters. It requires fundamentally different macro arrangements that local navigation (1-5 pair flips) cannot reach.

## Miftari / cheap-signal cluster screening (Apr 16) --- KILLED

Tested the `theory.md` §5a hypothesis that cheap signals from a single
solved "center" LP can rank-order nearby polyhedra for proxy quality —
the NASA-star-system analogy. Two-step verification on ibm01.

**Step 1 (experiment 1):** cheap signals vs ΔLP-HPWL across 120
cluster flips (k ∈ {1, 2, 5, 10}). Best signal `S_viol` (constraint
violation at x\*) → Spearman **ρ = 0.86**, 51 µs vs 2.2 s LP (≈ 42 700×
speedup), 91 % precision/recall at 50 % kept.
*The cheap signals are tight predictors of ΔLP-HPWL.*

**Step 2 (experiment 3):** LP-HPWL vs final refined proxy across 24
feasible topologies (Hamming 0–2 249 from base).

| correlation | ρ |
|---|---:|
| LP-HPWL → refined proxy | **−0.001** |
| LP-HPWL → refined WL | +0.852 |
| LP-HPWL → refined density | −0.536 |
| LP-HPWL → refined congestion | +0.072 |
| refined WL → refined density | −0.416 |
| refined congestion → refined proxy | +0.825 |

*LP-HPWL carries no usable information about refined proxy.* HPWL and
density anti-correlate physically (tighter WL ⇒ denser ⇒ worse density
cost); the two effects nearly cancel. Proxy on ibm01 is congestion-
dominated (ρ = 0.83) and LP-HPWL is uncorrelated with congestion.

**Chain:** `cheap signal → LP-HPWL → refined proxy`; first link ✓ ρ=0.86,
second link ✗ ρ≈0.

**Implications:**
- S_viol filtering is not a valid polyhedron-quality screen.
- Re-confirms overnight SP1 finding at the signal level:
  HPWL-based topology ranking is blind to the objective.
- A proxy-predicting cheap signal would need density + congestion
  components; `GridSurrogate` already integrates those with Spearman
  ≈ 0.17, so the naive combination has been tried.

Cost: ~2 h. Saved building a multi-level cascade on a signal that
doesn't track the objective.

## See Also

- [approach.md](approach.md) --- methodology and architecture
- [theory.md](theory.md) --- theoretical foundations and frameworks
- [roadmap.md](roadmap.md) --- next steps
- [evaluation.md](evaluation.md) --- navigator eval pipeline (screener + proposed incremental evaluator)
