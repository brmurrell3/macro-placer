# Historical Results

Per-benchmark tables and experiment data from superseded approaches.
Preserved for the writeup so we don't lose the evidence that motivated
each pivot. Operational `docs/results.md` only carries the current
champion; everything pre-CD lives here.

---

## DPO best-of-v2 — Prior Champion (1.3834 avg, +5.1% vs RePlAce)

Configuration: `submissions/dpo/best_of_v2_placer.py` (best-of SDF +
v2-steps DPO). Replaced 2026-04-27 by CDOnly.

### Per-benchmark (--all)

| Benchmark | Best-of-v2 | DPO v3 | Poly (50s) | SDF v5 | RePlAce | vs RePlAce | Winner |
|-----------|------------|--------|------------|--------|---------|------------|--------|
| ibm01 | 1.1285 | 1.2105 | 1.1871 | 1.1953 | 0.9976 | -13.1% | DPO-v2 |
| ibm02 | 1.6888 | 1.7560 | 1.6205 | 1.6888 | 1.8370 | **+8.1%** | SDF |
| ibm03 | 1.2508 | 1.2869 | 1.4058 | 1.4070 | 1.3222 | **+5.4%** | DPO-v2 |
| ibm04 | 1.3239 | 1.3570 | 1.3652 | 1.3826 | 1.3024 | -1.7% | DPO-v2 |
| ibm06 | 1.6434 | 1.7553 | 1.7003 | 1.7150 | 1.6187 | -1.5% | DPO-v2 |
| ibm07 | 1.3846 | 1.4352 | 1.4867 | 1.4898 | 1.4633 | **+5.4%** | DPO-v2 |
| ibm08 | 1.3816 | 1.4383 | 1.5080 | 1.5113 | 1.4285 | **+3.3%** | DPO-v2 |
| ibm09 | 1.0130 | 1.0529 | 1.1245 | 1.1337 | 1.1194 | **+9.5%** | DPO-v2 |
| ibm10 | 1.2540 | 1.2793 | 1.4067 | 1.4112 | 1.5009 | **+16.5%** | DPO-v2 |
| ibm11 | 1.0657 | 1.0999 | 1.2317 | 1.2336 | 1.1774 | **+9.5%** | DPO-v2 |
| ibm12 | 1.6497 | 1.7166 | 1.6482 | 1.6497 | 1.7261 | **+4.4%** | SDF |
| ibm13 | 1.2327 | 1.2384 | 1.3984 | 1.3986 | 1.3355 | **+7.7%** | DPO-v2 |
| ibm14 | 1.4719 | 1.5056 | 1.6025 | 1.6003 | 1.5436 | **+4.6%** | DPO-v2 |
| ibm15 | 1.3742 | 1.4047 | 1.6059 | 1.6073 | 1.5159 | **+9.3%** | DPO-v2 |
| ibm16 | 1.3819 | 1.4217 | 1.5421 | 1.5424 | 1.4780 | **+6.5%** | DPO-v2 |
| ibm17 | 1.6121 | 1.6038 | 1.7431 | 1.7431 | 1.6446 | **+2.0%** | DPO-v2 |
| ibm18 | 1.6614 | 1.6569 | 1.7897 | 1.7927 | 1.7722 | **+6.3%** | DPO-v2 |
| **AVG** | **1.3834** | **1.4246** | **1.4921** | **1.5002** | **1.4578** | **+5.1%** | 15 DPO / 2 SDF |

### Key findings

1. Best-of-v2 beats RePlAce on 15/17 benchmarks. Largest wins on ibm10 (+16.5%), ibm09/11 (+9.5%), ibm15 (+9.3%).
2. v2 step counts matter. Relaxing step_scale from min 0.25 to min 0.6 improved avg from 1.4246 to 1.3888 (+2.5%).
3. Best-of selection catches regressions. SDF wins ibm02 and ibm12 — high-density basin lock.
4. Density is the main improvement vector. DPO consistently achieves density ~0.51-0.57 vs SDF's ~0.9+.
5. RUDY congestion underestimates real congestion by ~3.1x. Top-5% hotspot overlap RUDY-vs-real: only 10.9%.
6. Congestion weight scaling doesn't help (sweep 0.5-1.5 monotonically worse). Direction wrong, not magnitude.
7. Overlap penalty continuation works (Phase 1 lambda=1, Phase 2 lambda=50, Phase 3 lambda=500).

---

## Polyhedra Navigation — Pre-DPO Champion (1.4867 avg, -2.0% vs RePlAce)

Implementation: `submissions/polyhedra/placer.py` (removed in
post-CD cleanup; init/sdf.py kept). Was best 2026-04-15.

### Per-benchmark results (--all)

| Benchmark | Phase 3 (50s nav) | Phase 2 (300s nav) | SDF v5 | RePlAce | vs RePlAce |
|-----------|-------------------|---------------------|--------|---------|------------|
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

Phase 2 (300s) is the best polyhedra result at ~5300s total runtime. Phase 3 (50s) is 5x faster (~1030s total) with 0.2% quality loss.

### Profiling results (Phase 2)

- Component breakdown: congestion 66.5%, density 29.3%, WL 4.2%
- Benchmark triage: 3 winning, 7 close (<5%), 7 hard-loss (>5%); ibm01 worst at +18.9%
- Strongest gap predictor: num_nets (rho=0.958)
- Surrogate global Spearman rho=0.899; within-benchmark rho=0.17
- 4 benchmarks get 0 nav improvements (ibm03, ibm06, ibm10, ibm16)

### Congestion / topology experiments (the dead ends that motivated DPO)

- **Local pair flips cannot reach congestion.** Real-proxy-guided navigation (300s) on ibm01/06/08: congestion moves <1%, density moves 2-6%.
- **Congestion IS reachable via different topologies** (swap+LP: congestion -44%) but LP destroys density after swap (+180%).
- **Group topology cascade fails:** spectral clustering groups cover 60-100% of canvas; expanding to macro-level creates circular constraint chains -> LP infeasible.
- **Sequence pair experiments:** single transpositions: 3205/3250 LP feasible, 14 overlap-free, 0 improvements. Multi-step (10-500): all worse on congestion.
- **SDF basin is a genuine local minimum** in all tested directions.

---

## Overnight Sweep (Apr 14-15) — 22 Experiments

Full log: `writeup/archive/overnight_run.log`.

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

**Lesson:** Surrogate already well-calibrated. Real bottleneck: only 1-7 candidates per benchmark are actually better.

### SP1 — Initial Topology (6 items)

| Hypothesis | Result | Avg | Finding |
|------------|--------|-----|---------|
| spectral_topology | killed | 1.78 | Spectral coords ignore macro sizes → overlapping clusters |
| replace_topology | killed | — | extract_assignment() erases congestion advantage |
| **congestion_aware_extraction** | **done** | **1.4930** | **Only SP1 finisher. Largest-gap already near-optimal** |
| hmetis_partitioning | killed | 1.91 | kahypar works, but shelf packing within partitions terrible |
| greedy_construction | killed | 1.75 | Clustering creates dense regions — WL vs density conflict |
| boundary_attraction | killed | — | Penalty inert at safe lambda, harmful at aggressive lambda |

**Lesson:** SDF's analytical spreading produces a topology that is extremely hard to beat. Every alternative produces either much worse density or identical topology after extract_assignment().

### SP4 — Congestion-Aware LP (6 items)

| Hypothesis | Result | Avg | Finding |
|------------|--------|-----|---------|
| net_weighting | done | 1.4974 | RUDY too uniform for meaningful hot cells; 3x LP overhead |
| separation_margins | done | 1.4932 | Margins added but navigation overwrites LP positions |
| **mccormick_area** | **done** | **1.4918** | **Best overall. Marginal — nearly inert at lambda=0.001** |
| dual_informed_targeting | killed | — | Uniform per-net congestion → no ranking change |
| real_proxy_feedback | killed | — | Congestion weighting inflates HPWL without reducing congestion |
| lp_navigate_reweight | done | 1.4973 | Splitting nav budget across 3 rounds cancels benefit |

**Lesson:** LP-level congestion modifications are washed out by 50s of navigation.

### Combine

Three-way combine (SP3+SP1+SP4 winners) scored 1.4918, tying sp4_mccormick_area. Changes don't compound.

### Strategic implications

1. The system is at a plateau. 22 independent experiments all within ±0.5% of baseline.
2. Navigation dominates everything upstream.
3. SDF init is not the bottleneck.
4. Congestion is structural, not parametric.

---

## Miftari / cheap-signal cluster screening (Apr 16) — KILLED

Tested the hypothesis that cheap signals from a single solved "center" LP can rank-order nearby polyhedra for proxy quality (the NASA-star-system analogy). Two-step verification on ibm01.

**Step 1 (experiment 1):** cheap signals vs ΔLP-HPWL across 120 cluster flips (k ∈ {1, 2, 5, 10}). Best signal `S_viol` (constraint violation at x*) → Spearman **ρ = 0.86**, 51 µs vs 2.2 s LP (≈42 700× speedup), 91% precision/recall at 50% kept.

*The cheap signals are tight predictors of ΔLP-HPWL.*

**Step 2 (experiment 3):** LP-HPWL vs final refined proxy across 24 feasible topologies (Hamming 0–2249 from base).

| correlation | ρ |
|---|---:|
| LP-HPWL → refined proxy | **−0.001** |
| LP-HPWL → refined WL | +0.852 |
| LP-HPWL → refined density | −0.536 |
| LP-HPWL → refined congestion | +0.072 |
| refined WL → refined density | −0.416 |
| refined congestion → refined proxy | +0.825 |

*LP-HPWL carries no usable information about refined proxy.* HPWL and density anti-correlate physically (tighter WL ⇒ denser ⇒ worse density cost); the two effects nearly cancel. Proxy on ibm01 is congestion-dominated (ρ = 0.83) and LP-HPWL is uncorrelated with congestion.

**Chain:** `cheap signal → LP-HPWL → refined proxy`; first link ✓ ρ=0.86, second link ✗ ρ≈0.

**Implications:**
- S_viol filtering is not a valid polyhedron-quality screen.
- Re-confirms overnight SP1 finding at the signal level: HPWL-based topology ranking is blind to the objective.
- A proxy-predicting cheap signal would need density + congestion components.

Cost: ~2 hours. Saved building a multi-level cascade on a signal that doesn't track the objective. **This is the diagnostic that triggered the DPO pivot** — combined with E8's 6/20/74 decomposition, it falsified every LP-only architecture.

---

## Champion lineage summary

| Era | Method | Best avg (--all) | Date | Replaced because |
|---|---|---|---|---|
| Pre-history | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Phase 1-5 | Polyhedra Navigation | 1.4867 | 2026-04-15 | Hit ceiling — congestion barrier structural (Miftari ρ=-0.001) |
| Phase 6 (DPO v1) | DPO v1 | 1.4255 | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| Phase 6 (DPO best) | best_of_v2 | 1.3834 | 2026-04-26 | Within-DPO refinements cap at 1-2% |
| Phase 7 | CDOnly (fixed 600s) | 1.1193 | 2026-04-27 am | Fixed budget left hard benchmarks mid-descent |
| Phase 7+ | **CDAdaptive (E9)** | **1.1055** | 2026-04-27 pm | (current champion, beats leaderboard 1.1172 by -1.05%) |

---

## See also

- `writeup/closing_the_gap.md` — leaderboard-beat narrative + E3/E4 algorithm sketches
- `writeup/dpo.md` — DPO theory and architecture
- `writeup/theory.md` — polyhedra theory, barrier analysis
- `writeup/cd_ibm10_results.md` — E2 single-bench CD breakthrough
- `writeup/experiment_notes.md` — DPO ablations, RUDY analysis, dead ends
- `docs/results.md` — current champion (CDAdaptive) only
- `docs/experiment_index.md` — rigorous catalog including falsified
