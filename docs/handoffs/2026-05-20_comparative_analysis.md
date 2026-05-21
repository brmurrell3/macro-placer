# Comparative analysis: thinkorplace-v2 vs top-2

How we stack up against the top public-leaderboard entries and where the
remaining gaps live.

## The numbers

| Placer | Avg IBM | Time/bench | Stack |
|---|---:|---:|---|
| Carrotato | **0.967** | 3.8 min | Xplace + Triton kernels + polish |
| Shoom | 0.978 | (unknown) | (unknown) |
| vmallela | 1.011 | (unknown) | (unknown) |
| DREAMPlaceProMaxUltra | 1.012 | (unknown) | DREAMPlace + multi-run + CD refinement |
| QuantSC | 1.029 | (unknown) | (unknown) |
| **thinkorplace-v2 (us)** | **1.008 EPYC / 1.003 M3** | **12 min** | V3 smooth descent + CD polish |
| QED | 1.031 | (unknown) | (unknown) |
| Cezar | 1.037 | (unknown) | (unknown) |

Our gap to Carrotato: +4.2%. To MultiDreamPlace: −0.4% (we beat them).

## Profile of our v2 (on ibm17, the hardest bench)

Total wall 357s; breakdown:

```
55% — compute_proxy_cost (canonical eval, called 3× for logging)
       └─ get_ref_node_id × 1.3M (Python dict lookups in PlacementCost)
40% — V3 descent (Adam + per-net trace + greedy legalize)
       └─ 11% — autograd backward
       └─  5% — compute_congestion (per-net trace)
       └─  4% — SDF init
21% — CD polish (run_cd_adaptive)
       └─ 18% — delta_cost_axis_batch
       └─ 11% —   _net_cong_contrib_flat
```

**~30% of our wall is logging overhead** (canonical proxy eval called
multiple times for log lines). Removing redundant calls would free
~100s/bench for more optimization without quality loss — but CD already
plateaus, so this is wall-time efficiency, not quality.

## Cost decomposition (v2 champion --all)

```
bench    proxy    wl     d     c     wl%  d%   c%
ibm01    0.843  0.082  0.540  0.981   10% 32% 58%
ibm17    1.200  0.069  0.531  1.731    6% 22% 72%   ← hard
ibm18    1.236  0.075  0.578  1.742    6% 23% 71%   ← hard
avg      1.003  0.076  0.530  1.324    8% 26% 66%
```

**Congestion dominates: 58-72% of total cost**, especially on hard benches.
Density is uniform (~0.53). WL is small (6-10%).

**This means the path to lower IBM is congestion-bound.** To close the
gap to Carrotato, we need to find macro placements with significantly
lower routing congestion.

## What top-1 (Carrotato) is likely doing differently

**Hypothesis: Xplace + Triton + challenge-objective patch.**

| Component | Carrotato (likely) | Us (v2) |
|---|---|---|
| WL smooth | Weighted-average HPWL | LSE-HPWL (similar) |
| Density smooth | **Electrostatic eDensity (FFT-Poisson)** | Grid-bin top-10% |
| Congestion smooth | Per-net trace + GPU-fast | Per-net trace (CPU autograd) |
| Optimizer | Nesterov + adaptive density-weight ramp | Adam |
| Multi-stage | γ schedule + density schedule | Single γ anneal |
| Speed | Triton kernels (~50× PyTorch) | Vanilla PyTorch |
| Iterations | Many (fast kernel) | ~500 Adam steps |
| Polish | Light (Xplace converges close) | CD (Python, ~10 min) |

**The big technical differences:**
1. **Electrostatic density** (eDensity): globally differentiable, smooth even at cell boundaries. Adam/Nesterov finds smoother basins.
2. **GPU speed via Triton**: 100× more gradient steps per minute → fuller convergence.
3. **Multi-stage scheduling**: density-weight ramp like ePlace converges to lower density without violating overlap constraints.

## What top-4 (MultiDreamPlace) likely does

**Hypothesis: DREAMPlace × N hyperparameters + CD refinement.**

- DREAMPlace optimizes WL + eDensity (NOT challenge's congestion)
- Multiple runs with different density-weights / target-densities
- Best output → CD refinement

**Why we BEAT them**: Our per-net-trace congestion is matched to the
challenge proxy; their DREAMPlace optimizes the wrong objective. They
get the wrong basin.

**Why we still might lose to them on EPYC**: Their iteration is GPU-fast,
they can multi-restart with many hyperparams. Variance reduction.

## Concrete improvement experiments tried tonight (ibm17)

| Variant | num_steps | CD wall | Proxy | wl/d/c |
|---|---:|---:|---:|---|
| v2 baseline | 500 | 600s | **1.200** | 0.068 / 0.531 / 1.731 |
| 300 steps | 300 | 180s | 1.217 | 0.068 / 0.543 / 1.754 |
| 1000 steps | 1000 | 180s | 1.213 | 0.068 / 0.527 / 1.763 |
| 300 steps + 600s CD | 300 | 600s | 1.210 | 0.068 / 0.539 / 1.746 |

**Findings:**
- More Adam steps doesn't help (1000 ≈ 300)
- Longer CD doesn't help (600s ≈ 540s)
- **Congestion floor is ~1.73-1.78** regardless of optimization length
- We're stuck at this congestion floor on ibm17

## Where the 4.2% gap to Carrotato lives

**Decomposition:**
- ~2-3% from our congestion floor (we can't find macro positions with
  much lower routing congestion within our smooth-proxy gradient)
- ~1-2% from our density model (grid-bin is less smooth than eDensity,
  Adam has trouble navigating the bin boundaries)
- <1% from optimizer choice (Adam vs Nesterov-BB)
- <1% from polish ceiling (CD has saturated)

## Improvements ranked by EV in 24 hr

| # | Improvement | EV | Effort | Status |
|---|---|---:|---:|---|
| 1 | **Cascade saddle escape on V3 basin** | 0.5-1% | 1 hr | Easy: V3 basin → saddle escape → polish. Adds ~10 min wall. |
| 2 | **Lower smooth_range / sharper per-net trace** | 0.5-2% | 2 hr | Tune per-net-trace hyperparameters for sharper gradient |
| 3 | **Gaussian-smoothed density** | 1-2% | 3 hr | Replace grid-bin with macro-as-gaussian |
| 4 | **Multi-stage density-weight ramp** | 1-2% | 3 hr | ePlace-style adaptive weighting |
| 5 | **Hybrid: cascade pipeline on V3 init** | 1-3% | 4 hr | V3 finds basin, cascade saturates it |
| 6 | **FFT-Poisson eDensity** (electrostatic) | 2-4% | 1+ day | Big engineering; Carrotato-style |
| 7 | **Triton kernels** | 5-10× speed | 3+ days | Out of scope |
| 8 | **Patched DREAMPlace** | 2-5% | 4-7 days | Out of scope |

## Realistic projection in remaining time

Combining #1-#3 (achievable in ~5 hr):
- Expected lift: 1.5-3% on M3 → 0.985-0.995 M3 → ~0.99-1.00 EPYC
- Projected rank: **#3 → fighting #2** (Shoom 0.978)

To actually BEAT Carrotato 0.967, we'd need #6+ (electrostatic eDensity)
or #8 (patched DREAMPlace) — neither feasible in 24 hr.

## Where to focus the next 24 hours

**Tier 1 (high EV, doable):**
- **#1 Cascade saddle on V3 basin** — quick to test, established technique
- **#5 Hybrid V3+cascade** — let cascade do its saturation pass on V3 init
  - V3 gives the BASIN, cascade gives the POLISH
  - Per-bench wall ~30 min, total ~3 hr at jobs=4

**Tier 2 (if Tier 1 doesn't move the needle):**
- **#2 Tune per-net-trace** (smooth_range, β, σ) — reduce bias to canonical
- **#4 Density-weight ramp** — better convergence

**Tier 3 (after final submission, for future iteration):**
- **#3 Gaussian density**
- **#6 eDensity (longer-term)**

## Recommendation

Spend 3-4 hours on Tier 1 (cascade saddle + V3+cascade hybrid). If
either gets us to M3 ≤ 0.99, ship the new variant. Otherwise ship the
current v2 (M3 1.003, EPYC 1.008).
