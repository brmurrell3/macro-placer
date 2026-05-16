# B-R2: Custom Canonical Losses in DREAMPlace's obj_fn

**Status:** SUBSTANTIALLY FALSIFIED as of 2026-05-14 06:15 UTC after
patch correctness fix.

## Critical correction (2026-05-14 04:00 UTC)

Most Phase 1/2/3 data was invalid: the B-R2 patch on cloud's
`PlaceObj.py` was silently overwritten by a later `make install` and
not reapplied. After verification, two bugs were found in the patch
itself:
  1. `n_canon = num_movable_nodes - num_filler_nodes` was wrong;
     correct is `n_canon = num_movable_nodes` (the property already
     excludes terminals; fillers are added beyond movable).
  2. `data_collections.node_size_y` is sometimes empty during obj_fn;
     use `placedb.node_size_y` (numpy, persistent) instead.

Phase 2 v3 (re-run with corrected patch) is the first valid B-R2
hybrid validation.
**Hypothesis:** Adding differentiable canonical losses (top-K density,
RUDY congestion) to DREAMPlace's `obj_fn` produces a better basin that,
when polished, beats stock-DP-polish on the canonical proxy.

## TL;DR — what happened

Built canonical_losses.py (diff top-K + diff RUDY, torch-native).
Patched cloud DREAMPlace's `PlaceObj.py` to read `BR2_LAMBDA_TOPK` and
`BR2_LAMBDA_RUDY` env vars and add weighted canonical-loss terms to
`obj_fn`. Tested on ibm01/ibm10 with multiple lambda configs:

| Bench | λ_topk | λ_rudy | DP basin | post-legal | vs stock-legal |
|---|---:|---:|---:|---:|---:|
| ibm01 stock | 0 | 0 | 1.030 | 1.141 | — |
| ibm01 B-R2 | 1.0 | 0.5 | **0.953** | **1.097** | **−3.9 %** |
| ibm01 B-R2 | 0.5 | 0.25 | 1.014 | 1.090 | −4.5 % |
| ibm10 stock | 0 | 0 | 1.218 | 1.513 | — |
| ibm10 B-R2 | 1.0 | 0.5 | 1.225 | 1.508 | tied |
| ibm10 B-R2 | 0.5 | 0.25 | 1.207 | **1.380** | **−8.8 %** |

Encouraging signal on the post-legalize numbers: -4 to -9% lift on
basin quality.

## Phase 1: B-R2 DP basin + full cascade-equivalent polish

Tested whether the basin lift survives the polish pipeline (greedy
legalize → CD-adaptive → LNS-gridbin → SA-v2 → cascading saddle).

| Bench | B-R2 (1.0, 0.5) | Hybrid baseline | Δ |
|---|---:|---:|---:|
| ibm01 | 0.86829 | 0.870 | −0.20 % (tied) |
| ibm17 | 1.29112 | 1.296 | −0.4 % (tied) |
| ibm02 | 1.12247 | 1.098 | **+2.3 %** |
| ibm10 | 1.13455 | 1.029 | **+10.4 %** |
| ibm12 | 1.21842 (Phase 2; Phase 1 had bug) | 1.147 | **+7.6 %** |

ibm10 with low lambdas (0.5/0.25): 1.094, still +6.3 % regression.

**Finding 1:** CD-adaptive polish dominates over DP basin choice on
easy benches — the B-R2 basin lift is mostly washed out (ibm01,
ibm17). On hard benches (ibm10, ibm12), DP-only-path regresses badly
because DP basin is the wrong starting point, period.

## Phase 2: hybrid placer with B-R2 DP lane

Built `cascade_dp_br2_placer.py` (copy of submissions/cd_lns_sa_cascade_dp_lane
with `_try_run_dreamplace` swapped to native DP + B-R2 patch). Same
best-of-3 plateau pick + cascade saddle. Tested with topk=0.5, rudy=0.25.

| Bench | Hybrid + B-R2 lane | Hybrid baseline (stock DP) | Δ | DP-B-R2 plateau |
|---|---:|---:|---:|---:|
| ibm10 | 1.036 | 1.029 | +0.6 % | 1.151 |
| ibm14 | 1.222 | 1.215 | +0.6 % | 1.282 |
| ibm18 | 1.361 | 1.349 | +0.9 % | 1.516 |
| ibm12 | 1.234 | 1.147 | **+7.6 %** | **2.125 (catastrophic)** |
| ibm17 | 1.375 | 1.296 | +6.1 % | (DP skipped or failed) |

**Finding 2:** Even with plateau-pick fallback, hybrid+B-R2 regresses
on every tested bench. Reason: B-R2's canonical loss terms at λ=0.5/0.25
overwhelm DP's primary objective (HPWL + eDensity), causing DP to land
in MUCH WORSE basins (proxy 1.5-2.1 vs stock DP 1.1-1.3). The plateau
pick correctly rejects the bad DP-B-R2 plateau and falls back to E25/E41,
but in doing so loses the otherwise-useful stock DP path. Net result:
hybrid+B-R2 acts like hybrid-without-DP-lane, which is worse than
hybrid-with-stock-DP on benches where DP wins the plateau (ibm12, ibm17).

## Why B-R2 disrupts DP

DP's obj_fn = wirelength + density_weight × eDensity. With density_weight=8e-5,
both terms have similar magnitude (~1e3-1e4). Our calibration sets
λ_topk such that λ_topk × topk_density ≈ wirelength at the initial
placement. But as DP optimizes, wirelength drops by 5-10× while
top-K density doesn't drop as fast — the canonical term becomes
DOMINANT mid-optimization, pulling DP toward a top-K-optimal but
HPWL/eDensity-suboptimal basin. The polished result is worse on the
canonical proxy than stock DP's polished result.

## Phase 2 v3 (corrected patch, λ_topk=1.0, λ_rudy=0.0)

After fixing both patch bugs, re-ran hybrid + B-R2 on 5 hard benches:

| Bench | Hybrid + B-R2 (λ=1.0) | Hybrid baseline | Δ | DP-B-R2 plateau |
|---|---:|---:|---:|---:|
| ibm10 | 1.033 | 1.029 | +0.4 % | 1.326 (worse than stock ~1.10) |
| **ibm12** | **1.143** | 1.147 | **−0.3 %** | 1.172 (slightly better) |
| ibm14 | 1.229 | 1.215 | +1.2 % | 1.253 (slightly better) |
| ibm17 | 1.365 | 1.296 | +5.3 % | skipped (timeout/short) |
| ibm18 | 1.370 | 1.349 | +1.6 % | 1.591 (worse) |

**Pattern:** B-R2 with λ=1.0 produces a real DP-basin shift on some
benches (ibm12 better, ibm14 better, ibm10/ibm18 worse). The net effect
through the polish pipeline + plateau pick + cascade saddle is
**mixed: 1 tiny win, 4 regressions**. Aggregate slightly worse.

## Phase 2 v4 (completed: λ_topk=0.1, 10× smaller)

5 hard benches, λ_topk=0.1, λ_rudy=0 (RUDY skipped — per-net Python
loop too slow for DP iteration cadence; vectorization deferred).

| Bench | v4 (λ=0.1) | v3 (λ=1.0) | Hybrid baseline | Δ vs baseline | DP-B-R2 plateau |
|---|---:|---:|---:|---:|---:|
| ibm10 | 1.035 | 1.033 | 1.029 | +0.6 % | 1.115 |
| **ibm12** | **1.129** | 1.143 | 1.147 | **−1.6 % ✓** | 1.160 |
| ibm14 | 1.234 | 1.229 | 1.215 | +1.6 % | 1.281 |
| ibm17 | 1.361 | 1.365 | 1.296 | +5.0 % | (skipped/timeout) |
| ibm18 | 1.368 | 1.370 | 1.349 | +1.4 % | 1.495 |

**5-bench aggregate:** B-R2 1.2254 vs baseline 1.2072 = **+1.5 % regression**.

**Verdict:** B-R2 produces a real but inconsistent basin lift — wins
on ibm12 (DP-favorable hard bench, −1.6 %), loses on most others. The
net effect is roughly flat or slightly negative. Cannot bridge the
−7 % gap needed for sub-1.0.

## What does work, hypotheses (for follow-up phases)

**Per-bench adaptive λ** (rule-compliant via observable formula):
Different benches need different λ. ibm12 likes λ=0.1; ibm10/18 need
even smaller. A formula like `λ = 0.1 / macro_density` or
`λ = clip(0.5 * (1 - macro_density), 0.01, 0.2)` might give bench-
appropriate strength. To be tested.

**Quad-lane hybrid** (best-of-4: E25, E41, stock-DP, B-R2-DP):
Currently B-R2 displaces stock-DP. If B-R2 produces a different basin
than stock-DP, keeping BOTH and letting plateau pick choose could
capture B-R2's ibm12 lift without losing stock-DP's wins elsewhere.
Wall-budget tight (4 × 660s + saddle).

**RUDY vectorization** (deferred): per-net Python loop is the
bottleneck. Replace with scatter_reduce ops. Then retest if RUDY
adds basin lift orthogonal to top-K.

## Coordination note

Other agent (cd_lns_sa_cascade_*_levy lineage) is actively running
E100_dual_levy `--all` on the cloud as of 2026-05-14 03:18 EDT.
This work-stream uses cloud DREAMPlace install in
`~/DREAMPlace_cpu/install/` (patched) and `experiments/B_R2_canonical_dp_loss/*`
and `submissions/cd_lns_sa_cascade_dp_br2/*` (new dir, no overlap).
Throttling to 2 parallel procs × 4 threads = 8 cores while other
agent runs.

## Phase 3 (portfolio placer, 4 lanes) — 2026-05-14 18:30 UTC

Built `cascade_dp_portfolio_placer.py` (E25 + E41 + stock-DP + B-R2-DP).
Idea: portfolio plateau pick lets each bench select its own best
basin without per-bench tuning. Tested on 5 hard benches with
`BR2_LAMBDA_PORTFOLIO_TOPK=0.05`.

| Bench | Portfolio | Hybrid baseline | Δ |
|---|---:|---:|---:|
| ibm10 | 1.041 | 1.029 | +1.2 % |
| **ibm12** | 1.234 | 1.147 | **+7.6 %** |
| ibm14 | 1.240 | 1.215 | +2.1 % |
| ibm17 | 1.357 | 1.296 | +4.7 % |
| ibm18 | 1.340 | 1.349 | −0.7 % |

**5-bench aggregate: +2.9 % regression.** Portfolio approach FAILED.
Issues:
- Tighter per-lane CD budget (0.10 vs 0.12) hurts polish quality
- Cascade saddle compressed to 0.18×B (vs 0.29×B) — less escape
- Stock-DP basin quality went bad on ibm12 (2.13 proxy — likely
  contention with other agent's runs or a deeper interaction with
  the patched PlaceObj.py even at λ=0)
- B-R2-DP basin worse than stock on most benches anyway

## FINAL B-R2 VERDICT (2026-05-14 19:15 UTC)

**B-R2 is FALSIFIED in every variant tested:**
- B-R2 DP-only (Phase 1): regression on hard benches
- B-R2 hybrid with single-λ (Phase 2 v3/v4): mixed; +1.5 % regression
  on 5-bench aggregate
- B-R2 portfolio (4-lane plateau): +2.9 % regression on 5-bench aggregate

**The fundamental issue:** B-R2 occasionally produces a real basin
lift (ibm12 -1.6 % at λ=0.1, ibm18 -1.3 % at λ=0.05) but no single
configuration generalizes. Per-bench optimal λ is forbidden by rules
("must be general algorithm"). Portfolio captures cross-bench best
in principle but in practice the budget split between extra lanes
costs more than the lift gains.

**Comparison to current best variants** (17-IBM aggregate):
- PATH A (submitted, `placer_adaptive.py`): **1.0770**
- Hybrid baseline (`cascade_dp_lane`): **1.0668** (−1.0 % vs PATH A)
- E100_dual_levy (other agent): **1.0732** (−0.4 % vs PATH A)
- B-R2 hybrid (Phase 2 v4, partial): **+1.5 %** vs hybrid baseline = worse

**Target to win Tier 1 ($20K):** sub-1.0109. Gap from hybrid 1.067
is −6.0 %. Neither B-R2 nor levy variants close this gap.

## What's left (post-B-R2)

Remaining ~7 hours to user return at 23:00 EDT. Concrete moves:
- **Multi-seed**: run hybrid with multiple random seeds, pick best
  per bench (requires modifying shared placer; coordination risk).
- **Tier 2 ORFS**: validate NG45 placements pass feasibility gate
  — wins $20K Grand Prize independent of proxy rank.
- **Reproduce vmallela 1.011**: their "Incremental CD+LNS" suggests
  same class but better tuning. Compare iteration counts, SA
  schedules. Without their code this is informed speculation.

Cloud is currently still running other agent's E100_dual lineage.
My B-R2 queue is fully terminated.

Hypothesis: smaller λ produces a gentler canonical-signal nudge that
doesn't disrupt DP's primary HPWL/eDensity objective on benches where
it overshoots (ibm10, ibm18). Best case: B-R2 ≈ hybrid baseline on
problematic benches, with small lift on receptive ones.

Status: in progress, ETA 2026-05-14 07:10 UTC.

## Verdict (preliminary)

B-R2 in its current formulation is NOT a path to sub-1.0. The
calibration problem is harder than expected, and even with optimal
lambdas the post-CD-polish lift may be too small to matter (the
ibm01 -3.9 % basin lift collapsed to -0.2 % post-polish).

**What worked:**
- canonical_losses.py (diff top-K density, diff RUDY) is correct and
  autograd-friendly.
- PlaceObj.py patch via env vars cleanly hooks into DREAMPlace.
- Native DREAMPlace build on cloud (with CXX11 ABI=1 fix) works.

**What didn't:**
- Single fixed lambda doesn't generalize across benches.
- The post-polish lift is too small even when basin lift is real.
- Hard benches (ibm12, ibm17) regress catastrophically when DP is
  disrupted, even with plateau-pick fallback.

## Code artifacts (kept for reference)

- `code/canonical_losses.py` — diff top-K + RUDY, autograd-friendly
- `code/placeobj_patch.py` — monkey-patch helper
- `code/canonical_placer.py` — pure-Python Nesterov placer (smoke test failed +0% from SDF init)
- `code/dp_light_smoke.py` — pure-Python Adam smoke test (E88-style)
- `code/native_dp_runner.py` — native DREAMPlace runner with B-R2 env vars
- `code/dp_br2_full_polish.py` — DP-only + full polish driver
- `code/cascade_dp_br2_placer.py` — hybrid placer with B-R2 DP lane

Patched on cloud at `~/DREAMPlace_cpu/install/dreamplace/PlaceObj.py`
(B-R2 patch inserted; reverts to stock when env vars unset/zero).

## Implications for sub-1.0 push

Need a different path. Current hybrid at 1.067 IBM is +5.7 % above
leaderboard top 1.0109. Closing that gap requires either:
- A new structural idea (not B-R2)
- Multi-seed restart at scale (lower confidence; +0.5-1 % typical)
- Reverse-engineering vmallela or DREAMPlaceProMaxUltra (private)
- ORFS Tier 2 path ($20K), where competitive NG45 quality may win
  the Grand Prize independent of proxy rank
