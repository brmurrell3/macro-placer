# E91: PATH B B-R0' — Stock DP + Full Polish vs Cascade

**Status:** hybrid `--all` validated 2026-05-13 10:52 UTC.
**Cloud:** lambda.ai A100 box at 129.213.18.245

## TL;DR (final)

Hybrid placer `submissions/cd_lns_sa_cascade_dp_lane/placer.py` validated
on full 17 IBM + 4 NG45 against cascade A4v2 (post-A1, post-LNS-skip)
reference.

| Set | Hybrid | Cascade A4v2 | Lift |
|---|---:|---:|---:|
| IBM 17 | **1.06687** | 1.07711 | **−0.95 %** |
| NG45 4 | **0.68086** | 0.68483 | **−0.58 %** |

Big wins (DP basin substantially better than SDF/DPO):
- ibm12 −5.35 %, ibm17 −3.84 %, ibm07 −3.13 %, mempool_tile −2.28 %,
  ariane133 −2.05 %, ibm09 −2.52 %, ibm03 −2.46 %, ibm01 −2.11 %.

Regressions (DP polishes worse, eats saddle budget):
- ibm02 +2.99 %, ariane136 +1.76 %, ibm10 +1.28 %, ibm18 +1.13 %,
  ibm08 +1.00 %, ibm16 +0.61 %, nvdla +0.49 %.

NG45 generalization works with auto-adaptive `target_density = clip(
macro_density * 1.5, 0.40, 0.85)` plus two-stage retry-on-illegal.
ariane133 lands 0.66167 — below the E74 reference 0.6641.

## Bottom line so far

**The original PATH B autopsy was wrong.** It compared DP basins (no polish)
vs cascade post-polish, then concluded "structural objective mismatch
can't be bridged." But basin-only quality doesn't predict polished
quality in this codebase — polish curves are typically −22 to −27 %.

Test: stock DP → greedy_macro_legalize → full E25 polish pipeline
(CD-adaptive + LNS-gridbin + SA-v2) → cascading saddle escape. Same
polish budgets as cascade gives its own SDF/DPO inits.

## Results

| Bench | DP basin | + legal | + CD | + LNS | + SA | + cascade | Cascade-capped ref | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ibm01 | 1.011 | 1.047 | 0.876 | 0.875 | 0.875 | **0.862** | 0.85 (uncapped) | **+1.4 %** |
| ibm09 | 0.954 | 0.955 | 0.799 | 0.796 | 0.794 | **0.789** | (unknown — not in autopsy) | — |
| ibm10 | 1.215 | 1.444 | 1.105 | 1.099 | 1.099 | **1.095** | 1.0775 | **+1.6 %** |
| ibm12 | 1.442 | 1.477 | 1.149 | 1.147 | 1.146 | **1.129** | 1.3031 | **−13.3 %** |
| ibm14 | 1.494 | 1.523 | 1.263 | 1.261 | 1.261 | **1.243** | 1.2919 | **−3.8 %** |
| ibm17 | 1.543 | 1.572 | 1.313 | 1.312 | 1.312 | **1.307** | 1.4546 | **−10.2 %** |

**Update 2026-05-13 00:15 UTC — ALL 6 IBM BENCHES FINAL + ariane133 NG45:**

| Bench | DP+full-polish | Cascade ref | Δ | Verdict |
|---|---:|---:|---:|---|
| ibm01 | 0.862 | 0.85 (uncapped) | +1.4 % | tie |
| ibm09 | 0.789 | (no ref) | — | likely win |
| ibm10 | 1.095 | 1.0775 (capped) | +1.6 % | marginal lose |
| ibm12 | **1.129** | 1.3031 (capped) | **−13.3 %** | **strong win** |
| ibm14 | **1.243** | 1.2919 (capped) | **−3.8 %** | win |
| ibm17 | **1.307** | 1.4546 (capped) | **−10.2 %** | **strong win** |
| **ariane133** | **0.741 (3 overlaps)** | 0.664 (E74 ref) | **+11.6 % AND INVALID** | **FAILS NG45** |

**IBM hard-bench aggregate** (ibm10/12/14/17):
- Cascade-capped average: 1.2818
- DP+full-polish average: 1.1936
- **Lift: −6.9 %**

**Headline (IBM):** the original autopsy is empirically wrong on 3 of 4
hardest IBM benches. ibm12 and ibm17 polish to double-digit lift vs
cascade-capped.

**CRITICAL NEGATIVE FINDING (NG45):** ariane133 result is **invalid** —
3 hard-macro overlaps remain after the full polish pipeline. The greedy
legalizer left 14 residuals (failed 5 of 71 moves), CD-LNS-SA worked on
near-legal placement (proxy improved to 0.723) but the subsequent
project_overlaps and cascade saddle left 3 overlaps. The whole NG45
pipeline assumes IBM-style tolerances (overlap_threshold 0.004 vs
ariane133's 0.000) and doesn't gracefully handle commercial-cell
geometry.

**Implication:** PATH B is an **IBM-only** lift in current form. NG45
needs either (a) a stricter legalize pipeline or (b) graceful fallback
to E25/E41 lanes. Hybrid placer has been patched to exclude invalid DP
output from plateau pick (returns None when ovl > 0 after polish).
Without that patch, hybrid would crash on ariane133. With the patch,
hybrid on NG45 = standard cascade (no DP lift, no regression).

## Update 2026-05-13 01:30 UTC — AUTO-ADAPTIVE DP CONFIG (rule-compliant)

The first NG45 fix used a per-benchmark branch (`if bench.name.startswith("ibm")`)
which is **against challenge rules** (no per-benchmark tuning). Replaced
with a single algorithm: `target_density = clip(macro_density * 1.5, 0.40, 0.85)`
where `macro_density = total_macro_area / canvas_area`. Same formula
applied to every input. Plus `stop_overflow=0.02`, `iter=2000`,
`lr=0.005` universally.

ariane133 result with auto-adaptive config (FINAL):
- macro_density = 0.496 → target_density = 0.743 (auto)
- DP basin: 0.760 (1 overlap; was 1.153 / 78 ovls with original config)
- Post-greedy-legalize: 0.817 (0 overlaps — no extended_legalize needed)
- + CD: 0.675, + LNS: 0.674, + SA: 0.674, + plateau: 0.674
- **+ cascade saddle: 0.66993 (0 overlaps), total wall 2029 s = 34 min**

**vs cascade-uncapped reference 0.6641: +0.88 % (basically tied)**.

NG45 generalization works with the auto-adaptive DP config. The first
ariane133 attempt (target_density=0.85) left 3 overlaps and proxy 0.741.
The auto-adaptive version reaches 0.670 with 0 overlaps in 34 min — half
the budget.

## Update 2026-05-13 10:52 UTC — HYBRID FULL 17 IBM + 4 NG45

Final overnight queue with auto-adaptive DP, two-stage retry, and
budget-aware cascade saddle. Hybrid placer:
`submissions/cd_lns_sa_cascade_dp_lane/placer.py`.

Reference: cascade `a4v2_postLNS` 17-IBM avg 1.07711 (PATH A's
post-A1+LNS-skip optimization on `placer_adaptive.py`).

### IBM 17 (full)

| Bench | Hybrid | Cascade A4v2 | Δ | Winner lane |
|---|---:|---:|---:|---|
| ibm01 | 0.86991 | 0.88864 | **−2.11 %** | cascade(DP) |
| ibm02 | 1.09793 | 1.06610 | +2.99 % | cascade(DP) |
| ibm03 | 0.92677 | 0.95013 | **−2.46 %** | cascade(DP) |
| ibm04 | 0.98942 | 0.98919 | +0.02 % | cascade(DP) |
| ibm06 | 1.13398 | 1.13495 | −0.09 % | cascade(E25) |
| ibm07 | 1.03869 | 1.07224 | **−3.13 %** | cascade(DP) |
| ibm08 | 1.11295 | 1.10192 | +1.00 % | cascade(E41) |
| ibm09 | 0.79995 | 0.82066 | **−2.52 %** | cascade(DP) |
| ibm10 | 1.02775 | 1.01476 | +1.28 % | cascade(E41) |
| ibm11 | 0.88031 | 0.87782 | +0.28 % | cascade(DP) |
| ibm12 | 1.14699 | 1.21182 | **−5.35 %** | cascade(DP) |
| ibm13 | 0.94123 | 0.95727 | −1.68 % | cascade(E41) |
| ibm14 | 1.21512 | 1.21921 | −0.34 % | cascade(DP) |
| ibm15 | 1.16254 | 1.18317 | **−1.74 %** | cascade(E41) |
| ibm16 | 1.14812 | 1.14116 | +0.61 % | cascade(E41) |
| ibm17 | 1.29611 | 1.34781 | **−3.84 %** | cascade(DP) |
| ibm18 | 1.34895 | 1.33394 | +1.13 % | cascade(?) |
| **avg** | **1.06687** | **1.07711** | **−0.95 %** | |

### NG45 4

Cascade reference here is the wall-uncapped cascade from 2026-05-11
(longer effective budget than the hybrid 3300 s cap allows).

| Bench | Hybrid | Cascade uncapped | Δ |
|---|---:|---:|---:|
| ariane133 | 0.66167 | 0.67554 | **−2.05 %** |
| ariane136 | 0.66071 | 0.64932 | +1.76 % |
| mempool_tile | 0.72071 | 0.73746 | **−2.28 %** |
| nvdla | 0.68033 | 0.67698 | +0.49 % |
| **avg** | **0.68086** | **0.68483** | **−0.58 %** |

ariane133 also beats the E74 hessian-saddle reference (0.6641) by
−0.36 %. This is significant — ariane133 had been the failure point
that killed E42/43/44/54/62. Auto-adaptive DP unlocks a slightly
deeper basin than E74's hessian path.

### Plateau-pick distribution (winner lanes after best-of-3 + saddle)

- **DP wins plateau on:** ibm01, ibm02, ibm03, ibm04, ibm07, ibm09,
  ibm11, ibm12, ibm14, ibm17 (10/17) — strongest on ibm12, ibm17.
- **E41 wins plateau on:** ibm08, ibm10, ibm13, ibm15, ibm16 (5/17).
- **E25 wins plateau on:** ibm06 (1/17).
- **ariane136 (NG45):** E25 wins plateau; DP polishes worse and eats
  saddle budget — direct cause of the +1.76 % regression.

The pattern: DP basin is genuinely different from SDF/DPO basin, and
when DP-polish lands a better proxy, hybrid captures it. When DP-polish
loses, hybrid still picks E25/E41 plateau — but the DP lane has
consumed ~22 % of the wall budget without payoff, leaving less for the
cascade saddle. ariane136 saddle got 1199 s (1 iter); ibm18 saddle got
similarly truncated.

### Wall-budget violations

ibm17 ran 3655 s = 60.9 min (over 60 min hard cap by 55 s). Cause:
ibm17's E25 + E41 phases took 3113 s, leaving only 177 s for saddle.
Saddle ran 1 iter and the eval harness post-processing (validate,
write JSON, log) pushed past 60 min. **For submission use:** tighten
the per-lane CD `cd_hard_cap_s` from 0.14×B to 0.12×B and reduce SA
budget to give cascade saddle a stricter floor of ≥300 s. Or skip DP
on the largest benches if remaining time after E41 is too short.

### Bottom line for first place

Hybrid combined IBM+NG45 21-bench avg: (1.06687 × 17 + 0.68086 × 4) /
21 = **0.9933**.

Cascade A4v2 IBM 1.07711 was a −5.3 % lift vs the original 1.137 floor
(PATH A's contribution from A1 speedup). Hybrid adds another −0.95 %
on IBM, −0.58 % on NG45. Cumulative `placer_adaptive.py` (1.137)
→ A4v2 (1.077) → hybrid (1.067) = **−6.2 % combined lift** since the
session started.

Gap to leaderboard top (~1.01 on IBM): hybrid 1.067 = **+5.7 %**. Still
above first place. Need additional lift to close, but hybrid is the
new submission-floor candidate.

**Recommendation:** promote `cd_lns_sa_cascade_dp_lane/placer.py` to
submission floor once wall-budget violations are resolved and a clean
`--all` re-run validates. Pull `cd_lns_sa_cascade/placer_adaptive.py`
as fallback. E96 multi-config DP variant (other agent) projected to add
another −0.5 to −1.5 % on top.

## Update 2026-05-13 01:30 UTC — HYBRID IBM (6 benches done, archive)

Hybrid placer uses best-of-{E25, E41, DP-polished} plateau pick + cascade saddle.
These runs used the **pre-fix** hardcoded `target_density=0.85` DP config
(cached at process start before the rule-compliant patch). Phase 4 of the
overnight queue reruns hybrid with auto-adaptive config.

| Bench | Hybrid winner | Hybrid proxy | Cascade-capped | Δ vs cascade-capped |
|---|---|---:|---:|---:|
| ibm01 | cascade (saddle on DP plateau) | 0.867 | 0.85 (uncapped) | +2.0 % |
| ibm09 | cascade (saddle on DP plateau) | 0.799 | (no ref) | — |
| ibm10 | cascade (saddle on E41 plateau) | **1.029** | 1.0775 | **−4.5 %** |
| ibm12 | cascade (saddle on DP plateau) | **1.145** | 1.3031 | **−12.2 %** |
| ibm14 | cascade (saddle on E41 plateau) | **1.225** | 1.2919 | **−5.2 %** |
| ibm17 | cascade (saddle on DP plateau) | **1.308** | 1.4546 | **−10.1 %** |

**Hard-bench aggregate** (ibm10/12/14/17): hybrid 1.177 vs cascade-capped
1.282 = **−8.2 % lift**.

Plateau-pick behavior:
- DP wins plateau on ibm12, ibm17 (the strong-win benches from B-R0').
- E41 wins on ibm10, ibm14.
- DP wins on ibm01, ibm09 (smaller benches; smoke-ish, less informative).

The plateau pick is doing its job: best-of-3 captures the lift wherever
it is. Cascade saddle on the best plateau adds another 1-2 % typical.

**Budget concerns:** ibm17 ran 62 min total (over 55-min internal budget,
within 60-min hard cap). ibm10/14 also ran 58-58 min. The new
budget-aware cascade saddle (rolling avg iter wall) is on disk but
those running processes had cached the old module. Phase 4 of overnight
queue will use the new code.

**Cascade saddle adds value:** on ibm10 (-0.3 % over plateau), ibm12 (-1.5 %),
ibm14 (-2.1 %), ibm17 (-1.4 %). Not huge but real.

**Budget violations:** ibm10 wall 57 min, ibm14 wall 57 min, ibm17 wall
59 min — over 55-min internal budget but within 60-min hard cap. Cause:
cascade saddle's last iteration can extend past budget by 200-500 s
because budget check happens at iter start, not within iter. Stricter
deadline enforcement needed for submission use.

**Implication for hybrid placer** (`submissions/cd_lns_sa_cascade_dp_lane/placer.py`):
plateau pick of best-of-{E25, E41, DP+polish} should pick DP on
ibm12/14/17 (where DP wins big) and SDF/DPO on ibm10 (where DP loses
marginally). Projected aggregate lift on `--all` ~−2 to −4 % vs cascade
alone.

**Implication for first place:** DP basins land in *substantially
different valleys* than cascade SDF/DPO inits — sometimes much better
under the canonical proxy. Hybrid (best-of-{E25, E41, DP+polish})
should lift aggregate `--all` proxy meaningfully on hard benches.
Scaffold at `submissions/cd_lns_sa_cascade_dp_lane/placer.py` (built,
not validated on `--all`).

**ibm01: only +1.4 % above cascade uncapped, despite CPU-only DP and a
short 600 s budget.** The polish hypothesis is empirically alive.

## What this means for first place

### Quantitative projection (with cascade-saddle yet to land)

Estimated cascade-saddle lift from plateau: ~−1.5 % (based on ibm01
0.877 → 0.862). Applied to current plateaus:

| Bench | Plateau | Projected final | Cascade-capped | Δ |
|---|---:|---:|---:|---:|
| ibm10 | 1.112 | ~1.10 | 1.0775 | +2.1 % |
| ibm12 | 1.157 | ~1.14 | 1.3031 | **−12.5 %** |
| ibm14 | 1.269 | ~1.25 | 1.2919 | **−3.2 %** |
| ibm17 | ~1.32 (est) | ~1.30 | 1.4546 | **−10.6 %** |

Sum of cascade-capped on hard 4 = 5.122; sum of projected DP+full
polish = 4.79; lift on hard 4 = **−6.5 %**.

Applied to `--all` 17-bench cascade-capped aggregate 1.137 (assuming
13 easier benches roughly tied between cascade and DP+polish): hybrid
expected aggregate **~1.115 = −1.9 % lift**.

Hybrid: ~1.115 aggregate. Leaderboard top ≈ 1.01. Gap: +10 % (down
from current +12.6 %).

**Honest P(first place):** stays low, ~10-15 % even with hybrid lift.
**P(top-3):** improves materially, ~35-40 %. The hybrid is a
**submission floor lift**, not a first-place lever.

For actual first place: need either (1) better-than-stock DP via
B-R1 / B-R2 / B-R3 (custom losses), (2) a different method we
haven't articulated, or (3) the leaderboard top doing nothing we
can't replicate. B-R1 onward is the only credible path within our
toolset.

## What's NOT changed

- The original sweep was correct as a basin-quality measurement.
- DP basin quality alone is worse than cascade post-polish — that's
  the data the autopsy reported.
- The interpretation ("can't be bridged by polish") was untested.

## B-R4 status: FALSIFIED

Cascade-as-init → DP perturber → cascade-repolish: both iter=100 and
iter=10 produced proxy 6.18 with 308 505 overlaps (= every hard-macro
pair overlapping on ibm10). DP's gradient on a canonical-optimal
init pulls the placement apart — not a small perturbation, a complete
destabilization. Either the bookshelf init write-back didn't take or
DP's Nesterov accelerates immediately to a "wrong" basin.

## Pointers

- Driver: `code/dp_full_polish.py`
- Cloud cmd: `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 DP_USE_GPU=0 DP_DOCKER_IMAGE=dreamplace:custom DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install python3 experiments/E91_dp_full_polish/code/dp_full_polish.py <bench> <budget>`
- Falsified B-R4: `code/dp_cascade_perturb.py`
- B-R1 scaffold: `experiments/E92_dp_tilos_rudy/`
- B-R2 scaffold: `experiments/E93_dp_topk_density/`
- B-R3 scaffold: `experiments/E94_dp_canonical_full/`
