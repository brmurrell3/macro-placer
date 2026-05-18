# 2026-05-18 evening — submission decision matrix

## Bottom line

**Ship Option C (cd_lns_sa_cascade_stacked_periphery, 1.05750 IBM / 0.68930 NG45).**
None of today's variant tests beat it cleanly. Hardware variance
(~1.4-2.4% M3 → EPYC) dominates over algorithmic variance (~0.1%).

**Action item:** Edit `placer.py` to switch from Option B to Option C
(switch instructions in [`2026-05-18_next_submission_candidate.md`](2026-05-18_next_submission_candidate.md)).

## Full results from today's variant exploration

All on the same M3 Max hardware (jobs=4) unless noted.

| Variant | Aggregate | Δ vs 1.05750 | Verdict |
|---|---:|---:|---|
| **stacked_periphery (champion)** | **1.05750** | — | Ship this |
| stacked_periphery_K3 (variance test, 2026-05-17) | 1.05689 | −0.06% | Tied — noise |
| tabu_stacked (E99 saddle replacement, 2026-05-18) | 1.05657 | −0.09% | Tied — noise |
| **4w portfolio (add (1,1,1) weight)** | **1.05870** | +0.11% | Tied — noise |
| **finalsa (T₀=1e-3 SA at end)** | **1.05759** | +0.009% | EXACTLY tied — noise |
| no_e41_deep (skip E41 lane) | crashed on ibm08 | n/a | **FALSIFIED** |
| 6w portfolio K=1 max_iters=1 (--fast only) | n/a | −0.12% (--fast) | Marginal; not pursued |
| 6w_k2 (--fast on aws-gpu) | n/a | +1.74% (--fast) | **FALSIFIED on aws-gpu** |
| dualseed (2 seeds, aws-gpu --all) | 1.08280 | +2.42% (mostly hw) | Dualseed mechanism gives <0.05% lift |

**Synthesis:** Five algorithm-tweak variants tested today (4w, finalsa,
tabu, 6w, dualseed). All land within ±0.12% of 1.05750 — the noise
floor. No clean structural win.

## Hardware variance (NEW today)

Pivotal finding: **identical placer on M3 Max vs aws-gpu (AMD EPYC) gives
~1.4-2.4% different aggregates**, much larger than any algorithm tweak.

| Test | M3 | aws-gpu | Δ |
|---|---:|---:|---:|
| stacked_periphery --fast (calibration) | 0.89844 | **0.91108** | +1.41% |
| stacked_periphery --all (proxy from dualseed) | 1.05750 | ~**1.08280** | ~+2.4% |

**Implication for submission:** Judges' AMD EPYC 9655P + RTX 6000 Ada
will produce ~1.07-1.09 NOT our M3-verified 1.05750. Cross-platform
variance is the dominant unknown. Still beats public leaderboard
(1.1172) and Cezar #2 (1.037), but vmallela #1 at 1.011 may be
unreachable from our basin.

## Failure modes observed

1. **Adding more portfolio weights truncates by deadline.** 6w portfolio
   on aws-gpu (slow CPU) ran fewer iters than 3w original → regression.
2. **Single-lane configurations crash.** no_e41_deep crashed on ibm08
   when E25 produced residual overlap (no E41 fallback).
3. **Higher-T SA finds nothing new.** Phase 4 SA-v2 with T₀=1e-3
   accepted on only 1/16 benches (ibm02). The cascade+portfolio basin
   is well-saturated by SA's standards.
4. **Seed diversity is illusory.** Dualseed portfolio_saddle picked
   seed=42 vs seed=43 yielded identical results on 50% of benches;
   on the rest, the better seed won by <0.20%.

## What to verify before submission

1. **Run baseline stacked_periphery on AWS C6a EPYC or similar** to get
   the true expected score on judges' hardware class. Use `--all`
   minimum. Allow ~7 hours.
2. **Smoke test the eval_docker harness once more** with the bundled
   v2 DREAMPlace install (should fall back to 2-lane cleanly).
3. **Switch placer.py to Option C** (1-line change, see switch instructions).
4. **Push to origin/main** so judges get the final version.

## finalsa_v2 result (2026-05-18 18:00)

Trimmed earlier phases by 0.16 B to guarantee Phase 4 SA ~500s budget.

| Hardware | finalsa_v2 --fast | baseline --fast | Δ |
|---|---:|---:|---:|
| M3 | 0.89782 | 0.89844 | −0.07% (tied) |
| aws-gpu | 0.89852 | 0.91108 | −1.38% (but vs aws-gpu baseline, not M3) |

All 4 M3 benches REJECTed Phase 4 SA. Budget reallocation **FALSIFIED**
— cascade+portfolio basin is already saturated against high-T SA.

**All 5 algorithm tweaks today tied or worse than baseline 1.0575.**

## M3 baseline noise floor (2026-05-18 19:00)

Re-ran identical stacked_periphery --fast on M3 (same hardware,
~1 day apart):

| Bench | Original (2026-05-17) | Re-run (2026-05-18 18:15) | Δ |
|---|---:|---:|---:|
| ibm01 | 0.85963 | **0.84691** | **−1.48%** |
| ibm04 | 0.97285 | 0.97285 | 0.00% (exact tie) |
| ibm09 | 0.82494 | 0.82328 | −0.20% |
| ibm13 | 0.93633 | 0.93765 | +0.14% |
| **AVG** | **0.89844** | **0.89517** | **−0.36%** |

**Critical insight:** M3 run-to-run variance is **~0.3-0.5% aggregate,
~1.5% per-bench worst case** — much higher than the ±0.1% I was using
to compare variants. **Today's 5 algorithm tweaks (4w, finalsa,
finalsa_v2, tabu, dualseed mechanism, 6w-fast) are all WELL WITHIN
the actual noise floor** — none can be statistically distinguished
from baseline.

**Implication for the 1.05750 verified number:** It has uncertainty
of about **±0.4% on M3**, so true M3 baseline ≈ 1.054-1.062. Plus
the ~1.4-2.4% hardware variance to EPYC, judges may see anywhere
from **1.07 to 1.09**.

## Files modified today

- 7 new placer variants in `submissions/` (6w, 6w_k2, 4w, dualseed,
  finalsa, finalsa_v2 — only research, not promoted)
- Memory: `portfolio_noise_floor.md`, `hardware_variance.md`
- Docs: `2026-05-18_afternoon_variants.md`, this file
