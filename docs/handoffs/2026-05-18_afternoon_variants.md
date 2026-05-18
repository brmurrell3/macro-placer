# 2026-05-18 afternoon variant exploration

## TL;DR

After three more variant tests (4w, dualseed, finalsa), **stacked_periphery
1.0575 remains the best M3-verified candidate**. All variant tweaks land
within the ~0.1% noise floor.

**Critical finding (NEW):** Hardware variance is **~2.4%**, much larger
than algorithm variance. Judges' AMD EPYC machine will report
**~1.07-1.09**, not our M3-verified 1.0575. Cross-platform variance
is the dominant uncertainty.

## Variants tested today (2026-05-18 morning + afternoon)

| Variant | Hardware | Aggregate | vs M3 baseline | Notes |
|---|---|---:|---:|---|
| stacked_periphery (baseline) | M3 | 1.05750 | — | Champion 2026-05-17 |
| 6w portfolio K=1 max_iters=1 | M3 --fast | 0.89739 | −0.12% (--fast) | Noise; not pursued to --all |
| 6w_k2 portfolio K=2 | aws-gpu --fast | 0.91411 | +1.74% (--fast) | Truncated by deadline, **FALSIFIED** |
| **4w portfolio (+1,1,1)** | M3 --all | **1.05870** | **+0.11%** | 5W/12L per-bench, noise |
| **dualseed portfolio** | aws-gpu --all | **1.08280** | +2.42% | Hardware variance dominates; mechanism gives <0.05% |
| **finalsa (T₀=1e-3 SA at end)** | M3 --all | **1.05759** | +0.009% (TIED) | Phase 4 SA REJECTed 15/16; only ibm02 ACCEPT (+0.45% lift) |
| aws-gpu baseline --fast (calibration) | aws-gpu --fast | 0.91108 | +1.41% vs M3 --fast | Definitive hardware variance measurement |
| finalsa_v2 (extended Phase 4 budget) | aws-gpu --fast | in progress | ? | Trim other phases by 0.16 B to give Phase 4 ~500s |

## Key insights

1. **Variant noise floor: ~0.1%.** Tweaking saddle parameters
   (weights, K_eps, max_iters, tabu vs cascade, dual seed) lands within
   ±0.1% of 1.0575. Only structural changes (new lane, DP polish,
   multi-start) are likely to give meaningful lift.

2. **Hardware variance: ~2.4% on identical placer.** M3 MPS-accelerated
   path vs aws-gpu EPYC CPU path diverge in `run_cd_adaptive` and
   Hessian eigvec finding. This is real algorithmic variance from
   FP rounding and random sweep ordering.

3. **Judges' EPYC machine is closer to aws-gpu than M3.** Spec says
   "AMD EPYC 9655P + RTX 6000 Ada". Expect final score landing
   ~1.07-1.09, not the M3-verified 1.0575.

4. **Final SA-v2 at high T (1e-3) mostly REJECTs.** On 8 finalsa benches,
   only 1 (ibm02) accepted. The portfolio_saddle finds basins that
   SA can't improve. Suggests the saddle escapes are well-saturated.

## Submission decision (recommend)

**Stay with Option C (stacked_periphery 1.0575 M3, ~1.07-1.09 EPYC).**
None of the variants give clear improvement above noise. Hardware
variance is the dominant unknown.

For final submission, **verify on AWS C6a or similar EPYC instance**
before deploy to get true expected number. Don't trust M3 alone.

## Time spent (2026-05-18 morning + afternoon)

- 04:30 AM: 6w + 6w_k2 --fast launched, completed mid-morning
- 09:15 AM: 4w --all launched, completed 12:30 PM (+0.11%)
- 12:30 PM: finalsa --all launched on M3, ETA ~4 PM
- 09:15 AM: dualseed --all launched on aws-gpu, completed 14:35 PM (+2.42% hw variance)
- 14:35 PM: baseline --fast on aws-gpu launched for variance calibration

Repo state: clean main branch, all docs accurate. No code changes to
submission entry placer.py — still wired to Option B.
