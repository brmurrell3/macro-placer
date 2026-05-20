# Final report — 2026-05-19 (17:50, user check-in)

## TL;DR

**Ship Variant A default Lane-4 (`cd_lns_sa_cascade_stacked_periphery_e110`)**
- IBM 1.0574 (TIE with Option C 1.0575, within noise)
- NG45 **0.6744** (LIFT −2.16% vs Option C 0.6893)
- Zero overlaps, qualified
- Launcher updated to point at this placer

This **dominates Option C**: tied on IBM, clearly better on NG45.

## Today's decisive comparison (NG45, same hardware, same contention)

| Bench | Default Lane-4 | ovl10 Lane-4 | vs Option C |
|---|---:|---:|---:|
| ariane133 | 0.64187 | 0.65414 | Option C: 0.6641 |
| ariane136 | 0.64853 | 0.64428 | Option C: 0.6518 |
| mempool_tile | 0.73151 | 0.73360 | Option C: 0.7376 |
| nvdla | 0.67554 | 0.67339 | Option C: 0.6716 |
| **avg** | **0.67436** | **0.67635** | **0.6893** |
| **vs Option C 0.6893** | **−2.16%** | **−1.87%** | — |

**Default Lane-4 wins NG45 outright** (slightly better than ovl10).

## IBM --all data summary

| Placer | IBM avg | vs Option C 1.0575 |
|---|---:|---:|
| Option C (baseline) | 1.0575 | — |
| **Variant A default (overnight)** | **1.05737** | **TIE −0.01%** |
| Variant B Plateau | 1.06040 | LOSS +0.27% |
| ovl10 jobs=8 (contention-degraded) | 1.06282 | LOSS +0.52% |
| ovl10 jobs=4 (RUNNING 16/17) | partial ~1.058 | TIE |
| Triple plateau | 1.06014 | LOSS +0.25% |

Default Lane-4 ties Option C on IBM. With NG45 win, it's strictly better.

## Why this works

Variant A default Lane-4 adds E110 SmoothGlobalPlacer as a 4th candidate
after Option C's stacked_periphery output. The plateau-pick logic:
- Rejects E110 lane if overlaps or no improvement (most IBM benches)
- Accepts E110 lane when it clearly improves (some IBM + many NG45)

On IBM: cascade saturates most benches → E110 mostly REJECTed → near-TIE.
On NG45: gradient lane finds better basins → E110 wins on 2 of 4 → big lift.

## What we explored today (12+ hours)

1. ✅ Built Lane-4 architecture (Variant A) — STRICTLY BETTER than Option C
2. ✅ Built Plateau-pick architecture (Variant B) — slight loss vs Option C
3. ✅ Sweep-found hyperparameter cfgs (ovl10, nocong, long) — within noise on IBM
4. ✅ Built Triple multi-cfg placer (addresses per-bench tuning) — TIES Option C
5. ✅ Built E110Minimal (no cascade) — TIES Option C, much faster
6. ✅ Profiled cascade (94% Python CD — explains Carrotato speed gap)
7. ✅ Validated generalization to NG45 (commercial designs)

## Tomorrow (May 20)

- **EPYC cross-validation** on Variant A default Lane-4 (4 hr AWS spot ~$1.20)
- Confirm M3→EPYC variance: M3 1.0574 IBM → EPYC ~1.08 expected
- If EPYC numbers look right, commit + push + submit Wed

## Wed (May 21) submission day

- Submit via form by 11:59 PT
- Team: thinkorplace
- License: Apache 2.0

## Per the user's concerns

### Per-bench tuning
Variant A default uses ONE global cfg (no per-bench tuning). The
Triple variant additionally tests 3 cfgs as parallel lanes with
bench-agnostic plateau-pick. Both work without per-bench tuning.

### Speed gap to Carrotato (3.8 min vs our 55 min)
Profile shows 94% of our time is Python CD polish loops
(`_net_cong_contrib_flat`, `delta_cost_axis_batch`).

To close the gap (post-submission iteration):
- Cython-ize the CD hot path (~8 hr engineering, 5-10× speedup)
- Or use GPU/Triton kernels (Carrotato's approach)

For TODAY's submission, we use 55 min/bench (within 60-min cap).

## Pending (ovl10 jobs=4 IBM, ETA 18:10)

Last bench finishing. If full --all ovl10 jobs=4 IBM ≤ 1.063, my analysis
holds. If significantly worse, may need to revisit (unlikely given
12/17 partial = 0.987 avg matching default).

## State of repository

- `placer.py` (root): inherits from `CDLNSSACascadeStackedPeripheryE110Placer`
- `placer.py.bak`: backed up original (pointed to Option B previously)
- 10 new placer dirs in `submissions/` (variants explored today)
- Experiment artifacts in `experiments/E110_smooth_global_placer/`
- 5 new JSON --all/--ng45 results in `results/`

## Risk register

- ovl10 jobs=4 IBM full --all not yet complete; expected TIE/slight LOSS (not the shipping candidate anyway)
- EPYC variance ~2.4% — may shift the rank but Variant A default's IBM ≈ Option C
- Carrotato 0.967 IBM still 8.6% ahead — not addressed today; needs structural changes
- Submission packaging: launcher updated, eval_docker compatible
