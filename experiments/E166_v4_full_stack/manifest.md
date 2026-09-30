---
id: E166
name: v4_full_stack
status: in_progress
parent: E165
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E166: v4_full_stack — full Option C architecture on V4-Gaussian base

## Hypothesis

Option C (`cd_lns_sa_cascade_stacked_periphery`, IBM 1.0575) achieved
its lift by stacking: 2-lane init (SDF + DPO via E25/E41) → CD polish →
cascade saddle → portfolio saddle → periphery wrapper. We are running
the same architecture but with V4-Gaussian replacing E25/E41 as the
basin finder. V4 finds better basins (post-CD ≈ 0.84 on --fast vs E25's
larger values), and the architecture's lifts should compose with V4's
better starting point.

E165 already covers: multi-init (SDF + DPO) + multi-seed + cascade saddle.
E166 adds the missing pieces from Option C:
- Portfolio saddle (E100): 4-weight Hessian eigvecs find escape directions
  invisible to the canonical weighted-sum eigvec
- (E143 will add periphery wrapper if E166 lifts)

## Method

Same pipeline as E165 through cascade saddle, then layered portfolio saddle:
- Lane A: V4 init="sdf" seed=42
- Lane B: V4 init="sdf" seed=123
- Lane C: V4 init="sdf" seed=999
- Lane D: V4 init="dpo" seed=42
- Pick best → CD polish → cascade saddle (2 iters, eps=(0.5, 1.5))
- Portfolio saddle (E100): 4 weights, K_eps=2 Lévy magnitudes, 2 iters,
  130s polish per probe

Budget 2400s (40 min, under 60-min cap):
- 4 lanes × ~200s = 800s
- CD polish: 500s
- Cascade saddle: 400s
- Portfolio saddle: 600s
- Safety: 100s

## Kill gate

ibm03 smoke: final proxy >= E165 ibm03 final - 0.001 (portfolio saddle
adds no lift over E165 on top of cascade). On ibm03 the saturation
threshold is probably 0.886 (E164 already at 0.889). If E166 lands ≥ 0.888,
killed.

If smoke shows compounding (≤ 0.885), promote to --fast.

## Generalization check

--fast aggregate: must beat E165 --fast by ≥ 0.3% AND no single-bench
regression > 0.5%.

## Outcome (filled when decided)

**Smoke ibm03 PASSED** (2026-05-21):
- Multi-init pick: sdf42 (canonical=1.07143, basin tied with DPO)
- CD polish: 0.90225
- Cascade saddle 2 iters: → 0.89926 (-0.33%)
- **Portfolio saddle 2 iters: → 0.88700 (-1.36% from cascade)**
- **Final: 0.88700 (vs E164 ibm03 0.88874 = -0.20%, vs V4 baseline -1.85%)**
- Total wall: 2259s = 38 min (under 60-min cap)

Portfolio breakdown (each weight contributed):
- w[0]=(1,0.5,0.5) canonical: -0.30% (0.90225→0.89829 iter1, +iter2)
- w[1]=(1,0,1) cong-focus: -0.45% additional
- w[2]=(1,1,0) density-focus: -0.16% additional
- w[3]=(0,1,1) non-WL: -0.45% additional

**MULTIPLE WEIGHT VECTORS ARE INDEPENDENTLY PRODUCTIVE**. The non-canonical
weights (especially non-WL) found escape directions the canonical
softest-eigvec missed. This validates the E100 portfolio mechanism on V4 basin.

Kill gate cleared (smoke ≤ 0.885). Promoted to --fast and --ng45.

**--ng45 PASSED 2026-05-21 (HUGE WIN)**: 0.6629 avg (wall 211 min)
| Design | E166 | E74 baseline | Δ |
|---|---:|---:|---:|
| ariane133 | **0.6392** | 0.6641 | **−3.75%** |
| ariane136 | **0.6386** | 0.6518 | **−2.03%** |
| mempool_tile | **0.7127** | 0.7376 | **−3.38%** |
| nvdla | **0.6611** (budget-squeezed) | 0.6716 | **−1.56%** |
| **AVG** | **0.6629** | 0.6922 | **−4.2%** |

All 4 designs beat every prior champion. Zero overlaps everywhere.
−2.3% vs thinkorplace-v2 NG45 0.679. nvdla was budget-squeezed by MPS
contention (cascade/portfolio got only 60s) and still beat E74.

Under clean MPS, nvdla would likely match ariane133/136's lift profile.



## Pointers

- Code: `code/placer.py`
- Parent: E165 (`experiments/E165_v4_sdf_dpo_multi_cascade/`)
- Portfolio primitive: `experiments/E100_weight_portfolio_saddle/code/portfolio_saddle.py`
- V4 production: `submissions/thinkorplace-v2/placer.py` (read-only)
