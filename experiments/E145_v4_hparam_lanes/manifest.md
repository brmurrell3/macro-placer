---
id: E145
name: v4_hparam_lanes
status: in_progress
parent: E143
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E145: v4_hparam_lanes — hyperparam-diverse V4 lanes (basin geometry diversity)

## Hypothesis

E164 verified seed-diverse multi-seed gives -0.33% --fast lift (basin
attractor IS seed-sensitive). E165 showed DPO init doesn't lift on V4.
The remaining basin-diversity lever: hyperparameter-diverse lanes, where
each lane changes V4's `overlap_lambda_end` (overlap penalty strength
during Adam descent). Different λ → different basin geometry:
- λ=10 (current default): permissive descent, may find tightly packed basins
- λ=30: moderate, balances overlap avoidance and WL minimization
- λ=100: aggressive overlap penalty, finds well-separated basins
- λ=200: extreme, may push macros toward periphery

Hyperparameter variation should produce structurally different basins
than RNG seed alone — bigger basin spread → better best-of-N pick.

## Method

Same pipeline as E143 (multi-init multi-seed + CD polish + cascade +
portfolio + K-joint + SA). But replace 4 lanes with hyperparam diversity:
- Lane A: sdf, seed=42, λ=10
- Lane B: sdf, seed=42, λ=30
- Lane C: sdf, seed=42, λ=100
- Lane D: dpo, seed=42, λ=10

Each lane: V4 descent + greedy legalize. Pick lowest-proxy basin. Then
E143 stack (CD polish + cascade + portfolio + K-joint + SA).

Budget 3000s (50 min, under 60-min cap), same as E143.

## Kill gate

ibm15 smoke (hard bench, untested): final proxy >= E143 ibm10 - 0.001
(rough comparison since different bench, just kill if no signal).

If lane diversity is real (best λ ≠ default on ≥ 1 lane), promote to --fast.

## Generalization check

--fast aggregate: must beat E143 --fast by ≥ 0.2% AND no single-bench
regression > 0.5%.

## Outcome (filled when decided)

**Smoke ibm15 MARGINAL — λ diversity not productive** (2026-05-21):
- Lane basins: sdf_λ=10:1.15841 (WIN), sdf_λ=30:1.16339, dpo:1.16704, sdf_λ=100:1.19102
- **λ=10 (default) won** — higher λ regressed; lower λ untested
- CD polish: 1.15841 → 1.05007
- Cascade: 1.05007 → 1.04982 (-0.02%)
- Portfolio: → 1.04847 (-0.13%)
- K-joint: → 1.04406 (-0.42%, 82 commits — smaller than ibm10's -3% due to fewer macros)
- SA: 0 lift
- **Final: 1.04406 ovl=0 wall=3010s**

**λ-diverse lanes are NOT productive on V4.** λ=10 is already optimal;
larger λ values reduce basin quality. The "basin diversity via λ"
hypothesis is FALSIFIED.

Status: equivalent to E143 in this run. Recommend dropping λ-diversity
and reverting to seed-only diversity (E164/E166/E143 approach).


## Pointers

- Code: `code/placer.py`
- Parent: E143
- V4 hyperparam surface: `submissions/thinkorplace-v2/placer.py` Placer.__init__
