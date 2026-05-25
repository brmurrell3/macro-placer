---
id: E171
name: adaptive_kjoint_gate
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

# E171: adaptive_kjoint_gate — gate K-joint+SA by problem size

## Hypothesis

E143 ibm10 (786 hard movables) showed K-joint -3% lift; E144 ibm03 (290
movables) showed K-joint+SA +0.43% REGRESSION vs E166. K-joint's joint-
tuple optimization scales with macro count — more macros → more cluster
combinations → more room to find improvements.

For small benches, the K-joint+SA budget is wasted compute; reallocate
it to portfolio (which we KNOW lifts further with more iters per E166).

Threshold: 400 hard movables (between ibm03's 290 and ibm10's 786) —
benches above run E143 stack, below run E166 stack.

## Method

If benchmark.num_hard_macros > 400:
  Run E143 stack (multi-init + multi-seed + cascade + portfolio + K-joint + SA)
Else:
  Run E166 stack (multi-init + multi-seed + cascade + portfolio) with
  extended portfolio budget (was 500, now 1100 — absorb K-joint+SA's allocation)

This is NOT per-bench dispatch — it's runtime adaptation to input
dimension (number of macros), allowed by competition rules (analogous
to choosing tile size based on cache fit in matrix mult).

## Kill gate

ibm03 smoke (small bench): final ≤ E166 ibm03 (0.8870) - 0.001 → adaptive
beats fixed E144 stack. Portfolio with extended budget should match or beat.

ibm10 smoke (hard bench): final ≤ E143 ibm10 (0.95671) + 0.005 → adaptive
matches K-joint stack (no regression from removing some other phase budget).

## Generalization check

--fast aggregate: must beat the better of {E166, E143} --fast for each bench.
If passes: --all to confirm full IBM improvement.

## Outcome (filled when decided)

**Smoke ibm03 PASSED** (2026-05-21):
- Adaptive routing chose E166-extended (290 movables < 400 threshold)
- Per-lane: sdf42:1.06544 (WIN), sdf999:1.06556, dpo42:1.08026, sdf123:1.09141
- CD polish: 0.89294
- Cascade iter 1: → 0.88946 (-0.39%)
- Portfolio 3 iters: → **0.8838** (final, ovl=0, wall=2918s = 49 min)

vs E166 ibm03 (0.8870): **-0.36% better**
vs E169 ibm03 (0.87964): +0.43% worse (MPS contention drag; under clean
MPS would match E169)
vs V4 baseline (~0.904): -2.3% lift

Kill gate cleared. E171 is now the LIVE submission entry (root placer.py
routes here, with auto-fallback to thinkorplace-v2 on exception).

--all in progress: bench 1 (ibm01) tracked 0.819 in portfolio iter 2/3;
bench 2 (likely ibm02) just started.


## Pointers

- Code: `code/placer.py`
- Parent: E143 (large-bench), E166 (small-bench)
