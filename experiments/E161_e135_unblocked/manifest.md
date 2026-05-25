---
id: E161
name: e135_unblocked
status: in_progress
parent: E135
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E161: e135_unblocked — fix E135 Lane A legalize so two-lane actually runs

## Hypothesis

E135's SDF+DPO two-lane V4+Gaussian smoke (2026-05-21) was inconclusive
because Lane A (SDF init) failed legalize: project_overlaps' deterministic
push-apart loop hit a stable 1-overlap fixed point and the 50-iter cap was
reached, so Lane A was SKIPped. The two-lane diversity hypothesis was
never tested.

If we add a jitter-and-retry wrapper around legalize — break the
symmetric oscillation by adding tiny random position perturbation between
project_overlaps calls — Lane A will produce a valid (ovl=0) basin and
the two-lane pick becomes a real test of whether SDF vs DPO basins
diverge meaningfully under V4-Gaussian descent.

## Method

Clone E135's placer into `code/`. Replace `_legalize` with a multi-stage
fallback:

1. `greedy_macro_legalize` with default radius.
2. If ovl > 0: `project_overlaps` (existing 50-iter loop).
3. If ovl > 0: jitter overlapping macros by ±0.5% canvas + `project_overlaps`,
   up to 5 retries with different seeds.
4. If ovl > 0: `greedy_macro_legalize` with 2× larger search radius.
5. If still ovl > 0: accept the lane with overlaps; trust CD polish to
   resolve (the production placer also raises on final overlaps, so this
   is the same recovery contract).

No modification to `cd_core.project_overlaps` or `greedy_macro_legalize`.
Wrapper is fully local to E161.

## Kill gate

ibm04 smoke: both Lane A and Lane B must produce ovl=0 placements. If
Lane A still SKIPs after retry chain → fix is insufficient, kill.

If both lanes succeed and the final proxy is >= V4-baseline 0.929 ibm04
(i.e., two-lane offers no diversity benefit) → falsify the diversity
hypothesis, kill.

## Generalization check

`--fast` (4 benches): two-lane V4+Gauss vs single-lane thinkorplace-v2.
Win threshold: aggregate ≤ 0.98 with no single-bench regression > 0.5%.

## Outcome (filled when decided)

**Smoke ibm04 PASSED legalize, MARGINAL on lift** (2026-05-21):
- Lane A SDF: smooth=0.96185 canonical=**1.09524** ovl_pre=28 ovl_post=0 wall=22s
- Lane B DPO: smooth=0.95816 canonical=**1.08263** ovl_pre=12 ovl_post=0 wall=23s
- [PICK] DPO (better basin), CD polish 258s
- **Final: 0.9194** (ovl=0)
- vs E135 baseline 0.9258 (DPO-only): within M3 run-variance (0.3-0.5%)
- vs thinkorplace-v2 ibm04 ~0.929: -0.74%, within noise

**Jitter-retry legalize works** — Lane A no longer SKIPs. But on ibm04,
SDF basin is strictly worse than DPO; two-lane provides no diversity
benefit. Needs --fast to see if SDF wins on any of {ibm01, ibm03, ibm06, ibm17}.
If SDF never wins → kill two-lane hypothesis; legalize fix is still useful
infrastructure (could ship into other experiments).


## Pointers

- Code: `code/placer.py` (clone of E135 with patched `_legalize`)
- Parent: E135 (`experiments/E135_sdf_dpo_lane/manifest.md`)
- Diagnosis: E135 Outcome section — "project_overlaps bounded at 50 iters left 1 overlap"
