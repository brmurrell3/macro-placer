---
id: E133
name: reverse_lambda
status: proposed
parent: E127
created: 2026-05-21
decided: null
champion_at_time: 0.987  # combined 21-bench (Option C stacked_periphery)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E133: reverse_lambda (V4Gauss with reverse-time overlap_lambda schedule)

## Hypothesis

With Gaussian density (E117/E127) the basin is smoother and less prone
to early local minima than the piecewise-linear `_grid_density`. We
hypothesise that a **reverse-time** overlap_lambda schedule
(`start=50`, `end=5`, ramp 70%) helps:

- HIGH lambda **early** forces clean macro separation while the
  Gaussian basin is mushy and macros can re-arrange freely.
- LOW lambda **late** lets WL + congestion dominate the final basin
  choice, since by then overlaps are mostly resolved.

The forward-time default (`start=0`, `end=10/50/100`) lets macros pile
up early and then has to push them apart late, which can lock in
WL-suboptimal configurations.

## Method

Pipeline mirrors thinkorplace-v2 / E127:

1. V4Gaussian descent with **`overlap_lambda_start=50, overlap_lambda_end=5`**
   (reverse-time), `overlap_ramp_pct=0.7`, 500 steps, init="sdf".
2. project_overlaps.
3. run_cd_adaptive (~600s).

Retry chain: same as v2 but with the new schedule on attempt 1.
Attempts 2/3 fall back to forward-time strong overlap (50/100) as a
safety net.

No edits to shipped V4 / V4Gaussian — both already accept
`overlap_lambda_start` kwarg via inheritance (verified). Placer
imported as `E133ReverseLambdaPlacer`.

## Kill gate

ibm04 smoke: final proxy worse than 0.93 (i.e. clearly worse than
thinkorplace-v2 baseline 0.929) → falsified.

## Generalization check

If ibm04 smoke ≤ 0.92 and --fast (ibm01/03/04/06) avg ≤ thinkorplace-v2's
~0.99, run --all on EPYC. Need combined 21-bench avg ≤ 0.987 to
graduate over Option C.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/placer.py`
- Imports: shipped `experiments/E127_v4_gaussian/code/smooth_global_placer_v4_gaussian.py`
  (no modifications — `overlap_lambda_start` kwarg already supported
  via `*args, **kwargs` → `SmoothGlobalPlacerV4.__init__`).
- Parent: E127 (V4 + Gaussian density baseline).
- Related: E132 (polish_relax — different escape mechanism).
