---
id: E138
name: bounded_saddle
status: in_progress
parent: E128
created: 2026-05-21
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E138: bounded_saddle

## Hypothesis
E128 (V4+Gauss -> cascade saddle -> CD) regressed +0.6-1.5% on EPYC because
each Lanczos eigsh iteration took ~380 s vs the 240 s saddle budget, leaving
no headroom for CD polish. Bounding eigsh to `maxiter=50, tol=1e-2` (vs
default `maxiter=500, tol=1e-4`) should let Lanczos return in ~50-60 s with
a slightly noisier soft-mode direction — still good enough to escape the
CD plateau, and the saved time goes into a longer CD polish.

## Method
Two parts:
- `code/bounded_saddle.py`: copies E84 `cascading_saddle_escape` and the
  required helper from E74, with the inner `eigsh` call wrapped so
  `maxiter` and `tol` are caller-controlled.
- `code/placer.py`: `E138BoundedSaddlePlacer` mirrors E128 with
  `cd_polish_s=900, saddle_budget_s=120, saddle_max_iters=2,
  eigsh_maxiter=50, eigsh_tol=1e-2, budget_seconds=1800`.

No changes to E84 / E128 / E74 source.

## Kill gate
EPYC ibm17 smoke. Compare to v2-extCD ibm17 EPYC 1.18269.
- Fail (kill): final proxy >= 1.183 (no improvement over v2-extCD)
  OR any saddle-iter wall > 120 s (didn't fix the wall problem).
- Marginal (do not promote, archive): final 1.176 <= proxy < 1.183.
- Promote to --all: final < 1.176 AND all saddle walls <= 120 s.

## Generalization check
If smoke passes, run `--all --json --hypothesis E138_full` on EPYC.
Promote only if --all avg < Option B 1.06650 AND NG45 no regression
beyond 0.6890.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/placer.py`, `code/bounded_saddle.py`.
- Parent: `experiments/E128_cascade_on_v4gauss/code/placer.py`.
- Reference: `experiments/E84_cascading_saddle/code/cascading_saddle.py`,
  `experiments/E74_hessian_saddle/code/hessian_saddle.py` (eigsh call
  at line 200).
