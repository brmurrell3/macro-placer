---
id: E147
name: lp_refine
status: falsified
parent: E140
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 1.05750
outcome: ibm04 smoke 0.91927 (CD1 baseline 0.91958; LBFGS branch REJECTed for unrecoverable overlaps)
champion_delta: null
graduated_to: null
superseded_by: null
---

# E147: lp_refine

## Hypothesis
After CD plateau-converges, the local minimum it finds is constrained by
its axis-aligned greedy moves. A continuous-relaxation refinement
(L-BFGS-B on the smooth proxy with FROZEN overlap penalty) can move
positions off the integer-grid greedy minima toward a globally better
configuration. Projecting back to legal positions and re-polishing with
CD may yield a strictly better basin than CD alone.

## Method
Pipeline (parent: E140 V4+Gaussian → CD1 → bounded-saddle → CD2):

  1. V4+Gaussian descent + legalize (~150 s)
  2. CD polish 1 (~400 s)
  3. **L-BFGS-B (PyTorch) on smooth proxy with frozen overlap lambda**
     starting from CD1 plateau (~200 s)
  4. project_overlaps to enforce legality
  5. CD polish 2 (~300 s)

LBFGS chosen over scipy.linprog: the smooth proxy (FastDiffProxyGaussian)
is already a differentiable PyTorch function with GPU/MPS gradient flow.
Scipy round-trip would force CPU and lose the autograd graph. LP/QP
linearization of LSE-HPWL would lose the exact proxy correspondence.

Overlap lambda is frozen at the value that the descent ramp ended at
(default 10.0). Higher would prohibit movement; lower would allow lots
of overlap that the project step can't repair.

## Kill gate
If smoke ibm04 shows: (a) LBFGS phase increases proxy versus CD1 input,
or (b) final proxy >= E140 baseline (ibm04 ~0.85 on M3), this approach
is over-constrained or projection erases the gain → falsify.

## Generalization check
If smoke passes, run --all and compare to E140 1.05750-equivalent on M3.
Lift must be >0.5% (above M3 run-to-run noise floor) on combined avg.

## Outcome (filled when decided)

**ibm04 smoke (M3, 2026-05-21):** total 205s, final proxy 0.91927.
- Phase 1 (V4+Gauss descent): proxy 1.07857, wall 17s.
- Phase 2 (CD1): proxy 0.91958 (delta -0.159), wall 89s.
- Phase 3 (L-BFGS): 70 closures in 3s. Smooth proxy 1.05844 -> 0.99945
  (-5.9%); soft overlap area stayed 0. But canonical
  `compute_overlap_metrics` flagged 9 overlaps on the L-BFGS output;
  `project_overlaps` reduced to 6 and stalled. REJECTed -> reverted to
  CD1 baseline.
- Phase 4 (CD2 on CD1 baseline): proxy 0.91927 (delta -0.00031 vs CD1,
  within M3 0.3-0.5% noise), wall 82s.

**Diagnosis:** L-BFGS with strong-Wolfe line search on the soft-overlap
smooth proxy DOES decrease the smooth objective (-5.9%), but it pushes
macros into near-touching positions whose discrete-coord overlap count
is non-zero (the smooth overlap penalty quadratically vanishes as
slack approaches zero — once macros are kissing, the penalty contributes
no gradient pressure to separate further). `overlap_lambda=10.0` (the
descent end value) is too low to prevent this in the refinement phase.
`project_overlaps` (greedy shove-to-legal) cannot repair the 6 stubborn
overlap pairs.

**Falsified per kill gate (a):** L-BFGS branch is a no-op via REJECT.
Final equals CD1 baseline within noise (-0.034% < 0.5% noise floor).

Two mechanism options for follow-up (NOT pursued in E147):
1. Raise `lbfgs_overlap_lambda` to 100-1000 so kissing configurations
   become penalised before LBFGS converges.
2. Replace `project_overlaps` with a stronger legalizer (Tetris /
   greedy_macro_legalize) after the soft-overlap LBFGS step.

## Pointers
- Code: `code/placer.py`.
- Smoke log: `/tmp/e147_smoke.log`.
- Result JSON: `results/E147LpRefinePlacer_20260521_130453.json`.
- Log row: `results/experiment_log.jsonl` (`E147_smoke`).
