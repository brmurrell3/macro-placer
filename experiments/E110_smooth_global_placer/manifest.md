---
id: E110
name: smooth_global_placer
status: in_progress
parent: E95
created: 2026-05-18
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E110: smooth_global_placer

## Hypothesis

The combinatorial cascade is basin-saturated at 1.06 (uncapped). Carrotato
reaches 0.967 with Xplace+Triton at ~4 min/bench — gradient-based global
placement on a smooth objective finds a structurally better basin than our
combinatorial methods can reach.

We don't need Xplace itself (its objective is HPWL + eDensity, mismatched
to our WL + 0.5·D + 0.5·C). E88/E95 already gives us a differentiable
approximation of the *exact* challenge proxy. What's missing is an Adam
descent driver that uses it.

If we wrap E95 DiffProxy in an Adam loop with γ-annealing and overlap-
penalty rampup, then legalize with greedy_macro_legalize, the resulting
basin should be reachable by cascade polish to <1.05 on M3 (Cezar-class
on EPYC, ~1.06-1.08).

## Method

1. **Init**: SDF (default) or random-uniform within canvas.
2. **Adam descent** on `loss_with_penalty` from E88/E95:
   - γ anneals linearly from `5e-3·cw` → `5e-4·cw` (smooth WL converges
     to canonical bbox max)
   - overlap_lambda ramps from 0 → 50 over the first 70% of steps (let
     macros pack tightly early, then push apart)
   - boundary_lambda fixed at 50 (always push escaped macros back)
   - Fixed-macro gradients zeroed each step
3. **Greedy legalize** (E76 macro_legalizer.greedy_macro_legalize): zero
   hard-macro overlaps via spiral search.
4. **Fallback**: if legalize fails to produce zero overlaps, run
   `project_overlaps` from SDF.

Intended as **Lane 4** in `stacked_periphery` plateau-pick. Time-budget
gated (skip if <300s remaining).

## Kill gate

- **Tonight (May 18 → 3am)**: ibm01 single-bench wall >15 min OR raw
  smooth-basin proxy >1.40 → kill the gradient-lane direction; ship
  Option C.
- **Tomorrow noon (May 19)**: --fast best config ≥1.06 → kill; ship
  Option C.
- **Tomorrow evening**: --all best ≥1.06 M3 → kill.

## Generalization check

Must run zero-overlap on all 17 IBM + 4 NG45 (cascade fallback in
plateau-pick covers regressions). EPYC cross-validation before May 21
submission.

## Outcome

[Empty until decided.]

## Pointers

- Code: `code/smooth_global_placer.py`, `code/run_ibm01.py`
- Results: `results/`
- Discussion: `docs/handoffs/2026-05-18_evening_decision.md` (the pivot
  decision), `writeup/evidence.md` §TBD.
