---
id: E61
name: ga_crossover
status: in_progress
parent: E25, E41
created: 2026-05-02
decided: null
champion_at_time: 1.0990 (E12; E48 hybrid 1.08151 strongest verified candidate; ADR-011 *Proposed*)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E61: ga_crossover (genetic algorithm crossover between E25/E41 basins)

## Hypothesis
E48 hybrid plateaus at the BETTER of {E25, E41} per-bench. The
fundamental ceiling is set by the basins those two pipelines
deterministically converge to. Multi-seed within DPO (E53) was
sub-noise. Replica exchange SA-v2 (E60) targets SA convergence —
wrong attack point because our SA isn't actually stuck.

The RIGHT attack point is **basin choice**: E25 ends in SDF basin,
E41 ends in DPO basin, and there are presumably OTHER deeper basins
that neither pipeline reaches.

**Crossover (genetic algorithm style)** is the standard cross-field
technique: take two converged solutions, recombine subsets of their
variables, polish the recombination → new local minimum, often
better than either parent. **Crystal structure prediction (USPEX)
wins by exactly this mechanism.**

For our problem: E25 and E41 disagree on macro positions on every
bench (different basins). Crossover would take per-macro positions
from a random partition: subset A from E25, subset B from E41.
Project overlaps + run CD+LNS+SA-v2 polish from there → potentially a
hybrid basin neither pipeline finds alone.

## Method
Per-benchmark pipeline:

1. Run E25 (CDLNSSAPlacer) → e25_placement.
2. Run E41 (CDLNSSADPOKJointPlacer) → e41_placement.
3. **Crossover**: for each hard macro, randomly pick its position from
   either e25_placement or e41_placement (Bernoulli p=0.5 with global
   seed). Soft macros stay at their SDF/E25 positions.
4. **Repair**: project_overlaps to clear any overlaps introduced by
   the random recombination.
5. **Polish**: rebuild evaluator on the crossed-over placement; run
   CD plateau (≤1200 s) + grid-bin LNS (≤300 s) + SA-v2 (≤300 s) to
   refine into the nearest local minimum. Reduced budgets vs E25/E41
   primary phases since we're polishing, not from scratch.
6. Compare proxies of {e25_placement, e41_placement, polished_crossover}.
7. Return the lowest-cost output (zero overlaps verified).

All hyperparameters global; no per-benchmark tuning. The crossover is
a deterministic random sampling (fixed seed), not bench-specific logic.

## Kill gate
- **Polished crossover never wins:** if avg --fast >= E48 fast 0.92024
  (no improvement from crossover beyond what E48 picks), kill —
  crossover doesn't find new basins productively.
- **Regression:** if avg --fast > E48 fast 0.92024 + 0.5 % → kill,
  something broke in the polish phase.

## Generalization check
If --fast lifts ≥ 0.3 % over E48 fast 0.92024, run --ng45.
If --ng45 doesn't regress, queue --all.

Wall: per-bench wall = E25 + E41 + polish ≈ 80 + 30 = ~110 min/bench.
Total --all wall ≈ 30 hr serial, ~10 hr `--jobs 4`. Tight but within
17-hr envelope.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_ga_crossover.py`.
- Parents: E25 (SDF basin pipeline), E41 (DPO basin pipeline).
- Related: E48 hybrid (just picks per-bench best of E25/E41).
- Discussion: standard GA crossover from crystal-structure prediction
  / USPEX literature. Direct attack on basin choice.
