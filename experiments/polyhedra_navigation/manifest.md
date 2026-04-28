---
id: polyhedra_navigation
name: polyhedra_navigation
status: superseded
parent: null
created: 2026-04-15
decided: 2026-04-23
champion_at_time: 1.4578
outcome: 1.4867
champion_delta: -0.0289
graduated_to: null
superseded_by: dpo_best_of_v2
---

# polyhedra_navigation (Phases 1–5 umbrella)

## Hypothesis
Macro placements live on a polyhedral arrangement: each cell of the
arrangement is a feasible topology (consistent set of L/R/A/B
relationships) and within a cell, optimal positions can be found by an
LP. *Navigating between cells* by flipping one binding constraint at a
time should let us reach the global optimum without ever solving the
combinatorial topology problem directly.

## Method
SDF init → polyhedra navigation: at each step, identify the binding
inequality that improves objective most when flipped, recompute the
resulting LP, accept if proxy improves. Multiple phases tested:
- Phase 1 — pure LP.
- Phase 2 — polyhedra navigation, 300 s budget per benchmark (best variant).
- Phase 3 — polyhedra navigation, 50 s budget.
- Phase 4 — coarse SA + LP-fine refinement.
- Phase 5 — LP cap and overlap variants.
- Plus optimal-transport reassignments and cluster-screener pipelines.

## Kill gate
Must beat RePlAce baseline (1.4578) on `--all` and show a clear
trajectory toward the leaderboard.

## Generalization check
The 22-experiment overnight sweep across SP1 / SP3 / SP4 (topology /
surrogate / LP modifications) was the cross-axis stress test. All 22
landed within ±0.5 % of the polyhedra Phase-2 baseline.

## Outcome (filled when decided)
**Superseded by DPO best-of-v2.** Best variant: Phase 2 (300 s nav) at
**1.4867 avg `--all`** (−0.6 % vs RePlAce). Hit a structural ceiling:

- 22-experiment overnight sweep best 1.4918 vs baseline 1.4921 — within
  ±0.5 %, plateau confirmed.
- Per-benchmark gap predictor: `num_nets` (ρ = 0.958). Surrogate global
  Spearman ρ = 0.899; **within-benchmark ρ = 0.17** (proxy is locally
  unpredictable).
- 4 benchmarks get **0** nav improvements (ibm03, ibm06, ibm10, ibm16).
- Six follow-up experiments (real-proxy pair flips, swap+LP, swap-only,
  group cascade, sequence pair, position blending) confirmed: **swap+LP
  CAN drop congestion 44 %, but density jumps +180 %**. Different (good)
  topologies exist but cannot be reached by local moves.

*Lesson: the barrier is structural. SDF's basin is locally optimal;
nearby topologies are worse. Different (good) topologies exist but are
not reachable by polyhedra navigation. This is the diagnosis that
triggered the DPO pivot.*

SDF init logic (the only piece worth keeping) is retained at
`macro_place/sdf_init.py` (formerly `macro_place/sdf_init.py`; another
agent is moving it).

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16`). The polyhedra
  era's Phase-1–5 placers, LP solvers, and 22 overnight experiments were
  archived.
- Discussion: `writeup/evidence.md` §2.4 (per-bench table), §4 (the
  congestion barrier diagnosis), §6 (hierarchical decomposition
  failures); `docs/experiment_index.md` "Earlier polyhedra-era
  experiments" rows.
- Successor: `experiments/dpo_best_of_v2/manifest.md`.
