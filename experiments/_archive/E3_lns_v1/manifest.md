---
id: E3_v1
name: lns_v1_full_canvas
status: falsified
parent: null
created: 2026-04-27
decided: 2026-04-27
champion_at_time: 1.1193
outcome: 1.3846
champion_delta: 0.2653
graduated_to: null
superseded_by: E12
---

# E3 v1: lns_v1_full_canvas

## Hypothesis
After CDOnly converges, a rip-up-and-reinsert LNS phase that searches the
*full canvas* (every grid cell as a candidate position for a destroyed
macro) should escape CD's local minimum. CD finds a per-axis fixed point;
full-canvas search reasons jointly across the 2D grid, so it should reach
positions CD cannot.

## Method
After CD, destroy `K` macros (cost-aware ranking), reinsert each by
evaluating the proxy at every grid cell on the canvas and taking the
minimum via the incremental evaluator. Iterate until wall budget exhausted.

## Kill gate
LNS phase must improve baseline by ≥ 0.5 % on at least one IBM benchmark
within a reasonable wall budget. If a single LNS iteration costs
~hundreds of seconds and yields flat results, kill.

## Generalization check
Validate on ibm17 (hardest IBM) before scaling.

## Outcome (filled when decided)
**Falsified.** Full-canvas 2 244 grid candidates × ~100 ms each = 200+ s
per macro reinsertion. Only 1 LNS iteration in 428 s. Final ibm17 = 1.3846
vs CDOnly 1.3830 — flat. The cost-based destroy selector saturates after
1–2 accepts, and the wall-cost-per-iteration is too high to cover the
search budget.

*Lesson: single-macro local LNS does NOT escape CD's local minimum at
this fidelity. Real LNS would need cluster-level joint reinsertion or a
randomized destroy strategy — see E12.*

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16` "Archive superseded
  submissions, planning docs, and historic runs"). Originally at
  `submissions/cd/lns.py`, `submissions/cd/cd_lns_placer.py`; preserved at
  `writeup/archive/submissions/lns.py`,
  `writeup/archive/submissions/cd_lns_placer.py`.
- Discussion: `writeup/evidence.md` §9.4; `docs/experiment_index.md` (E3 v1 row).
- Successor: `experiments/E12_grid_bin_lns/manifest.md` — different move
  type (grid-bin centers, not full-grid cells), different destroy strategy.
- Sibling: `experiments/E3_lns_v2/manifest.md` (5×5 local-window variant).
