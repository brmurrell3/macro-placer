---
id: E3_v2
name: lns_v2_local_window
status: falsified
parent: E3_v1
created: 2026-04-27
decided: 2026-04-27
champion_at_time: 1.1193
outcome: 1.3824
champion_delta: 0.2631
graduated_to: null
superseded_by: E12
---

# E3 v2: lns_v2_local_window

## Hypothesis
E3 v1 was wall-bound by the full-canvas search (2 244 candidates per
macro). Restricting reinsertion to a 5×5 local window around the macro's
current position should be ~90× faster, allowing many more LNS iterations
within the budget. If the local search radius is the bottleneck, more
iterations should compound into a real improvement.

## Method
Same destroy/reinsert structure as v1, but the reinsertion candidate set
is the 5×5 grid window centered on the destroyed macro's current cell
(`window_radius = 5`). 90× speedup vs v1 enables 15 LNS iterations in 600 s
on ibm17.

## Kill gate
Same as v1 — must improve baseline by ≥ 0.5 % on a hard benchmark.

## Generalization check
ibm17 single-bench, same as v1.

## Outcome (filled when decided)
**Falsified.** 90× faster than v1 (15 iters in 600 s) but final ibm17 =
**1.3824 vs CDOnly 1.3830** — flat. Cost-based destroy selector still
saturates after 1–2 accepts; the 5×5 window cannot move clusters of
macros far enough to find a different basin. Single-macro local LNS does
NOT escape CD's local minimum, regardless of search radius.

*Lesson: the move type matters more than the wall budget. v2 confirmed
v1's failure mode wasn't a wall-budget problem.*

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16`). Originally at
  `submissions/cd/lns.py` (`window_radius=5`),
  `submissions/cd/cd_lns_placer.py`; preserved at
  `writeup/archive/submissions/lns.py`,
  `writeup/archive/submissions/cd_lns_placer.py`.
- Discussion: `writeup/evidence.md` §9.4; `docs/experiment_index.md` (E3 v2 row).
- Parent: `experiments/E3_lns_v1/manifest.md`.
- Successor: `experiments/E12_grid_bin_lns/manifest.md`.
