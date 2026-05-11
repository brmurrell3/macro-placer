---
id: E11
name: diverse_priors
status: falsified
parent: null
created: 2026-04-27
decided: 2026-04-27
champion_at_time: 1.3834
outcome: 1.3839
champion_delta: 0.0005
graduated_to: null
superseded_by: null
---

# E11: diverse_priors

## Hypothesis
DPO basin lock on ibm02 / ibm12 is structural — every DPO seed lands on
the same byte-identical placement. Maybe the basin is a property of the
*init*, not the optimizer. Running DPO from four different priors (SDF,
Will's pre-fork seed, greedy row, random) and taking best-of per
benchmark should reach different basins on the locked benchmarks.

## Method
4-prior best-of: each prior produces an init, DPO runs from each, best
proxy per benchmark wins. Code: `submissions/dpo/diverse_priors_placer.py`,
plus `init_strategies.py`, `e11_sdf_only.py`, `e11_sdf_random.py`.

## Kill gate
Avg `--all` must improve by ≥ 1 % vs DPO best-of-v2 (1.3834 → ≤ 1.3696)
AND ibm02/ibm12 specifically must improve.

## Generalization check
`--fast` improvement must replicate on `--all`.

## Outcome (filled when decided)
**Falsified.**

- 4-prior best-of `--fast`: 1.1704 (−0.4 % vs SDF-only). Will-prior wins
  ibm09 / ibm13 on `--fast` set.
- `--all`: **1.3839 (+0.04 %, FLAT)**.
- ibm02 and ibm12 got **WORSE** with alternative priors.

*Lesson: basin diversity exists but doesn't scale to hard benchmarks.
Basin-locked benchmarks are locked by topology, not by init. The
attractor in DPO's loss landscape is so strong that even a starkly
different init (random) is dragged toward the same basin. This was the
final negative result that motivated the move from "diversify within
DPO" to "change the optimizer entirely" (CD).*

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16`). Originally at
  `submissions/dpo/diverse_priors_placer.py`,
  `submissions/dpo/init_strategies.py`,
  `submissions/dpo/e11_sdf_only.py`,
  `submissions/dpo/e11_sdf_random.py`.
- Discussion: `writeup/evidence.md` §9.3; `docs/experiment_index.md`
  (E11 row).
