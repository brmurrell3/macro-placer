---
id: E152
name: ensemble
status: falsified
parent: E138
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.98387
outcome: ibm17 smoke 1.18768 (WORSE than v2-extCD 1.18269 by +0.42%, WORSE than E138 1.17996 by +0.66%)
champion_delta: positive (regression)
graduated_to: null
superseded_by: null
---

# E152: ensemble

## Hypothesis
Offline analysis of v2-extCD (--all avg 0.98387) and E138 (--all avg 0.98379)
EPYC runs shows the per-bench MIN avg is 0.98064 (-0.33% lift). 7 benches won
by E138, 10 by v2-extCD. They find STRUCTURALLY DIFFERENT basins. Running both
lanes in-process and selecting the canonical-best at runtime captures the
ensemble lift WITHOUT per-bench tuning.

## Method
In-process two-lane ensemble in `code/placer.py`:
- Lane A = v2-extCD pipeline: V4+Gauss descent + legalize + CD polish 700s,
  rng_seed=42.
- Lane B = E138 pipeline: V4+Gauss descent + legalize + CD polish 400s
  + bounded cascade saddle escape 240s + CD polish 200s, rng_seed=142.
- Pick lane with lower canonical proxy.
- Total budget ~30 min/bench, fits within 60-min/bench partcl cap.

Reuses (DOES NOT modify):
- `experiments/E127_v4_gaussian/code/smooth_global_placer_v4_gaussian.py`
- `experiments/E138_bounded_saddle/code/bounded_saddle.py`
- `macro_place/cd_core.run_cd_adaptive`

## Kill gate
EPYC ibm17 smoke. Compare to v2-extCD ibm17 1.18269 and E138 ibm17 1.17996.
- Kill: final proxy >= 1.180 (no improvement over both lanes).
- Marginal: 1.176 <= proxy < 1.180.
- Promote to --all: proxy < 1.176 AND total wall < 35 min.

## Generalization check
If smoke passes, run `--all --json --hypothesis E152_full` on EPYC.
Promote only if --all avg < v2-extCD 0.98387 (offline projection: 0.98064).

## Outcome
**FALSIFIED on EPYC ibm17 smoke (2026-05-21).** Kill gate triggered.

Smoke result:
- Lane A: V4+Gauss basin 1.28235 (225s) -> CD polish 570s (budget squeezed
  from 700s by 45% deadline split) -> proxy 1.18768, wall 892s.
- Lane B: V4+Gauss basin 1.27426 (228s) -> CD1 400s -> 1.19484 (799s
  cumulative wall) -> saddle budget collapsed to 30s by deadline math
  (only 4s actually used, 0 iters, no saddle escape) -> CD2 budget
  collapsed to 30s -> proxy 1.19552 (slight regression from CD1 1.19484).
- PICK=A, final=1.18768, total_wall=2045s.

Comparison:
- v2-extCD standalone ibm17 1.18269 -> E152 +0.42% (WORSE).
- E138 standalone ibm17 1.17996 -> E152 +0.65% (WORSE).

Root cause: budget split between two sequential lanes shortens each lane's
CD polish below the plateau-saturation point, so neither lane reaches its
standalone basin quality. The in-process ensemble at a 30-min/bench cap
cannot match standalone v2-extCD (45 min/bench) because CD polish on hard
benches (ibm17) was already budget-limited, not plateau-saturated.

Reasoning the offline ensemble worked (0.98064) but in-process did not:
the offline numbers were taken from FULL 45 min/bench runs of each lane.
Compressing both into 30 min total destroys the basin quality both lanes
were supposed to find.

Implication: ensemble is only valuable if each lane runs at full standalone
budget. That requires DOUBLED wall (60 min/bench) or PARALLEL execution
(2 GPU lanes simultaneously) — neither feasible at the current cap and
deadline.

Decision: do NOT run --all. Ship v2-extCD as Tier-1 submission. Surface to
human; E152 falsified.

## Pointers
- Code: `code/placer.py`
- Parents: v2-extCD (`submissions/thinkorplace-v2/placer.py`),
  E138 (`experiments/E138_bounded_saddle/code/placer.py`).
