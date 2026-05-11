---
id: E5
name: batched_seeds
status: falsified
parent: null
created: 2026-04-26
decided: 2026-04-26
champion_at_time: 1.3834
outcome: 1.1698
champion_delta: -0.2136
graduated_to: null
superseded_by: null
---

# E5: batched_seeds

## Hypothesis
DPO has 0.45 % seed-to-seed variance. Running batched seeds (B = 64) on
the M3 Max GPU should let us best-of-N a much wider exploration of seed
space at near-constant wall and find a better basin than any single seed.

## Method
`submissions/dpo/batched_seeds_placer.py` (679 L) — B = 64 DPO seeds run
in parallel on MPS, with σ = 0.04 × canvas perturbation between inits.
Best-of-N selection per benchmark. Companion `batched_seeds_b1.py` is the
B = 1 control.

## Kill gate
Either (a) wall scales sub-linearly so B = 64 fits within budget AND
best-of-64 beats best-of-5 by > 0.5 %, OR (b) keep one of those true.
Failing both kills the hypothesis.

## Generalization check
`--fast` first; only escalate to `--all` if best-of-64 vs best-of-5 shows
real signal.

## Outcome (filled when decided)
**Falsified.**

- B = 64 wall = **8.9× B = 1** (RUDY congestion kernel scales 27× on MPS;
  not the ~1.5× we hoped for from GPU parallelism).
- All 64 seeds with σ = 0.04 × canvas perturbation **collapse to the same
  basin**.
- Best-of-N within a single basin doesn't help: best 1.1638 (`--fast`
  B = 1), 1.1698 (`--fast` B = 64).

*Lesson: the basin IS the limit. Seed perturbation alone is too small to
reach a different basin; the differentiable proxy attractor is too strong
for σ = 0.04 to escape. This is the proof point that drove the search for
basin-changing moves (CD, LNS).*

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16`). Originally at
  `submissions/dpo/batched_seeds_placer.py` and `batched_seeds_b1.py`.
- Discussion: `writeup/evidence.md` §9.1; `docs/experiment_index.md`
  (E5 row).
