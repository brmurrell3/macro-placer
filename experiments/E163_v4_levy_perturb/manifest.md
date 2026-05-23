---
id: E163
name: v4_levy_perturb
status: in_progress
parent: E136
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E163: v4_levy_perturb — V4-Gaussian basin + Lévy ε saddle escape

## Hypothesis

E97 proved Lévy (half-Cauchy) ε magnitudes beat fixed Gaussian by 0.13-
0.23% on 3/3 IBM benches when paired with cascade saddle escape on
SDF/DPO basins. E136 is testing fixed-ε cascade on V4 basin. E163 layers
Lévy on top: heavy-tail ε magnitudes (median=1.0, occasional 5-10× jumps)
along the V4 basin's softest Hessian eigvec.

Mechanism: V4-Gaussian's Adam descent + CD polish reaches a local min in
the smooth-proxy landscape. Hessian's softest eigvec identifies the
shallowest escape direction. Lévy ε samples test both nearby valleys
(small jumps) and distant basins (large jumps), with the heavy-tail
distribution giving magnitude diversity a fixed grid lacks.

Strictly orthogonal to E136 (fixed-ε): E136 tests if cascade-on-V4 works
at all; E163 tests if the optimal perturbation distribution differs from
fixed-grid Gaussian.

## Method

Same wrapper as E136 — V4 basin via thinkorplace-v2 Placer (read-only) —
but swap `cascading_saddle_escape` for `levy_saddle_escape` (E97).
Parameters: K_eps=4, eps_scale=1.0, max_iters=2, polish_budget=120s,
cascade_reserve_s=420s.

## Kill gate

ibm03 smoke: E163 final proxy ≥ E136 final proxy + 0.002 → kill (Lévy
not better than fixed-ε on V4 basin; falsifies hypothesis).

If both E136 and E163 fail to lift V4 basin → V4+CD already reaches a
true local min on the smooth proxy; no saddle escape mechanism helps;
the cascade family is exhausted for V4.

## Generalization check

If smoke passes: `--fast` aggregate ≤ V4 baseline minus 0.2% AND no
single-bench regression > 0.5%.

## Outcome (filled when decided)

**Smoke ibm03 FALSIFIED** (2026-05-21):
- V4 basin: proxy=0.90407 ovl=0 wall=474s
- λ_min = -0.382 (NEGATIVE — even softer mode than E136 found)
- Lévy iter 1 ε_grid = [0.824, 1.943, 2.697, 4.428] (K=4, σ=1.0, seed=42)
- No improvement on iter 1 — saturated → cascade exited
- **Final: 0.90407 (NO LIFT, +0.00%)**, wall=935s
- vs E136 same bench (fixed-ε): 0.89530 (-1.00% lift)

**Killed**: Lévy K=4 under-samples small ε on V4 basin. E136 fixed-ε
(0.5, 1.5, 3.0) found wins at ε=0.5 and ε=1.5 — Lévy's smallest sample
0.824 missed both sweet spots. The V4 basin's soft mode is best escaped
at small step magnitudes, and Lévy's heavy tail biases toward larger ε.
Opposite finding from E97 (Lévy helped on SDF basin); the V4 basin
geometry is structurally different.

Status: falsified vs E136 on ibm03. Do not promote.
Could re-test with K=8 (denser ε coverage) but expected to add wall for
minimal lift; not pursuing.


## Pointers

- Code: `code/placer.py`
- Lévy primitive: `experiments/E97_levy_saddle/code/levy_saddle.py`
- V4 production placer: `submissions/thinkorplace-v2/placer.py` (read-only)
- Sibling: E136 (`v4_cascade_hybrid` — same wrapper, fixed-ε)
