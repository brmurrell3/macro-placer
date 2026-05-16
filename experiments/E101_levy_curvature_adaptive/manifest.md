---
id: E101
name: levy_curvature_adaptive
status: in_progress
parent: E97
created: 2026-05-14
decided: null
champion_at_time: 1.0770
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E101: levy_curvature_adaptive — σ scales with eigval magnitude

## Hypothesis
E97 Lévy uses fixed eps_scale=1.0 across all benches. But the Hessian's
smallest eigenvalue λ_min varies per iter (and per bench). Classical
optimization theory: Newton step size ~ 1/√(curvature). Setting the
half-Cauchy median to σ = β/√|λ_min| auto-tunes per-iter jumps to
landscape softness — softer mode → larger step. No bench-name branches;
λ_min is observable per iter.

## Method
Modified `levy_saddle_escape` (E97) to set `eps_scale_adaptive = β/√|λ_min|`
each iter before sampling K Lévy magnitudes. Clamped to
`[eps_scale_floor=0.1, eps_scale_ceiling=10.0]` to keep ε reasonable when
λ_min is near zero or very large. Default β=3.0 (allows escape beyond
Newton optimum).

Same K_eps=3, same Cauchy distribution. Combined with polish_budget=60
(from multi-iter finding) so 3+ saddle iters fit per bench.

## Kill gate
On --all 17 IBM, if aggregate ≥ 1.0770 (no lift vs fixed σ baseline), kill.
If aggregate ≤ 1.072, surface as candidate.

## Pointers
- Core: `code/levy_curvature_adaptive.py`
- Placer: `submissions/cd_lns_sa_cascade_levy_curvadapt/placer.py`
- Parent: `experiments/E97_levy_saddle/`
- Cloud hypothesis: `E101_levy_curvadapt_all`
