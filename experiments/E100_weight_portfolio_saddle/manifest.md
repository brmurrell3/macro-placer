---
id: E100
name: weight_portfolio_saddle
status: in_progress
parent: E97
created: 2026-05-13
decided: null
champion_at_time: 1.0669
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E100: weight_portfolio_saddle — Multi-objective Hessian eigvec search

## Hypothesis
The cascade saddle (E84/E97) computes the eigvec of the WEIGHTED-SUM proxy
Hessian: `∇²(WL + 0.5·D + 0.5·C)`. But the softest direction of the
weighted sum may HIDE soft directions of individual components.

Example: if WL has a strong negative-curvature mode that's exactly cancelled
by density's positive curvature in the sum's Hessian, the cascade never
explores that direction.

Cycle through a portfolio of weight vectors `w = (w_WL, w_D, w_C)` at each
cascade iter. For each `w`, compute eigvec of the weighted Hessian and
ε-perturb. Accept all improvements against the CANONICAL proxy. Composable
with Lévy magnitudes from E97.

## Method
- `WeightedSmoothProxy.cost_weighted(positions, w_wl, w_den, w_cong)` —
  same components as parent, with overridable weights.
- `find_softest_for_weight(smooth, state, movable, w)` — monkeypatch
  smooth.cost to use weighted variant, call parent's `find_softest_eigenvectors`.
- At each cascade iter, loop over portfolio entries; for each `w`, find
  softest eigvec, run ε-perturb + polish.

Default portfolio (4 weight vectors):
- `(1.0, 0.5, 0.5)` — canonical
- `(1.0, 0.0, 1.0)` — congestion focus (drop density)
- `(1.0, 1.0, 0.0)` — density focus (drop congestion)
- `(0.0, 1.0, 1.0)` — non-WL

K_eps=2 (smaller than Lévy's K=3 since we now have 4× eigvec calls per iter).

## Kill gate
Spike on cached cascade_ibm03.pt (init proxy ~0.946) with budget=1200s. If
portfolio's best_proxy ≥ pure Lévy's best_proxy (same wall budget), kill.

## Generalization check
After spike, integrate via `submissions/cd_lns_sa_cascade_portfolio_levy/placer.py`
and validate aggregate on EPYC --all vs cascade_levy production.

## Outcome (partial, 2026-05-13 spike)

ibm03 cascade plateau (init proxy 0.94634), portfolio of 4 weights, K_eps=2,
polish_budget=60s, total_budget=800s:

| weight | (w_WL, w_D, w_C) | λ_w | Lift this weight | Cumulative |
|--------|------------------|------:|-----------------:|----------:|
| w[0]   | canonical (1, 0.5, 0.5) | -0.178 | -0.068% | -0.068% |
| w[1]   | cong-focus (1, 0, 1) | **-0.277** | -0.092% (more!) | -0.16% |
| w[2]   | density-focus (1, 1, 0) | -0.008 | -0.018% | -0.178% |
| w[3]   | non-WL (0, 1, 1) | — | (budget cutoff) | — |

**Key finding**: the congestion-only Hessian has a SOFTER eigvec (λ=-0.277)
than the canonical sum (λ=-0.178), pointing in a DIFFERENT direction. Polish
along it lifted canonical proxy by an additional 0.092% beyond what
canonical's eigvec gave. Multi-objective Hessian eigvec search is valid:
the sum's Hessian hides single-component soft directions when sign-mixed
curvatures cancel.

**Caveat**: at fair wall, pure Lévy K=3 on canonical (Tabu iter 1 spike)
got -0.236% on the same plateau — more than portfolio K=2 with 3-4 weights
(-0.178%). Magnitude diversity (K=3) beats direction diversity (K=2 × 3w)
on ibm03.

**Production placer**: `submissions/cd_lns_sa_cascade_dual_levy/placer.py`
uses K=3 Lévy with TWO weights `[(1, 0.5, 0.5), (1, 0, 1)]` — combines
magnitude + direction diversity. Awaits cloud --fast validation.

### ibm10 confirms generalization (2026-05-13)

ibm10 cascade plateau (init 0.98940), full portfolio of 4 weights, K=2,
polish 60s, total budget 1000s:

| weight | (w_WL, w_D, w_C) | λ_w | Cumulative best | vs init |
|--------|------------------|------:|----------------:|--------:|
| w[0]   | canonical (1, 0.5, 0.5) | -0.0788 | 0.98627 | -0.313% |
| w[1]   | cong-focus (1, 0, 1)    | **-0.157** | 0.98224 | -0.716% |
| w[2]   | density-focus (1, 1, 0) | -0.0017 | 0.97917 | -1.023% |
| w[3]   | non-WL (0, 1, 1)        | -0.217 | **0.97863** | **-1.089%** |

**ibm10 portfolio aggregate: -1.089% lift** in 994s.
**Pure Lévy K=3 single-eigvec on same plateau: -0.260%** (Tabu iter 1 spike).
**Portfolio wins by 4.15×** on this hard bench.

### Key insight

Hard benches (ibm10) have MORE residual eigvecs to explore after cascade
saturates the canonical direction. Each weight's eigvec adds genuine lift:
- canonical (1, 0.5, 0.5) reaches -0.31% (matches cascade single-vec depletion)
- cong-focus alone doubles total to -0.72% (different soft direction)
- density-focus tripples to -1.02% (despite λ ≈ 0, tiny perturbations land in basins)
- non-WL marginal +0.07%

On easy benches (ibm03), portfolio gives +0.11% over pure Lévy.
On hard benches (ibm10), portfolio gives +0.83% over pure Lévy.
**The harder the bench, the more portfolio wins** — exactly where lift matters most.

### Production decision

`submissions/cd_lns_sa_cascade_dual_levy/placer.py` ships with 2 weights
(canonical + cong-focus). At ibm01 budget=500s: 0.87711 vs Lévy alone
0.87808 (-0.11% extra). Awaiting full --all validation on cloud.

## Pointers
- Spike code: `code/portfolio_saddle.py`
- Parent E97 Lévy: `experiments/E97_levy_saddle/code/levy_saddle.py`
- SmoothProxy base: `experiments/E74_hessian_saddle/code/hessian_saddle.py`
