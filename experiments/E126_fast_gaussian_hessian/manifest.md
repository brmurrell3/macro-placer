---
id: E126
name: fast_gaussian_hessian
status: in_progress
parent: E125
created: 2026-05-20
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E126: 3-way composition - E115 FastProxy + E117 Gaussian density + E120 Hessian saddle

## Hypothesis

Three validated architectural improvements should compose. E125 already
showed the 2-way composition E117+E120 is the next step from V3+saddle.
E126 adds E115's FastDiffProxy backbone:

1. **E115 FastDiffProxy** — `index_select` for pin gather + single-pass
   per-net-trace congestion. 3-16x per-Adam-step speedup. Numerically
   identical WL and congestion to V3.
2. **E117 Gaussian density** — C^inf erf-integrated Gaussian replaces the
   piecewise-linear `_grid_density`. Validated -2.71% avg on dense benches.
3. **E120 Continuous Hessian saddle escape** — Hessian smallest-algebraic
   eigvec perturbation after Adam basin convergence, then resume Adam.

The E115 speedup is the new variable: at 1500s budget, the V4 backbone
buys 3-16x more Adam steps than E125's V3 backbone. That additional
descent compute enables:
- Longer Phase A (300 -> 500 steps): better initial basin
- Longer resume descents (150 -> 200 steps): better Adam re-convergence
  from saddle perturbations
- Deeper saddle cascade (2 -> 3 stages): more chances to escape
- AND/OR CUDA scaling: A10G achieves 16.7x ibm17 per-step speedup vs CPU

Conceptually: same composition as E125 but the per-step bottleneck is
removed, so we can spend the saved time on more saddles.

## Method

`code/ultimate_placer.py`:

1. `FastDiffProxyGaussianDensity` — subclass of E115's `FastDiffProxy`
   that overrides `cost()` to call E117's `GaussianGridDensity` for the
   density term. WL and congestion paths unchanged (still
   `index_select`-fast).

2. `UltimatePlacer` — subclass of `ContinuousHessianSaddlePlacer` (E120)
   that overrides `place()` to instantiate `FastDiffProxyGaussianDensity`
   instead of `DiffProxyV3`. All saddle escape mechanics (eigsh,
   perturbation, Adam resume) work unchanged because they only call
   `proxy.cost()`, which our new proxy implements identically.

3. Submission wrapper at `submissions/e126_ultimate/placer.py`:
   - 1500s budget, 720s CD polish reserve
   - phaseA=500 steps, resume=200 steps, max_saddle_stages=3
   - sigma_scale=1.0, sigma_floor_frac=0.5 (E117 validated defaults)
   - device=cpu (CUDA available via env or constructor override)

## Kill gate

- ibm17 + CD polish >= 1.16 (V3Min baseline 1.20, E117 alone 1.181,
  E120 alone 1.21, E125 composed target < 1.16). If composed >= 1.16,
  the 3-way composition is at best tied with E125 and the V4 speedup
  didn't unlock new basin quality.
- Target: ibm17 < 1.13 (compound win means each layer is doing useful
  work).

## Generalization check

If ibm17 passes, run `--fast` (4 benches). If --fast avg beats champion's
fast subset and stays zero-overlap, run `--all` (17 IBM).

Champion gate: `--all` avg < 1.05 (would beat champion 1.0575 by clear
margin). Stretch goal: `--all` avg < 0.95 (would beat Carrotato's
leaderboard 0.967).

## Outcome (filled when decided)

[TBD - testing.]

## Pointers
- Code: `code/ultimate_placer.py` (FastDiffProxyGaussianDensity + UltimatePlacer)
- Submission: `submissions/e126_ultimate/placer.py`
- Parents (must validate before E126):
  - `experiments/E115_triton_kernels` (FastDiffProxy + V4)
  - `experiments/E117_gaussian_density` (Gaussian density model)
  - `experiments/E120_continuous_saddle` (Hessian saddle escape)
- Cousin: `experiments/E125_gaussian_hessian` (2-way E117 + E120 composition)
