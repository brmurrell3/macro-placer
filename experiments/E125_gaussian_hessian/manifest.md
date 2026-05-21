---
id: E125
name: gaussian_hessian
status: falsified
parent: E120
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.0575
outcome: ibm17 = 1.2247 (worse than E117 alone 1.181, worse than V3Min 1.200, fails kill gate <1.16)
champion_delta: null
graduated_to: null
superseded_by: null
---

# E125: Composed Gaussian density + Hessian saddle escape

## Hypothesis

Two architectural wins composed should compose multiplicatively (not just
additively). Both lift through landscape geometry rather than fidelity:

1. **E117 Gaussian density (-2.71% avg on dense benches):** replaces the
   piecewise-linear `_grid_density` with a C^inf erf-integrated Gaussian.
   Adam sees a smooth gradient everywhere instead of step jumps at cell
   edges.
2. **E120 Continuous Hessian saddle escape (-1 to -2.2%/bench):** at
   Adam basin convergence, compute the smallest-algebraic eigenvector of
   the smooth-proxy Hessian and perturb +/- epsilon along it. Resume
   Adam from each candidate; pick the best.

The composition hypothesis: a smoother density landscape (E117) gives
the Hessian saddle escape (E120) a richer set of *negative-curvature*
directions to exploit. On the C^0 grid-bin density, the Hessian has
spurious zero/discontinuous eigenstructure at cell boundaries. The C^inf
Gaussian removes those artifacts so eigsh's smallest-algebraic search
points to real saddles, not to numerical artifacts.

## Method

`code/composed_placer.py` subclasses `ContinuousHessianSaddlePlacer` from
E120 and overrides only `place()` to instantiate
`DiffProxyV3GaussianDensity` from E117 instead of the standard
`DiffProxyV3`. All other mechanics (Adam descent, Hessian eigvec,
+/-epsilon perturbation, Adam resume, legalize, CD polish) are unchanged
because they operate on `proxy.cost()`, which `DiffProxyV3GaussianDensity`
overrides while keeping the same interface.

Pipeline (mirrors E120, only the proxy class changes):
  Phase A: V3+Gaussian Adam descent, ~300 steps
  Phase B: Hessian saddle escape on Gaussian smooth proxy
  Phase C: 1-2 more saddle escapes if budget remains
  Phase D: greedy_macro_legalize + CD polish

Default budget: 1500s/bench (12 min CD polish + ~13 min descent+saddle).

## Kill gate

- ibm17 + CD polish >= 1.16 (V3Min baseline ~1.20, E117 alone 1.181, E120
  alone at 720s 1.21). If composed >= 1.16, the two mechanisms do not
  add useful capability beyond the better of the two alone.

## Generalization check

If ibm17 passes, run --fast (4 benches). If --fast avg beats champion's
fast subset, run --all. Champion gate: --all avg < 1.05.

## Outcome (decided 2026-05-20)

**FALSIFIED on ibm17 head-to-head test.**

| Method | ibm17 proxy | wall (s) | notes |
|---|---:|---:|---|
| V3Min ovl10 720s | 1.200 | ~720 | baseline |
| E117 Gaussian alone @720s | **1.181** | 847 | Gaussian density only |
| E120 saddle alone @720s | 1.212 | 720 | Hessian saddle only |
| **E125 composed @1500s** | **1.2247** | 1713 | this experiment |

Result is **+3.7% WORSE than E117 alone** and **+1.0% worse than E118
multistage baseline**. Composition does NOT compose: at minimum the saddle
escape steals budget from CD polish (1500s budget, but saddle stages ate
~250-500s of that), and at most the C^∞ smooth Gaussian landscape lacks
the negative-curvature modes that E120 saddle escape requires.

Kill gate was ibm17 + CD polish < 1.16. **Not met.** Composition is dead
on this objective.

### Why composition fails (hypotheses)

1. **Budget displacement**: At 1500s/bench, E117 alone would get ~600-720s
   CD polish at 1.181. E125 ate ~500-700s on saddle stages, leaving only
   ~500-800s for CD polish on a basin that's only marginally lower than
   E117's. Net effect: less CD polish on a similar basin → worse.

2. **Landscape geometry mismatch**: The Gaussian density landscape is
   C^inf smooth (no step jumps at cell boundaries). The Hessian therefore
   has fewer pronounced negative eigenvalues compared to the V3 grid-bin
   landscape (which has sharp eigendirections at cell boundary
   discontinuities). The saddle escape is finding directions, but they
   land in basins that don't polish meaningfully better than the
   Gaussian-only basin.

3. **Smoke test on ibm01 showed -2.28% basin lift via saddle (0.807 →
   0.789), but final proxy after CD polish was 0.840 — only 1-2% better
   than V3 baseline.** The smooth-basin lift does not transfer linearly
   through legalize+CD polish.

### Smoke ibm01 result (sanity, NOT falsifying)

| Phase | smooth proxy | Δ |
|---|---:|---:|
| init (SDF) | 1.174 | - |
| Phase A (200 Adam steps) | 0.808 | -31.1% (Gaussian descent works) |
| Saddle eigvec | λ_min = -0.0147 (negative ✓) | - |
| After saddle ε perturbation + Adam resume | 0.789 | -2.3% over basin (works) |
| Final (legalize + CD polish 120s) | 0.840 | usable polish |

So the *mechanism* of saddle escape works on the Gaussian landscape — it
finds a negative-curvature direction, ε-perturbs along it, and converges
to a strictly lower smooth basin. But the resulting final-proxy lift
after CD polish doesn't beat E117 alone on the hardest bench.

### Recommendation

**Do not promote.** Kill the composition. Future work should test:
- (a) Bigger Adam-step budget (more phase A) with no saddle — does E117
  alone scale better than E120 at large budget?
- (b) Saddle escape on V3 grid (not Gaussian) — E120 should be re-tested
  at 1500s budget directly, not as a composition input.
- (c) E126 (parallel agent) adds Fast diff proxy — same composition with
  more Adam steps available. May change result.

## Pointers
- Code: `code/composed_placer.py` (subclass of E120 with Gaussian proxy)
- Submission: `submissions/e125_gaussian_hessian/placer.py`
- Parent: experiments/E120_continuous_saddle (Hessian saddle escape)
- Parent: experiments/E117_gaussian_density (Gaussian density model)
