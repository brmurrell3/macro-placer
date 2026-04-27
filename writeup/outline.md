# Outline

Stripped-down structure. Every section maps to something we built,
tested, or measured. Page estimates assume 12-14 page target.

The paper has two diagnosis-pivot cycles:
  polyhedra → barrier → DPO (first pivot)
  DPO → RUDY limit → incremental evaluator + CD (second pivot)

---

## 1. Introduction (1 page)

- Problem: place N rectangular macros, minimize proxy cost
  (WL + 0.5*density + 0.5*congestion), zero overlaps
- Competition context, RePlAce baseline (1.46), leaderboard (1.12)
- Our result: 1.12 avg (23% improvement over RePlAce, matches
  leaderboard), and the diagnostic trajectory that got there
- Two pivots: (1) objective mismatch diagnosis → DPO, (2) RUDY
  fidelity diagnosis → incremental evaluator + full-proxy CD
- Contribution summary: polyhedral decomposition, barrier diagnosis,
  DPO with congestion gradient, RUDY fidelity analysis, incremental
  evaluator (4657x speedup), full-proxy coordinate descent

## 2. The polyhedral decomposition (2 pages)

- Pairwise non-overlap as 4-term disjunction
- Complete assignment sigma defines a convex polyhedron P_sigma
- Within P_sigma, HPWL minimization is an LP (proof sketch)
- LP duals as sensitivity map: marginal cost of each pairwise constraint
- The disconnected topology of F = union of P_sigma
- Comparison to RePlAce (relaxes F into smooth set + legalizes)

Figures: 3-macro schematic, LP dual interpretation.

## 3. The navigation system (1.5 pages)

- SDF initialization (density-aware analytical spreading)
- Assignment extraction from SDF positions
- HiGHS LP solve + dual extraction
- GridSurrogate for fast candidate evaluation
- Surrogate-guided navigation with cluster moves
- Robust projection (cascade repair + direct overlap repair)
- Result: 1.49 avg proxy, 3/17 benchmarks beat RePlAce

## 4. The congestion barrier (2 pages)

The empirical heart of the paper. First diagnosis-pivot cycle.

- Component breakdown: congestion 66.5%, density 29.3%, WL 4.2%
- The overnight sweep: 22 experiments, all within noise
- The Miftari experiment: rho(LP-HPWL, proxy) = -0.001
  - LP-HPWL predicts WL (rho=0.852) but not proxy
  - HPWL and density anti-correlate (rho=-0.536)
  - Congestion dominates proxy (rho=0.825)
- The swap+LP experiment: congestion IS reachable (-44%) but at
  density cost (+180%)
- Structural diagnosis: LP optimizes the wrong function. No amount of
  better navigation fixes an objective mismatch.

Figures: LP-HPWL vs proxy scatter, congestion vs proxy scatter,
overnight sweep bar chart.

## 5. First pivot: Differentiable proxy optimization (2 pages)

- Motivation from barrier analysis: differentiate through f(p) directly
- Making each component differentiable:
  - LSE-HPWL (standard, cite Naylor 2001)
  - Grid density with top-10% via torch.topk
  - RUDY congestion: smooth bbox -> fractional cell overlap -> ABU-5%
  - Overlap penalty with pairwise ReLU
- 3-phase penalty continuation (exploration -> refinement -> sharpening)
- Legalization via iterative overlap repair
- The congestion gradient: what it tells each macro and why this signal
  doesn't exist in HPWL-based optimization
- Result: 1.38 avg proxy (5.1% over RePlAce), all seeds beat RePlAce
- DPO ablation: density gradient +21.7%, congestion gradient +5.9%,
  multi-phase +2.5%, SDF init +247%

## 6. Why DPO crosses the barrier (1 page)

- **Legal-state representations are trapped.** 22 experiments + 0/190
  cluster swaps + hierarchical decomposition all confirm: local moves
  cannot cross the congestion barrier. The feasible region's disconnected
  structure prevents reaching better basins.
- **Penalty continuation as barrier crossing.** DPO changes 5-12% of
  pairwise L/R/A/B assignments between init and output. Overwhelmingly
  perpendicular flips (L↔B, R↔A).
- **Dimensional lifting interpretation.** The overlap penalty is
  equivalent to lifting macros into R^{3N} (adding z-coordinates),
  where the feasible region becomes connected. Annealing the z-penalty
  continuously deforms a 3D arrangement into a 2D one. Connection to
  complexification (Zariski 1962).
- **Low seed variance (0.45% range)** as evidence the continuation
  reliably collapses the landscape to a consistent attractor.

## 7. The RUDY limit (1.5 pages)

Second diagnosis-pivot cycle. DPO's success revealed a new ceiling.

- **DPO worsens 4/17 benchmarks vs SDF init.** Optimization actively
  degrades ibm01, ibm02, ibm06, ibm12. Not just insufficient — harmful.
- **Congestion weight sweep: all worse.** Weights 0.5-1.5 tested;
  monotonically worse. The problem is gradient direction, not magnitude.
- **Cell-by-cell RUDY analysis (ibm01):**
  - Real/RUDY gap is 3.1x (not 2x as estimated from aggregate metrics)
  - Top-5% hotspot overlap: 10.9% (near-random, Jaccard 0.057)
  - Three structural sources: L-routing vs uniform bbox (2.74x),
    macro blockage (28% of real congestion missing entirely),
    spatial smoothing
  - Vertical congestion worst (correlation 0.327 vs horizontal 0.635)
- **The diagnosis:** DPO's congestion gradient points at the wrong
  cells. The gradient is not just noisy — it is systematically wrong.
  No amount of DPO tuning (seeds, steps, weights, diverse priors)
  can overcome this. Confirmed: 64-seed batched DPO collapses to
  identical basin; diverse priors (Will, greedy, random) flat on --all.
- **The implication:** Don't fix the differentiable approximation —
  bypass it. Evaluate the real proxy cost directly.

Figures: RUDY vs real congestion heatmap, per-cell ratio distribution.

## 8. Second pivot: Full-proxy coordinate descent (2 pages)

- **The infrastructure prerequisite: incremental evaluator.** Real
  proxy cost via compute_proxy_cost takes ~30s per call. CD needs
  thousands of move evaluations per sweep. Solution: IncrementalProxy-
  Evaluator with per-net min/max trackers, bin-density grid, RUDY
  congestion deltas, smoothing via cumsum. 4657x speedup (6.5 ms/move
  vs 30s full). Bit-for-bit parity verified.
- **Breakpoint enumeration.** For each macro on each axis, the proxy
  cost is piecewise-smooth with breakpoints at net endpoints and bin
  grid lines. Enumerate breakpoints, evaluate via incremental cost
  queries, pick the minimum. No gradient, no golden section — exact
  1D search on the actual objective.
- **The ibm02 basin lock.** All 5 DPO seeds converge to byte-identical
  placement on ibm02 (1.6888). CD breaks this: 1.6888 → 1.1534 (-32%).
  DPO's RUDY gradient cannot see the path; CD's exact evaluator can.
- **CD trades cheap WL for expensive density/congestion.** WL goes UP
  13-15% while density drops 21-32% and congestion drops 23-40%.
  Exactly the trade-off DPO cannot make because RUDY misidentifies
  which cells are congested.
- Result: 1.12 avg proxy (23.2% over RePlAce). Every benchmark improves
  over DPO. Zero regressions. 10 min/benchmark budget.

## 9. Empirical results (2 pages)

- Full results table: CD vs DPO vs RePlAce vs polyhedra nav vs SDF
  on all 17 benchmarks
- The improvement trajectory: 1.50 (SDF) → 1.49 (polyhedra) → 1.38
  (DPO) → 1.12 (CD). Each pivot driven by diagnosis, not intuition.
- DPO ablation study (unchanged):
  - With/without congestion gradient (+5.9%)
  - With/without density gradient (+21.7%)
  - Single-phase vs 3-phase (+2.5%)
  - SDF init vs random init (+247%)
- DPO multi-seed stability: 0.45% range (v3), 1.0% range (best-of-v2)
- CD convergence curve: sweep 1 captures ~60% of improvement; 3-13
  sweeps in 10 min capture ~85% of 40-min value
- Per-component analysis: CD reduces congestion 23-40% per benchmark
  while DPO was limited to ~5-10%
- Dead ends: congestion weight sweep (killed), batched seeds (basin
  collapse), diverse priors (flat on --all), congestion-only refine
  (marginal -0.33%)

## 10. Discussion (1.5 pages)

- **Non-decomposability confirmed, then transcended.** DPO optimized
  f(p) jointly but through an inaccurate model. CD optimizes f(p)
  jointly through the exact evaluator. The lesson: non-decomposability
  applies to the model as well as the objective. A joint optimizer
  on a wrong model is worse than a coordinate optimizer on the right
  one.
- **The objective mismatch recurs at every level.** LP-HPWL vs proxy
  (rho=-0.001); RUDY congestion vs real congestion (10.9% hotspot
  overlap); DPO's gradient vs the true gradient. Each approximation
  introduces an objective mismatch that caps performance. The
  diagnosis methodology (systematic ablation + correlation analysis)
  detected each one.
- **"Bypass, don't fix" as an algorithmic design principle.** The
  natural response to "RUDY is wrong" is to build a better RUDY.
  The winning response was to bypass RUDY entirely via an incremental
  evaluator. This is a general lesson: when an approximation is
  structurally wrong (not just noisy), exact evaluation with a faster
  data structure beats a more accurate approximation.
- **Infrastructure unlocks algorithms.** CD was not a novel algorithm —
  coordinate descent is textbook. The 4657x speedup from the
  incremental evaluator was the gate. Without it, CD at 30s/eval
  would complete ~2 sweeps in an hour; with it, 13 sweeps in 10 min.
  The same algorithm, gated entirely by infrastructure speed.
- **Limitations.** CD plateaus after 10-15 sweeps (ibm17/18 still
  improving at budget). LNS rip-up-and-reinsert would escape CD
  local minima but is not yet implemented. Per-benchmark budget
  allocation could further improve results.
- **The polyhedral decomposition is the right framework for
  *understanding*.** It correctly describes the feasible region's
  structure, explains the congestion barrier, and motivated both
  pivots. But it doesn't yield algorithmic advantage — a theme
  throughout: structural insight guides diagnosis, not solution.

## References (~1 page)

Key citations (~25-30, only things we used or directly built on):
- Balas (disjunctive programming)
- Kronqvist et al. 2025 (P-split formulation)
- Naylor 2001 (LSE-HPWL)
- Lin et al. 2019 (DREAMPlace)
- Lu et al. 2020 (congestion-aware DREAMPlace)
- NV-Place / C3PO (ASP-DAC 2026, differentiable RUDY)
- Hazan et al. 2016 (graduated optimization)
- Bertsekas 1982 (penalty methods)
- Zariski 1962 (hyperplane arrangement complement connectivity)
- Slaney & Walsh 2001 (backbone variables)
- Miftari et al. 2024-2026 (LP sensitivity analysis)
- Mobahi & Fisher 2015 (Gaussian smoothing)
- RePlAce, TILOS benchmarks
- Partcl/HRT competition
