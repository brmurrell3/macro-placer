# Outline

Stripped-down structure. Every section maps to something we built,
tested, or measured. Page estimates assume 10-12 page target.

---

## 1. Introduction (1 page)

- Problem: place N rectangular macros, minimize proxy cost
  (WL + 0.5*density + 0.5*congestion), zero overlaps
- Competition context, RePlAce baseline (1.46)
- Our result: 1.42 (2-3% improvement), and more importantly, the
  trajectory that got there
- Contribution summary: polyhedral decomposition system, barrier
  diagnosis, DPO resolution

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

The empirical heart of the paper.

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

**Depends on:** TODO item 2 (generalize rho=-0.001).

## 5. Differentiable proxy optimization (2 pages)

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

**Depends on:** TODO item 3 (literature check on congestion novelty).

## 6. Why DPO crosses the barrier (1.5 pages)

- **The fundamental wall: legal-state representations are trapped.**
  Any method that represents only non-overlapping placements — polyhedra
  navigation, sequence pairs, B*-trees — is confined to a disconnected
  feasible region. Local moves (1-5 pair flips) cannot cross the
  congestion barrier because it is hundreds of flips wide. 22 experiments
  + hierarchical clustering (0/190 swaps accepted) confirm this
  empirically. The feasible region's structure prevents the search from
  reaching better basins.

- **Penalty continuation as barrier crossing.** At low lambda, the
  overlap penalty lets the optimizer traverse infeasible configurations.
  At high lambda, it converges to a feasible placement in a (potentially
  different) polyhedron. Quantified: DPO changes 5-12% of pairwise
  L/R/A/B assignments between init and output.

- **Connection to complexification.** The overlap penalty is the
  real-variable analog of Zariski's result: paths between disconnected
  real polyhedra exist if you allow "imaginary" (infeasible) intermediate
  states. DPO's penalty parameter lambda controls the cost of traversing
  the infeasible region, analogous to the distance through the
  complexified space.

- **Geometric interpretation: dimensional lifting.** An alternative to
  the flat penalty: give each macro a z-coordinate. Two macros at
  different z-heights don't overlap even if their x-y projections
  collide. A z-penalty (mu * sum(z^2)) pushes macros toward z=0.
  Annealing mu from 0 → infinity continuously deforms a "3D" arrangement
  (where macros pass over each other freely) into a 2D one. The
  z-dimension IS the imaginary dimension from complexification, made
  geometric. This provides the cleanest description of why infeasible-
  state traversal is necessary: in R^{2N} the feasible region is
  disconnected, but in R^{3N} (with z) it is connected.

- Graduated optimization convergence theory (cite Hazan et al. 2016).

- **Low seed variance as evidence of effective continuation.** 5 seeds
  show 0.45% range (1.4237–1.4301), all beating RePlAce. Best-of-200
  under a normal model gains only ~0.006 additional. A method trapped
  in random local minima would show HIGH variance. The low variance
  means the penalty schedule reliably collapses the landscape to a
  consistent attractor — the practical signature of a good continuation.
  Contrast: random init gives 4.94 (variance across the full landscape
  is enormous); SDF + penalty continuation compresses this to 0.45%.

**Now quantified:** ibm01: 3,556/30,135 pairs change (11.8%); ibm10:
15,539/308,505 (5.0%). Overwhelmingly perpendicular flips (L↔B, R↔A).

## 7. Empirical results (2 pages)

- Full results table: DPO vs RePlAce vs polyhedra nav vs SDF on all
  17 benchmarks
- Ablation study: contribution of each component
  - With/without congestion gradient (+5.9%)
  - With/without density gradient (+21.7%)
  - Single-phase vs 3-phase (+2.5%)
  - SDF init vs random init (+247%)
- Multi-seed stability: 0.45% range, diminishing returns analysis
- Runtime analysis (all under 60s/benchmark)
- Where DPO wins (density improvement, congestion gradient helps even
  though RUDY is approximate) and where it loses (ibm01: RUDY mismatch
  worst on small dense designs)

**All TODO items now completed.** Ablation data, multi-seed, version
cleanup, and literature check are in `writeup/experiment_notes.md`.

## 8. Discussion (1.5 pages)

- **Non-decomposability of the proxy cost.** The proxy cost couples
  WL, density, and congestion through shared macro positions. Every
  decomposition we tested — by component (LP optimizes HPWL only),
  by structure (polyhedra: topology vs position), by scale
  (hierarchical: coarse arrangement vs fine positioning) — produced
  worse results than joint optimization. 0/190 cluster-swap tests
  improved proxy cost; 22/22 overnight sweep experiments were noise.
  The proxy cost is fundamentally resistant to divide-and-conquer.
- **The objective mismatch as a general phenomenon.** Many optimization
  systems optimize a proxy of the true objective; our rho=-0.001 is a
  stark quantified example. The diagnosis methodology (systematic
  ablation + correlation analysis) is transferable.
- **The polyhedral decomposition is the right framework for
  *understanding*, not for *solving*.** It correctly describes the
  feasible region's structure and explains why penalty methods can
  cross barriers. But the decomposition doesn't align with the
  objective's structure, so it doesn't yield algorithmic advantage.
- **Limitations.** RUDY underestimates real L-routing congestion by
  ~2x; penalty+legalize is the same paradigm as RePlAce (novelty is
  in the objective, not the feasibility handling).
- **Can more compute help?** Multi-seed analysis shows diminishing
  returns: 5 seeds give 0.45% range; best-of-200 under a normal model
  gains ~0.1% more. The low variance is evidence that the penalty
  continuation is effective, not a sign that better basins are just
  out of reach. Brute-force parallelism (multi-start) adds little
  because DPO already reliably finds the same attractor. Fundamentally
  different exploration — Hamiltonian Monte Carlo, parallel tempering
  across penalty schedules, or learned congestion models — would be
  needed to access qualitatively different basins, if they exist.
- **Open questions.** (1) Can the non-decomposability be overcome?
  The evidence suggests the coupling is deep enough that end-to-end
  gradient methods may be inherently superior for composite objectives.
  (2) Is the RUDY congestion model the binding constraint? The 2x
  underestimate limits the congestion gradient's accuracy. A more
  faithful differentiable congestion model is the clearest path to
  further improvement, but requires routing-level training data.
  (3) Does the z-lifting formulation (dimensional lifting to R^{3N})
  provide better gradient geometry than the flat overlap penalty?
  The current penalty is the 2D projection of this mechanism; the
  explicit 3D version might provide smoother barrier crossings.

## References (~1 page)

Key citations (~20-25, only things we used or directly built on):
- Balas (disjunctive programming)
- Kronqvist et al. 2025 (P-split formulation)
- Naylor 2001 (LSE-HPWL)
- Lin et al. 2019 (DREAMPlace)
- Lu et al. 2020 (congestion-aware DREAMPlace)
- Hazan et al. 2016 (graduated optimization)
- Bertsekas 1982 (penalty methods)
- Zariski 1962 (hyperplane arrangement complement connectivity)
- Slaney & Walsh 2001 (backbone variables)
- Miftari et al. 2024-2026 (LP sensitivity analysis)
- Mobahi & Fisher 2015 (Gaussian smoothing)
- RePlAce, TILOS benchmarks
- Partcl/HRT competition
