# Contributions

Every claim below has corresponding code, data, or experiments that we
ran. Nothing is included on the basis of "we read about it" alone.

---

## 1. Polyhedral decomposition applied to macro placement

**Claim:** The non-overlap feasible region is a union of convex
polyhedra (one per pairwise L/R/A/B assignment). Within any polyhedron,
HPWL minimization is a linear program. LP dual variables give exact
marginal costs of each pairwise constraint — a complete sensitivity
map available for free from each solve.

**What's novel:** The decomposition itself is well-known in disjunctive
programming (Balas 1979, Kronqvist et al. 2025). What we contribute is
a *complete working system* built on it for macro placement: SDF init,
assignment extraction, HiGHS LP with dual extraction, surrogate-guided
navigation, robust projection. And more importantly, we contribute the
empirical characterization of its failure mode (see contribution 2).

**What's not novel:** The math. Disjunctive programming, LP duality,
piecewise-linear HPWL formulation — all textbook.

**Evidence:** `submissions/polyhedra/placer.py`, 1.49 avg proxy on 17
IBM benchmarks, 0 overlaps.

---

## 2. Barrier diagnosis: the objective mismatch

**Claim:** The polyhedra navigation system plateaus because LP-HPWL has
rho = -0.001 correlation with the competition proxy cost. Congestion is
66.5% of proxy cost. HPWL is uncorrelated with congestion (rho = 0.072).
HPWL and density anti-correlate (rho = -0.536). No LP-based ranking of
polyhedra can predict proxy quality.

**What's novel:** This specific empirical finding for the ICCAD04 proxy
metric. The methodology: systematic 22-experiment ablation across three
independent subproblems (surrogate accuracy, initial topology, LP
formulation), combined with Miftari-style correlation analysis between
LP values and refined proxy cost across 24 feasible topologies. The
diagnosis that "this is an objective mismatch, not a search problem" is
a structural insight that redirects algorithm design.

**What's not novel:** Correlation analysis, ablation studies, the
general idea that proxy objectives can mislead.

**Evidence:**
- 22-experiment overnight sweep: `results/overnight_run.log`
- Miftari correlation matrix: `docs/results.md` "Miftari" section
- Component breakdown: congestion 66.5%, density 29.3%, WL 4.2%
- All 22 experiments within +/-0.5% of baseline

**Caveat:** The Miftari rho=-0.001 was measured on ibm01 only. Must
generalize to 2-3 more benchmarks before the writeup can make the claim
broadly. (See `todo.md`.)

---

## 3. DPO: differentiable proxy optimization

**Claim:** Differentiating through the actual competition metric
f(p) = WL + 0.5*D + 0.5*C — including a fully differentiable RUDY
congestion model — beats RePlAce by 2-3% on the IBM benchmarks.

**What's novel (pending verification):** Including the congestion
gradient directly in placement optimization. DREAMPlace (Lin et al.
2019) differentiates HPWL + electrostatic density. Congestion-aware
DREAMPlace variants (Lu et al. 2020) use congestion as *net weights on
HPWL* — the gradient of "congestion-weighted HPWL" is not the gradient
of congestion itself. Our gradient computes dC/dp directly: how moving
each macro changes the top-5% congested cells.

**THIS CLAIM NEEDS LITERATURE VERIFICATION.** If any 2020-2026 work
already includes dC/dp in the backward pass for analytical placement,
our contribution downgrades to "applied to this specific competition
metric with top-k focusing." Still useful, but not "first."

**What's not novel:** LSE-HPWL (Naylor 2001), differentiable grid
density (standard), penalty continuation (Bertsekas 1982, Hazan 2016),
Adam optimizer, top-k via PyTorch.

**Evidence:** `submissions/dpo/placer.py`, 1.4264 ± 0.0025 avg proxy
across 5 seeds (range 1.4237–1.4301). All seeds beat RePlAce (1.4578).
Ablation: removing congestion gradient → 1.5092 (+5.9%), removing
density gradient → 1.7342 (+21.7%), random init → 4.94 (+247%).

---

## 4. The penalty-as-barrier-crossing interpretation

**Claim:** Legal-state representations are fundamentally trapped.
Any method that represents only non-overlapping placements is confined
to a disconnected feasible region where local moves cannot cross the
congestion barrier. DPO's overlap penalty provides the escape: at low
lambda, the optimizer traverses infeasible configurations between
polyhedra; at high lambda, it converges to a feasible placement in a
different polyhedron.

**Geometric interpretation:** The barrier-crossing mechanism can be
understood as dimensional lifting. Give each macro a z-coordinate;
two macros at different z-heights don't overlap even if their x-y
projections collide. A z-penalty (mu * sum(z^2)) pushes macros
toward z=0. Annealing mu: 0 → infinity continuously deforms a 3D
arrangement (where macros freely pass over each other) into a 2D one.
The z-dimension IS the "imaginary dimension" from complexification
(Zariski 1962), made geometric: in R^{2N} the feasible region is
disconnected, but in R^{3N} it is connected. DPO's flat overlap
penalty is the projection of this mechanism onto 2D — it achieves
the same barrier crossing without explicitly adding the z-coordinate.

**What's novel:** (1) The empirical demonstration that legal-state
methods are trapped: 22 polyhedra experiments all noise, 0/190
cluster swaps accepted, hierarchical decomposition 18-28% worse
than flat DPO. (2) The interpretive connection between penalty
continuation and complexification, with the z-lifting as a geometric
bridge between the two. We claim observation, not theorem.

**What's not novel:** Penalty methods, complexification, graduated
optimization — all well-established.

**Evidence (quantified):** On ibm01, DPO changes 3,556 of 30,135
pairwise L/R/A/B assignments (11.8%). On ibm10, 15,539 of 308,505
(5.0%). Transitions are overwhelmingly perpendicular flips (L↔B,
R↔A — ~98% of changes), consistent with crossing nearby polyhedra
boundaries where the binding constraint switches between horizontal
and vertical separation. DPO does not perform deep topological
restructuring — it optimizes within and across adjacent polyhedra.

---

## 5. Non-decomposability of the proxy cost

**Claim:** The proxy cost f(p) = WL + 0.5*D + 0.5*C cannot be
productively decomposed — by component, by scale, or by structural
partition. Every decomposition produces worse results than joint
optimization.

**What's novel:** The empirical demonstration across multiple
decomposition strategies, with quantified failure modes.

**Evidence:**
- *By component:* LP-HPWL (optimizing WL alone) has rho=-0.001 with
  proxy cost. WL and density anti-correlate (rho=-0.536). Congestion
  is 74.9% of proxy cost but uncorrelated with HPWL (rho=0.072).
- *By structure:* Polyhedra decomposition (which polyhedron vs where
  within it) optimizes 5.8% of the objective. 22-experiment sweep
  across surrogate, topology, and LP: all within noise.
- *By scale:* Hierarchical clustering experiments. Three approaches
  tested (coarse DPO + expand, cluster-pull refinement, cluster-swap
  search). All worse than flat DPO. 0/190 cluster-pair swaps improved
  proxy cost — the coarse arrangement is not the bottleneck.
- *Resolution:* DPO optimizes f(p) directly, jointly over all
  components at all scales. This is not a better decomposition —
  it is the abandonment of decomposition.

**Implication for the field:** Macro placement with composite
objectives (WL + density + congestion) may be fundamentally resistant
to the divide-and-conquer strategies that work for single-objective
problems. The coupling between cost components through shared grid
cells means that any separation of concerns loses the information
that matters most.

---

## 6. Low seed variance as evidence of effective continuation

**Claim:** DPO's 0.45% seed variance (5 seeds, all beating RePlAce)
is evidence that the penalty continuation is working, not a limitation.
A method trapped in random local minima would show high variance. The
low variance means the penalty schedule reliably collapses the
landscape to a consistent attractor.

**What this implies for compute scaling:** Multi-start parallelism
has steep diminishing returns. Best-of-200 seeds under a normal model
gains only ~0.1% over best-of-5. The continuation already does the
heavy lifting. Accessing qualitatively better basins (if they exist)
would require fundamentally different exploration mechanisms, not
more starts.

**Contrast:** Random init → 4.94 avg (massive variance across the
raw landscape). SDF + DPO continuation compresses this to a 0.45%
range. The combined method has effectively solved the exploration
problem for these benchmarks.

**Evidence:** 5-seed ablation data; diminishing returns analysis
under normal model.

---

## 7. Methodology: the complete experimental trajectory

**Claim:** The sequence *build system on structural insight* ->
*hit wall* -> *run systematic ablation to diagnose root cause* ->
*use diagnosis to pivot* is a transferable methodology for applied
optimization research.

**What's novel:** The specific application and the completeness of the
documentation. Each step is quantified: 1.49 (system), 22 experiments
(diagnosis), rho=-0.001 (root cause), 1.42 (resolution). Additionally,
the post-resolution analysis (ablation study, non-decomposability
experiments, seed variance analysis) quantifies why the resolution
works and what its limits are.

**Evidence:** The full experiment log, results history, and approach
documents.

---

## Explicitly excluded

The following topics from `docs/theory.md` are excluded from the
writeup because we cannot defend them from direct experience:

- Quantum tunneling / SQA analysis
- Survey propagation / cavity methods / RSB
- Stratified Morse theory / persistent homology
- Information geometry / Fisher-Rao gradient flows
- Population annealing
- Homotopy continuation (PHCpack, Bertini)
- Spectral decomposition of L(K_n)
- Benders decomposition / backdoor variables
- Ejection chains / ALNS
- Diffusion models (DiffPlace, DiffUCO)
- Graphs of Convex Sets (GCS)
- Dead-End Elimination (DEE)
- Discrete Schrodinger Bridges
- Kolmogorov complexity sampling
- Semi-discrete optimal transport / Laguerre tessellations

These are interesting literature connections but we never implemented
or tested them. They can live in `docs/theory.md` as supplementary
material with a footnote, but they do not belong in the writeup body.
