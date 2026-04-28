# Writeup Plan: Innovation Prize Submission

Goal: 10-15 page writeup for portfolio. Demonstrates engineering depth,
systematic experimentation, and the ability to use mathematical structure
to guide algorithm design. The tone is "engineer who reads math and
applies it well," not "mathematician proving theorems."

**Credibility calibration.** The author has taken real analysis but not
graduate-level algebra, topology, or statistical mechanics. The writeup
should:
- Observe and describe mathematical structure (the polyhedra, the LP
  duals, the barrier) with precision and clarity
- Reference existing theorems where they illuminate the problem, with
  proper attribution and honest framing ("this is known from X" not
  "we prove")
- Present empirical methodology rigorously (the 22-experiment sweep,
  correlation analysis, ablation)
- Avoid claiming novel theorems or deep formal contributions beyond
  what can be defended in conversation

The strength is the combination of mathematical taste + engineering
execution + honest empirical methodology, not formal novelty.

---

## Narrative Arc

The story in three acts:

**Act 1: A structural observation.** The feasible region of
non-overlapping macro placements is a union of convex polyhedra (one
per pairwise L/R/A/B assignment). Within any single polyhedron,
optimal placement reduces to a linear program. This observation ---
well-known in disjunctive programming but underexploited in VLSI
placement --- decomposes the problem into "which polyhedron"
(combinatorial) and "where within it" (LP). We built a complete
system around this decomposition.

**Act 2: The barrier.** The system plateaued. A systematic
22-experiment sweep isolated the cause: the competition metric is
66.5% congestion, but the LP objective (HPWL) has rho = -0.001
correlation with proxy cost. The decomposition is correct, but the
LP we solve within each polyhedron optimizes the wrong thing. This
is an objective mismatch problem, not a search problem.

**Act 3: Resolution.** The barrier analysis pointed directly to the
fix: differentiate through the actual competition metric instead of
a proxy of a proxy. DPO (Differentiable Proxy Optimization) builds
smooth approximations of all three cost components and runs gradient
descent directly on macro positions. This beats the RePlAce baseline
by 2.3%. The decomposition theory didn't win on its own, but the
understanding it provided was essential to diagnosing the failure and
finding the right approach.

---

## Proposed Structure

### 1. Introduction (1-1.5 pages)

- Problem statement: place N rectangular macros on a 2D chip canvas
  minimizing proxy cost (WL + 0.5*density + 0.5*congestion) with zero
  overlaps
- Context: VLSI macro placement, competition setting, RePlAce baseline
- Our contribution: (1) polyhedral decomposition theorem, (2) barrier
  analysis showing congestion is structural, (3) DPO as a resolution
- Result summary: 1.4246 avg proxy, beats RePlAce by 2.3%

### 2. The Polyhedral Decomposition (2-2.5 pages)

The mathematical heart of the paper. This section should stand alone as
a clean, rigorous development.

- **Definition.** Pairwise non-overlap as a 4-term disjunction. Each
  complete assignment sigma of {L, R, A, B} to all pairs defines a
  convex polyhedron P_sigma. The feasible region F = union of P_sigma.
- **Theorem.** Within any P_sigma, HPWL minimization is an LP.
  Proof sketch: HPWL is a piecewise-linear convex function; the
  constraints are linear; the LP formulation is standard (introduce
  variables for net bounding box coordinates).
- **The LP-dual sensitivity structure.** Dual variables give exact
  marginal costs of each pairwise constraint. This is the "sensitivity
  map" of the local landscape --- available for free from each LP solve.
- **The disconnected topology.** F is disconnected in R^{2N}: no
  continuous overlap-free path connects different polyhedra. Navigation
  between polyhedra requires discrete topology changes (pair flips).
  The graph G = (polyhedra, adjacency) has 4^{N(N-1)/2} vertices and
  degree proportional to N^2 at each vertex.
- **Comparison to standard approaches.** RePlAce relaxes F into a
  smooth convex set and legalizes afterward. Sequence pairs enumerate
  polyhedra implicitly. Our decomposition is exact and makes the
  discrete/continuous structure explicit.

Figures:
- Schematic of 3-macro example showing polyhedra, shared faces, LP
  optima
- The graph G for a small instance (4-5 macros)

### 3. The Navigation System (1.5-2 pages)

What we built on top of the decomposition.

- **SDF initialization.** Signed distance field spreading for good
  density. Produces a starting polyhedron.
- **Assignment extraction.** Reading L/R/A/B from SDF positions.
- **LP-optimal placement.** HiGHS solver, sparse formulation.
- **Surrogate-guided navigation.** GridSurrogate for fast candidate
  evaluation (~0.1ms vs ~50ms for real proxy). Multi-pair cluster
  moves. Acceptance criteria.
- **Robust projection.** Cascade repair + direct overlap repair.
  Guarantees zero overlaps.
- **ClusterScreener.** Multi-tier pruning (Zobrist, displacement
  floor, HPWL bound, axis crowding). Rejects 20-40% of bad moves
  at ~1-10us.

Result: 1.4867 avg proxy (2.0% behind RePlAce). Beats RePlAce on
3/17 benchmarks.

### 4. The Congestion Barrier (2-2.5 pages)

The key empirical finding that motivates the theoretical analysis.
This section should read like a detective story.

- **Profiling.** Component breakdown of proxy cost: congestion 66.5%,
  density 29.3%, WL 4.2%. The gap to RePlAce is almost entirely
  congestion.
- **The overnight sweep.** 22 automated experiments across three
  subproblems: surrogate accuracy (8 experiments), initial topology
  (6), LP formulation (6), plus 2 combinations. All within noise of
  baseline. Table of results.
- **The Miftari experiment.** Cheap signals predict LP-HPWL (rho=0.86)
  but LP-HPWL has zero correlation with proxy cost (rho=-0.001). The
  signal chain breaks. LP optimization is blind to congestion.
- **Swap+LP experiment.** Swapping macro positions and re-solving LP
  can reduce congestion by 44%, but destroys density (+180%). The
  degrees of freedom exist, but no single-objective optimizer can
  exploit them.
- **The structural diagnosis.** The polyhedra decomposition solves the
  wrong LP. The HPWL objective within each polyhedron is uncorrelated
  with the metric that matters (congestion). No amount of better
  navigation can fix an objective mismatch. This is not a search
  problem --- it is a modeling problem.

Figures:
- Scatter plot: LP-HPWL vs proxy cost (rho=-0.001)
- Scatter plot: congestion vs proxy cost (rho=0.825)
- Bar chart: overnight sweep results (all flat)

### 5. Connections to Existing Mathematics (2-2.5 pages)

Frame as: "existing mathematical structures that illuminate why this
problem behaves the way it does." Not claiming novelty --- showing
that the problem has rich structure that existing theory describes,
and that this guided our engineering decisions.

**5a. The disconnected feasible region.** The non-overlap constraints
form a hyperplane arrangement in R^{2N}. Each chamber is one of our
polyhedra. A classical result in algebraic geometry (Zariski, 1962)
shows that the complement of a complex hyperplane arrangement is
always path-connected --- meaning paths between any two chambers
exist in C^{2N} even when they don't exist in R^{2N}. This is why
penalty relaxation (allowing temporary overlaps) works: it's the
real-variable analog of going "around" the walls through a higher-
dimensional space. DPO's overlap penalty with lambda -> infinity is
exactly this continuation.

**5b. LP duals and soft degrees of freedom.** In combinatorial
optimization, "backbone" variables are those frozen across all
optimal solutions (Slaney & Walsh, 2001). Variables with small LP
dual values are not part of the backbone --- they are flexible.
This connects to the concept of "joker" variables in statistical
physics (survey propagation). Our empirical finding that flipping
high-dual pairs (the backbone) destroys quality while the problem
has many near-zero-dual pairs (flexible degrees of freedom) is
consistent with this theory. The 22-experiment sweep can be
interpreted as evidence that the SDF topology sits near a clustering
transition where local moves cannot bridge between solution clusters.

**5c. Tropical structure of HPWL.** HPWL = max(x_i) - min(x_i) is
literally a tropical polynomial in the (max, +) semiring. This means
the objective is piecewise-linear and tropically convex. Recent work
on tropical gradient descent (Talbut & Monod, 2024) shows convergence
guarantees for this class of functions. We mention this not because we
use tropical methods, but because it explains why LP is the natural
solver within each polyhedron: LP solves tropically convex problems
exactly.

**5d. What the math suggested and what actually worked.** The theory
suggested flipping soft pairs (small duals) to cross barriers between
polyhedra. We tested this; it didn't help --- soft-pair flips didn't
move congestion either, because the LP objective is blind to
congestion regardless of which pairs are flipped. The theory correctly
identified the mechanism (soft modes) but couldn't overcome the
objective mismatch. The resolution (DPO) came from the empirical
finding, not the theory. We include the mathematics because it provides
the correct *language* for understanding the problem structure, even
though it didn't directly produce the winning algorithm.

### 6. Differentiable Proxy Optimization (2 pages)

The practical resolution.

- **Motivation from barrier analysis.** The polyhedra decomposition
  optimizes HPWL. The metric is congestion-dominated. Direct
  differentiation through the proxy cost eliminates the mismatch.
- **Architecture.** SDF init, LSE-HPWL, differentiable grid density
  (top-10%), differentiable RUDY congestion (ABU-5%), overlap penalty,
  3-phase penalty continuation, legalization.
- **Connection to the theory.** DPO's overlap penalty is the practical
  implementation of complexification: lambda controls how "expensive"
  it is to traverse infeasible (overlapping) configurations.
  Graduated optimization (Hazan et al., 2016) provides convergence
  theory. The penalty continuation crosses barriers that
  topology-local search cannot.
- **Limitations.** RUDY congestion underestimates real L-routing by
  ~2x. The penalty + legalize paradigm is the same as RePlAce ---
  the novelty is in the objective, not the feasibility handling.

### 7. Empirical Results (1.5-2 pages)

- **Full results table.** DPO v3 on all 17 IBM benchmarks vs RePlAce,
  polyhedra navigation, SDF init, SA baseline.
- **Component analysis.** Where DPO wins (density improvement,
  congestion gradient helps even though RUDY is approximate).
  Where DPO loses (ibm01: RUDY mismatch worst on small dense
  designs).
- **Ablation.** How much does each component contribute?
  SDF init alone (1.50), + DPO phase 1 only (?), + all 3 phases
  (1.42), + SA polish (?). If we have time to run these.
- **Runtime.** All under 60s/benchmark, adaptive scaling.

### 8. Discussion and Open Questions (1-1.5 pages)

- **What we learned.** The polyhedral decomposition is the right
  structural framework for understanding macro placement. It explains
  why RePlAce loses quality at legalization, why SA is slow, and why
  surrogate-guided navigation plateaus. But exploiting the structure
  algorithmically requires solving the right LP --- and the right LP
  includes congestion, which makes it non-convex.
- **The objective mismatch as a general phenomenon.** Many optimization
  systems optimize a proxy of the true objective. Our empirical
  finding (rho=-0.001) is a stark example. The barrier analysis
  methodology (systematic ablation to isolate the mismatch) is
  transferable.
- **Open questions.**
  - Can congestion be incorporated into the LP within each polyhedron?
    (Non-convex, but maybe via successive LP approximation.)
  - Can the polyhedra decomposition be combined with DPO?
    (DPO for global optimization, polyhedra navigation for local
    refinement within the DPO basin.)
  - Is there a practical algorithm for the discrete GAD (soft-pair
    flipping) that actually crosses the congestion barrier?
  - Can tropical geometry provide better optimization algorithms for
    the HPWL component?

### References (~1 page)

Key citations (roughly 30-40):
- Polyhedra / disjunctive programming: Balas, Kronqvist et al.
- Tropical: Allamigeon-Gaubert-Joswig, Talbut-Monod
- Spin glass: Mezard-Parisi-Zecchina, Krzakala-Montanari
- Complexification: Zariski, Allgower-Georg
- Mountain pass: Ambrosetti-Rabinowitz, Henkelman-Jonsson (dimer)
- Analytical placement: DREAMPlace (Lin et al.), RePlAce
- SA baselines: TILOS MacroPlacement
- GAD: Weinan E & Xiang Zhou
- Backbone: Slaney-Walsh
- LP sensitivity: Miftari-Derval
- Graduated optimization: Hazan-Levy-Shalev-Shwartz
- LSE-HPWL: Naylor et al.
- This competition: Partcl/HRT

---

## What Makes This a Strong Portfolio Piece

1. **Systematic engineering methodology.** Built a complete system
   (SDF init, LP solver, surrogate, navigator, projection), hit a
   wall, ran a rigorous 22-experiment ablation to diagnose the root
   cause, and used the diagnosis to pivot to a better approach. This
   is how good engineering research works.

2. **Honest failure analysis.** The polyhedra navigator plateaus. We
   don't hide this --- we quantify exactly why (rho=-0.001, congestion
   is 66.5% of cost) and show the systematic evidence. The failure is
   more interesting than the success because it reveals something
   structural about the problem.

3. **Mathematical literacy applied to engineering.** The writeup shows
   the ability to read across fields (combinatorial optimization, LP
   theory, analytical placement, tropical geometry) and apply relevant
   ideas to a concrete problem. The math is used as a lens, not as
   an end in itself.

4. **Practical results.** DPO beats the RePlAce baseline by 2.3%.
   The theory-guided understanding of the problem led to a working
   solution. The full pipeline (SDF -> DPO -> legalization) is
   production-quality code.

5. **Intellectual honesty about scope.** The mathematical connections
   are presented as observations and references to existing theory,
   not as original formal contributions. The original contributions
   are the decomposition-based system design, the empirical barrier
   analysis, and the DPO algorithm.

---

## Work Required

| Task | Effort | Dependency |
|------|--------|------------|
| Write sections 1-3 (decomposition, navigator, barrier) | 2-3 days | Existing code + results |
| Write section 4 (mathematical connections) | 2-3 days | Existing theory.md (needs editing/tightening) |
| Write sections 5-6 (DPO, results) | 1-2 days | Existing results |
| Run ablation experiments for section 6 | 0.5 days | Existing code |
| Write sections 7-8 (discussion, refs) | 1 day | Everything else |
| Figures and formatting | 1-2 days | Parallel with writing |
| Revision pass | 1-2 days | Complete draft |
| **Total** | **~8-12 days** | |

---

## Competitive Improvement (Parallel Track)

To make results more competitive while writing, low-effort improvements:

1. **Position normalization + adaptive phases** (1 day) --- eliminates
   hardcoded constants, should improve weak benchmarks (ibm01, ibm06)
2. **Multi-seed (3 seeds, take best)** (0.5 day) --- free +0.5-1%
3. **SA polish after DPO** (0.5 day) --- free +0.2-0.5%
4. **Time-budgeted optimization + GPU support** (1 day) --- scales
   to hidden test case

These are orthogonal to writing and can be done in parallel. Target:
1.38-1.40 avg proxy (4-5% better than RePlAce) with respectable but
not heroic effort on the implementation side.

---

## Key Decision: What to Cut

theory.md is ~15 pages of dense mathematics. Most of it represents
literature review and speculative connections, not things we actually
built or tested. The writeup should include only what we can speak to
from direct experience: things we implemented, tested, or used to
make decisions.

**Include (we built or tested it):**
- Polyhedral decomposition (implemented: LP solver, assignment extraction)
- LP duals as sensitivity map (implemented: dual-guided navigation)
- Barrier analysis (tested: 22-experiment sweep, Miftari correlation)
- DPO (implemented: full pipeline, beats RePlAce)

**Include briefly as context (we read it, it informed design):**
- Complexification / penalty relaxation connection (explains why DPO works)
- Backbone theory / soft pairs (explains dual-guided navigation rationale)
- Tropical HPWL (explains why LP is the right solver within polyhedra)

**Cut (interesting but we can't defend it in depth):**
- Quantum tunneling / SQA analysis
- Survey propagation / cavity methods
- Stratified Morse theory / persistent homology
- Information geometry / Fisher-Rao
- Population annealing
- Homotopy continuation
- Spectral decomposition of L(K_n)
- Benders / backdoor variables / ejection chains
- Diffusion models, GCS, DEE
- Everything in theory.md sections 4b-4d and 5b-5d

This material can live in theory.md as a supplementary reference
document for interested readers, with a footnote in the writeup.

---

## See Also

- [theory.md](theory.md) --- full theoretical foundations (source material)
- [approach.md](approach.md) --- polyhedra navigation architecture
- [results.md](results.md) --- empirical results
- [dpo.md](dpo.md) --- DPO theory
- [dpo_next.md](dpo_next.md) --- DPO improvements
