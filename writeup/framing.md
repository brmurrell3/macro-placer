# Framing

The writeup tells one story in three acts. Each act has a clear
deliverable that we built, tested, and can speak to from direct
experience. Nothing in this document references work we only read
about or connections we only sketched.

---

## The story

**Act 1 — A structural observation.**
The feasible region of non-overlapping macro placements is a union of
convex polyhedra, one per pairwise L/R/A/B assignment. Within any
single polyhedron, HPWL minimization is a linear program. This
decomposes the problem into *which polyhedron* (combinatorial, NP-hard)
and *where within it* (continuous, polynomial). We built a complete
system on this decomposition: SDF initialization, LP solver with dual
extraction, surrogate-guided navigation across polyhedra, robust
overlap repair. Result: 1.49 avg proxy, 2% behind RePlAce.

**Act 2 — The barrier.**
The system plateaued. A 22-experiment ablation sweep across surrogate
accuracy, initial topology, and LP formulation produced no improvement
beyond noise. Correlation analysis revealed the cause: LP-HPWL has
rho = -0.001 with the competition metric. The metric is 66.5%
congestion, and HPWL is blind to congestion. The decomposition solves
the *wrong LP*. This is an objective mismatch, not a search problem.

But the barrier is deeper than just the LP objective. The proxy cost
*cannot be productively decomposed* — not by component (HPWL vs
density vs congestion), not by structure (which polyhedron vs where
within it), not by scale (coarse cluster arrangement vs fine
positioning). Hierarchical experiments confirmed this: 0/190 cluster-
pair swaps improved proxy cost. Every decomposition loses information
about the coupling between cost components through shared grid cells.
Legal-state representations that stay in the feasible region are
trapped: the disconnected polyhedra prevent the coordinated moves
needed to cross the congestion barrier.

**Act 3 — Resolution.**
The diagnosis pointed directly to the fix: differentiate through the
actual competition metric, and allow temporary violations of the
feasibility constraint. DPO builds smooth approximations of all three
cost components and runs gradient descent on macro positions. The
overlap penalty with annealed lambda is the key mechanism: it
temporarily dissolves the walls between polyhedra, letting the
optimizer reach arrangements that no legal-state method can access.
Geometrically, the penalty serves as a dimensional lift — the
"z-dimension" from complexification made practical — connecting the
disconnected 2D feasible region through infeasible intermediate states.

This beats RePlAce by 2-3%. The decomposition theory didn't win on
its own, but the understanding it provided was essential: it told us
exactly *why* legal-state methods are trapped, *why* decomposition
fails, and *why* the resolution must optimize the full coupled
objective through infeasible space.

---

## Tone

Engineer who reads math and applies it well. Not a mathematician
proving theorems. The strength is the combination of:

- Structural observation applied to a concrete problem
- Rigorous empirical methodology (22-experiment sweep, correlation
  analysis, component decomposition)
- Honest failure analysis that reveals something about the problem
- A resolution that follows from the analysis, not from trial and error

The math is a lens for understanding the problem, not an end in
itself. Every mathematical reference in the writeup is either something
we implemented, something we tested, or something that directly
explains an empirical observation.

---

## What the reader takes away

1. **Macro placement has clean mathematical structure** (union of
   polyhedra, LP within each) that is underexploited in practice.

2. **That structure alone doesn't win** because the natural LP
   objective (HPWL) is uncorrelated with the metric that matters
   (congestion-dominated proxy cost). This is quantified, not
   hand-waved.

3. **Decomposition is counterproductive.** The proxy cost couples
   wirelength, density, and congestion through shared macro positions.
   Every attempt to separate the problem — by component (LP optimizes
   HPWL only), by scale (hierarchical coarse/fine), or by structure
   (which polyhedron vs where within it) — loses information and
   produces worse results. Tested via 22-experiment sweep, hierarchical
   clustering experiments (0/190 cluster swaps improved proxy cost),
   and ablation studies.

4. **Systematic diagnosis beats intuition.** The 22-experiment sweep
   and correlation analysis are transferable methodology. The failure
   is more interesting than the success.

5. **Direct optimization of the true objective resolves the mismatch.**
   DPO is not a complex system — it is the simple, correct response to
   a precisely diagnosed problem. It succeeds because it does not
   decompose: it differentiates through the full objective, moving all
   macros at once, seeing all cost components in every gradient step.

---

## What this is NOT

- Not a theory paper. We do not prove new theorems.
- Not a systems paper. The codebase is competition-grade, not a
  reusable framework.
- Not a survey. Mathematical connections are included only when they
  explain something we observed empirically.
- Not a leaderboard paper. The 2-3% over RePlAce matters less than the
  trajectory that got there.
