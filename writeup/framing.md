# Framing

The writeup tells one story in three acts with two pivots. Each act
has a clear deliverable that we built, tested, and can speak to from
direct experience. Nothing references work we only read about or
connections we only sketched.

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
overlap repair, ClusterScreener for pre-projection pruning. Result:
1.49 avg proxy, 2% behind RePlAce.

**Act 2 — The first barrier.**
The polyhedra system plateaued. A 22-experiment ablation sweep across
surrogate accuracy, initial topology, and LP formulation produced no
improvement beyond noise. Correlation analysis revealed the cause:
LP-HPWL has rho = -0.001 with the competition metric. The metric is
74% congestion, and HPWL is blind to congestion. The decomposition
solves the *wrong LP*. This is an objective mismatch, not a search
problem. The deeper finding: the proxy cost *cannot be productively
decomposed* — not by component, not by structure, not by scale. Legal-
state representations that stay in the feasible region are trapped:
the disconnected polyhedra prevent the coordinated moves needed to
cross the congestion barrier.

**First pivot — DPO.**
Differentiate through the actual competition metric, and allow
temporary violations of the feasibility constraint. DPO builds smooth
approximations of all three cost components and runs gradient descent
on macro positions. The overlap penalty with annealed lambda
temporarily dissolves the walls between polyhedra, letting the
optimizer reach arrangements no legal-state method can access. Result:
1.38 avg proxy, beats RePlAce by 5.1%.

**Act 3 — The second barrier.**
Multi-seed verification revealed that DPO converges to byte-identical
placements on ibm02/ibm12 across all 4 seeds. Within-DPO improvements
(seeds, congestion-only refinement, diverse priors) all capped at 1-2%.
A cell-by-cell analysis of RUDY congestion vs the real congestion grid
showed the gradient direction is structurally wrong: top-5% hotspot
overlap is only 10.9% (Jaccard 0.057, near-random). DPO's congestion
gradient points at the wrong cells. No amount of better DPO tuning can
overcome this.

**Second pivot — Full-proxy CD on an incremental evaluator.**
The natural response to "RUDY is wrong" is to build a better RUDY.
The correct response was to bypass RUDY entirely. We built an
incremental evaluator that maintains per-net min/max trackers, a
bin-density grid, and per-net RUDY congestion contributions, with
bit-for-bit parity to `compute_proxy_cost` at 4657× speedup. This
unlocked coordinate descent on the full proxy with breakpoint
enumeration. Each per-axis search considers 30-300 candidate positions
(net endpoints + bin lines) and picks the proxy-minimizing one.
Result: 1.12 avg proxy with a fixed 600s/benchmark budget, matching
the leaderboard within 0.18%.

**Resolution — Per-benchmark plateau detection (E9).**
CDOnly's fixed budget left hard benchmarks (ibm17/18/14/12) mid-
descent and wasted time on easy ones. CDAdaptive lets each benchmark
exit when its 3-sweep delta-window drops below 0.005, capped at 1hr.
Hard benchmarks gain 1.7-3.6% from the extra time the easy ones save;
all 17 exit via plateau, none hit the cap. **Final: 1.1055 avg —
beats the public leaderboard 1.1172 by -1.05%.** -24.2% vs RePlAce.

---

## Tone

Engineer who reads math and applies it well. Not a mathematician
proving theorems. The strength is the combination of:

- Structural observation applied to a concrete problem
- Rigorous empirical methodology (22-experiment sweep, correlation
  analysis, ablation studies, cell-by-cell RUDY/real comparison)
- Honest failure analysis at each pivot — the failures reveal
  something about the problem
- A resolution at each stage that follows from the diagnosis, not
  from trial and error
- Infrastructure unlocking algorithm: the 4657× incremental evaluator
  was the gate, not the algorithm itself (CD is textbook)

The math is a lens for understanding the problem, not an end in
itself. Every mathematical reference in the writeup is either something
we implemented, something we tested, or something that directly
explains an empirical observation.

---

## What the reader takes away

1. **Macro placement has clean mathematical structure** (union of
   polyhedra, LP within each) that is underexploited in practice.
   But the structure alone doesn't win — it tells you what to *avoid*.

2. **The objective mismatch is the binding constraint**, and it
   recurs at every level. LP-HPWL vs proxy (rho=-0.001); RUDY
   congestion vs real congestion (10.9% hotspot overlap); DPO's
   gradient vs the true gradient. Each approximation introduces a
   mismatch that caps performance. The same diagnostic methodology
   (systematic ablation + correlation analysis) detected each one.

3. **Decomposition is counterproductive** for this objective. The
   proxy cost couples wirelength, density, and congestion through
   shared macro positions. Every attempt to separate the problem
   loses information and produces worse results. Tested via
   22-experiment sweep, hierarchical clustering, swap-then-LP, and
   per-component ablation.

4. **"Bypass, don't fix"** as an algorithmic design principle.
   When an approximation is structurally wrong (not just noisy),
   exact evaluation with a faster data structure beats a more
   accurate approximation. We demonstrate this twice on the same
   problem (LP-HPWL → DPO; DPO RUDY → CD).

5. **Infrastructure unlocks algorithms.** CD is textbook. The
   4657× speedup from the incremental evaluator was the gate.
   Without it, CD at 30s/eval would do ~2 sweeps per hour; with
   it, 13+ sweeps in 10 min. Same algorithm, different infrastructure.

6. **Per-benchmark plateau detection** transfers to unseen designs
   without per-benchmark tuning. The 1-hour cap matches the
   competition rule. All 17 IBM benchmarks exit via plateau under
   default parameters; none hit the cap.

---

## What this is NOT

- Not a theory paper. We do not prove new theorems.
- Not a systems paper. The codebase is competition-grade, not a
  reusable framework.
- Not a survey. Mathematical connections are included only when they
  explain something we observed empirically.
- Not a leaderboard paper. The 24% over RePlAce matters less than
  the trajectory that got there — two diagnosis-pivot cycles, each
  driven by quantitative analysis.
