# Approach: Polyhedra Navigation

Last updated: 2026-04-15

---

## 1. The Decomposition Theorem

The feasible region of non-overlapping placements is a **union of convex polyhedra**. Each polyhedron = a fixed L/R/A/B pairwise assignment. Within any polyhedron, the problem is a **linear program**.

The problem decomposes into:
1. **Which polyhedron** (discrete, NP-hard) — the topology
2. **Where within it** (continuous, polynomial) — solved exactly via LP

### Why this is structurally better than RePlAce

| Standard pipeline (RePlAce) | Polyhedra navigation |
|---|---|
| Relaxes non-overlap -> smooth convex basin | Feasible region = union of convex cells |
| Legalization is one-shot, loses 10-15% | Every state is legal by construction |
| Gradient of relaxed objective (heuristic) | LP dual variables (exact marginal costs) |
| Continuous only | Discrete (topology) + continuous (LP) |

---

## 2. Architecture

### What's implemented

- SDF v5 as initial placement (good density structure)
- Assignment extraction + HPWL LP solve (HiGHS) + dual extraction
- GridSurrogate for fast (~0.1ms) candidate evaluation
- Surrogate-guided navigation with SA acceptance
- Cluster moves (net-correlated + macro-centered multi-pair flips)
- Robust projection (cascade repair + overlap repair)

---

## 3. Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | -- |
| **Polyhedra (50s nav)** | **1.4918** | **-2.3%** |
| **Polyhedra (300s nav)** | **1.4867** | **-2.0%** |
| SDF v5 (init only) | 1.5002 | -2.9% |

- Beat RePlAce on 3/17 benchmarks (ibm02, ibm10, ibm12)
- Congestion is **66.5%** of proxy cost, density 29.3%, WL 4.2%
- Navigation improves density effectively but **cannot move congestion**
- **Phase 5 sweep (22 experiments):** surrogate fixes, alternative inits, LP modifications — all within noise of baseline. The architecture is at a plateau for incremental changes.

---

## 4. Comparison vs Traditional Approaches

| Approach | Paradigm | Overlap Handling | Congestion Strategy | Weakness |
|---|---|---|---|---|
| **RePlAce** (target, 1.4578) | Nesterov on smooth relaxation | Continuous penalty -> one-shot legalize | Implicit via density spreading | Legalization loses 10-15% quality |
| **ePlace / DREAMPlace** | Same as RePlAce, GPU-accelerated | Same | Same | Same paradigm, just faster |
| **SA on B\*-tree / Seq. Pair** | Stochastic combinatorial search | Always feasible (compact repr.) | Evaluated but not optimized | Slow: O(N^2) eval per move, can't do 500+ macros in 60s |
| **Partitioning (Capo)** | Recursive bisection | Top-down slot assignment | Coarse: only at partition boundaries | Greedy, loses global structure |
| **MaskPlace (RL)** | Learned policy, sequential | Sequential feasible placement | Reward shaping | Generalization to unseen designs is poor |
| **Our approach** | LP inside combinatorial navigation | Always feasible by construction | Surrogate (RUDY) guides search | Congestion barrier; weak surrogate (rho=0.17 within-benchmark) |

### Where we're mathematically stronger than RePlAce

- No information loss from legalization (always feasible)
- LP duals give exact marginal costs (vs. heuristic gradients of relaxed problem)
- Decomposition is principled, not an approximation

### Where RePlAce is empirically stronger

- Its electrostatic spreading implicitly handles congestion through smooth density redistribution
- The continuous gradient can make *coordinated multi-macro moves* cheaply -- one gradient step moves ALL macros simultaneously
- We move 1-5 pairs per iteration; RePlAce moves all N macros per step

---

## 5. The Congestion Barrier

Six experiments established that local navigation cannot reach congestion:

| Experiment | Congestion delta | Density delta | Conclusion |
|------------|-----------------|---------------|------------|
| Real-proxy pair flips (300s) | <1% | -2.8% to -5.6% | Local flips only move density |
| Swap + LP | **-44%** | +180% | Different topologies CAN have better congestion |
| Swap only (no LP) | <0.5% | ~0% | Single swaps don't change routing structure |
| Group topology cascade | LP infeasible | N/A | Changing 200+ pairs creates constraint cycles |
| Sequence pair (1-500 steps) | +1% to +2% (worse) | +2% to +5% (worse) | Random distant topologies are worse |
| Position blending (SDF<->LP) | Monotonically worse until pure LP | -- | No smooth path in position space |

### Key insight: congestion-density tradeoff

The HPWL LP, when freed from the SDF topology, finds placements with 44% lower congestion but 180% higher density. Congestion and density are anti-correlated under unconstrained LP optimization. The SDF topology constrains the LP to find a good density-congestion compromise. But the current topology is stuck in a **local optimum** that all neighbor topologies (by any local move) make worse.

### The real problem

We're not searching inefficiently -- the SDF basin is genuinely a local minimum in all directions we can probe. We need to **tunnel through a barrier** to reach a distant basin with better congestion. This requires crossing topologies with higher cost (going uphill) before descending into a new basin.

---

## 6. Subproblem Decomposition

The proxy cost `f(p) = WL + 0.5*D + 0.5*C` creates **five separable subproblems**, each with different mathematical character:

### SP1: Initial Topology Selection

**Status: Decent but locked-in.** SDF gives a good density-respecting topology, but we've proven the SDF basin is a local minimum in all local directions. The initial topology determines ~95% of the final congestion.

**Math:** This is a combinatorial problem over 4^{O(N^2)} states. No known polynomial algorithm. Current approach: inherit from SDF.

### SP2: Wirelength (4.2% of cost) -- SOLVED

LP-optimal within any polyhedron. Not a bottleneck at all. Mathematically complete.

### SP3: Density (29.3% of cost) -- GOOD

Navigation successfully reduces density. Top-10% is a non-convex order statistic, but the surrogate handles it well enough for local improvement. Going from SDF's density to navigated density shows clear gains.

### SP4: Congestion (66.5% of cost) -- THE BOTTLENECK

This is where 100% of the remaining gap to RePlAce lives. Six experiments proved local moves can't reach it. The swap+LP experiment proved better congestion *exists* in other topologies (44% lower!) but at 180% density cost.

**Math insight:** Congestion is a function of net bounding box distribution across grid cells. It's controlled by the *relative spatial ordering* of net-connected macros -- i.e., the topology. But navigation only changes topology *locally* (1-5 pair flips), and congestion requires *global* topology restructuring.

### SP5: Feasibility Maintenance -- SOLVED

Cascade repair + overlap repair ensures zero overlaps. This is a clear advantage over relax-then-legalize approaches.

---

## 7. Innovation Opportunities

Ranked by potential impact x feasibility:

### A. Congestion-Aware LP Objective (HIGH impact, MEDIUM effort)

**The gap:** The LP minimizes only HPWL. But RUDY congestion is approximately linearizable. For each net j with bounding box area A_j, congestion demand ~ A_j / (grid cells in bbox). The sum of net bbox areas is:

```
sum_j (x_max_j - x_min_j) * (y_max_j - y_min_j)
```

This is bilinear -- not directly LP-compatible. But we could:
1. Add a penalty `lambda * sum_j (x_span_j + y_span_j)` to the LP (this IS linear -- it's just scaled HPWL)
2. Or add a **net-weighted HPWL** where high-congestion nets get higher weight
3. Or iteratively solve: after each LP, identify the top-5% congested cells, increase weights on nets passing through them, re-solve

This is essentially **Lagrangian relaxation** of the congestion constraint, a well-studied technique. RePlAce does something analogous via density bin potentials. We could do it within the LP framework.

**Why this is differentiated:** Nobody else has an LP solver inside a feasible-topology framework. Adding congestion awareness to the LP lets us optimize congestion *within* the polyhedron, not just across polyhedra.

### B. Fix the Surrogate (HIGH impact, MEDIUM effort)

Within-benchmark Spearman rho = 0.17 means **the navigator is nearly blind**. Of 1000+ candidates evaluated per benchmark, we're essentially picking randomly among them. Improving this to rho > 0.5 could double effective navigation quality.

**Options:**
1. **Calibrate surrogate against real proxy** -- fit a per-benchmark linear correction (a,b) so `real ~ a*surrogate + b`, using the first few verified candidates
2. **Delta-based surrogate** -- instead of absolute cost, predict cost *change* from a verified baseline. Relative predictions are often more stable
3. **Use real proxy selectively** -- for the top-5 candidates, compute real proxy (we already do top-3 verify). Could also use real proxy to train the surrogate online

**Why this is differentiated:** Most competitors either use expensive real evaluation or hand-tuned surrogates. An online-calibrated surrogate that gets better during search is novel.

### C. Hierarchical Topology Decomposition (HIGH impact, HIGH effort)

**The gap:** Flat O(N^2) pair flips can't make coordinated global changes. Group topology (changing 200+ pairs at once) creates LP infeasibilities. We need something in between.

**Approach:** Build a **macro hierarchy** via recursive partitioning of the netlist (e.g., hMETIS). At each level:
1. **Coarse level:** Treat macro clusters as single super-macros. The assignment between super-macros has O(K^2) pairs (K clusters << N). Large topology changes at this level are feasible.
2. **Fine level:** Within each cluster, run current navigation.
3. **Refinement:** Alternately optimize at coarse and fine levels.

This is the **multilevel paradigm** from partitioning (Karypis/Kumar), adapted to the polyhedra framework. It lets us make large global topology changes (at coarse level) without creating constraint cycles.

**Why this is differentiated:** Multilevel is used in partitioning and SA, but NOT inside an LP-based polyhedra framework. The combination is novel.

### D. Soft-Mode Tunneling (MEDIUM impact, already planned)

The tunneling theory (Phase 4 in the roadmap) targets exactly the right thing: crossing the congestion barrier via soft-pair flips. The math (mountain pass, Potts model, complexification) is elegant.

**Risk assessment:** The theory correctly identifies that *soft pairs* are the degrees of freedom congestion might live in. But the empirical question -- "does flipping 5-20 soft pairs actually move congestion?" -- is still open. If the answer is no (barrier is 500 flips wide, not 5), the theory is right but impractical.

**Suggestion:** Test this immediately. It's 1 day of work and resolves the biggest uncertainty.

### E. Congestion-First Initialization (MEDIUM impact, LOW effort)

Instead of SDF -> navigate, try:
1. Solve an HPWL LP with **no initial topology** (use a random or congestion-minimizing assignment)
2. Use the LP positions as init
3. Then navigate for density

Currently we optimize: good density init -> navigate for density. Try: good congestion init -> navigate for density. The swap+LP experiment showed that LP-solved positions from different topologies CAN have 44% less congestion.

**How to get a congestion-minimizing topology:**
- Start from the SDF positions
- Solve LP
- Identify top-congested nets
- For pairs within those nets, try ALL 4 directions, pick the one that minimizes the net's bounding box area
- Re-solve LP

This is a greedy congestion-aware topology construction, not navigation. It builds a NEW polyhedron from scratch rather than walking between neighbors.

### F. Sequence Pair as Topology Representation (MEDIUM impact, HIGH effort)

The current topology is an explicit O(N^2) pair assignment. A **sequence pair** (SP) encodes the same information in O(N) space -- two permutations of N macros. Benefits:
- SP perturbations (adjacent transpositions) are guaranteed to produce feasible topologies
- The SP naturally avoids the infeasibility issues from group topology changes
- Well-studied in floorplanning literature (Murata et al., 1996)

**Risk:** We already tried sequence pair transpositions (3205/3250 feasible, 0 improvements). But those were *random* transpositions. SP + LP solve + dual-guided selection of which transpositions to make could work.

---

## See Also

- [problem.md](problem.md) -- formal mathematical formulation
- [theory.md](theory.md) -- tunneling frameworks and theoretical backing
- [results.md](results.md) -- experiment history and per-benchmark data
- [roadmap.md](roadmap.md) -- action plan
