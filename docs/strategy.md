# Strategy

Last updated: 2026-04-05

---

## 1. The Problem

Place 200-537 rectangular macros on a 2D chip canvas to minimize proxy cost (wirelength + 0.5×density + 0.5×congestion) with zero overlaps. Beat RePlAce baseline (1.4578 avg on 17 IBM benchmarks). Deadline: May 21, 2026.

## 2. Our Approach: Polyhedra Navigation

### The decomposition theorem

The feasible region of non-overlapping placements is a **union of convex polyhedra**. Each polyhedron = a fixed L/R/A/B pairwise assignment. Within any polyhedron, the problem is a **linear program**.

The problem decomposes into:
1. **Which polyhedron** (discrete, NP-hard) — the topology
2. **Where within it** (continuous, polynomial) — solved exactly via LP

### Why this is structurally better than RePlAce

| Standard pipeline (RePlAce) | Polyhedra navigation |
|---|---|
| Relaxes non-overlap → smooth convex basin | Feasible region = union of convex cells |
| Legalization is one-shot, loses 10-15% | Every state is legal by construction |
| Gradient of relaxed objective (heuristic) | LP dual variables (exact marginal costs) |
| Continuous only | Discrete (topology) + continuous (LP) |

### What's implemented

- SDF v5 as initial placement (good density structure)
- Assignment extraction + HPWL LP solve (HiGHS) + dual extraction
- GridSurrogate for fast (~0.1ms) candidate evaluation
- Surrogate-guided navigation with SA acceptance
- Cluster moves (net-correlated + macro-centered multi-pair flips)
- Robust projection (cascade repair + overlap repair)

## 3. Current Position

| Entry | Avg Proxy | Gap to RePlAce |
|-------|-----------|----------------|
| RePlAce (target) | 1.4578 | — |
| **Polyhedra (50s nav)** | **1.4921** | **-2.4%** |
| **Polyhedra (300s nav)** | **1.4890** | **-2.1%** |
| SDF v5 (init only) | 1.5002 | -2.9% |

- Beat RePlAce on 3/17 benchmarks (ibm02, ibm10, ibm12)
- Congestion is **66.5%** of proxy cost, density 29.3%, WL 4.2%
- Navigation improves density effectively but **cannot move congestion**

## 4. What We've Proven (Apr 5 experiments)

### The congestion barrier

Six experiments established that local navigation cannot reach congestion:

| Experiment | Congestion Δ | Density Δ | Conclusion |
|------------|-------------|-----------|------------|
| Real-proxy pair flips (300s) | <1% | -2.8% to -5.6% | Local flips only move density |
| Swap + LP | **-44%** | +180% | Different topologies CAN have better congestion |
| Swap only (no LP) | <0.5% | ~0% | Single swaps don't change routing structure |
| Group topology cascade | LP infeasible | N/A | Changing 200+ pairs creates constraint cycles |
| Sequence pair (1-500 steps) | +1% to +2% (worse) | +2% to +5% (worse) | Random distant topologies are worse |
| Position blending (SDF↔LP) | Monotonically worse until pure LP | — | No smooth path in position space |

### Key insight: congestion-density tradeoff

The HPWL LP, when freed from the SDF topology, finds placements with 44% lower congestion but 180% higher density. Congestion and density are anti-correlated under unconstrained LP optimization. The SDF topology constrains the LP to find a good density-congestion compromise. But the current topology is stuck in a **local optimum** that all neighbor topologies (by any local move) make worse.

### The real problem

We're not searching inefficiently — the SDF basin is genuinely a local minimum in all directions we can probe. We need to **tunnel through a barrier** to reach a distant basin with better congestion. This requires crossing topologies with higher cost (going uphill) before descending into a new basin.

## 5. Tunneling Theory

Three mathematical frameworks converge on the same algorithmic insight. Full details in [`tunneling-theory.md`](tunneling-theory.md).

### Framework 1: Complexification

In C^{2N}, the disconnected polyhedra become path-connected. Paths exist between any two topologies through constraint space (not position space). The practical realization is **parametric LP**: continuously rotate a constraint from direction A to direction B, tracking the LP solution as it crosses polyhedron boundaries.

### Framework 2: Least Action / Mountain Pass

Between two basins, the **mountain pass theorem** guarantees a saddle point — the lowest barrier. The optimal tunneling path crosses this saddle. The saddle corresponds to topologies where specific pairs have minimal cost to flip (the "softest modes").

### Framework 3: Quantum Tunneling / Potts Model

The problem maps exactly to a **4-state Potts spin system**. The Feynman path integral shows tunneling follows the softest modes. The cavity method (belief propagation) can identify which spins are "frozen" (locked) vs "soft" (free to change). Only soft spins should be flipped for tunneling.

### The unified insight

**We've been flipping the wrong pairs.** Current navigation flips pairs with LARGE duals (most expensive for wirelength). These are "frozen" spins — the LP needs them. Tunneling requires flipping pairs with SMALL duals — the "soft" modes that wirelength doesn't care about. Congestion may live in these degrees of freedom.

## 6. What We Don't Know Yet

1. **Are there soft pairs whose flipping moves congestion?** The LP has many zero-dual constraints (slack). Do any of them control congestion?
2. **How wide is the barrier?** If it's 5 soft-pair flips, we can cross it. If it's 500, we can't.
3. **Can parametric LP trace a path through the barrier?** The theory says yes, but it may be computationally impractical for N=500+ macros.
4. **Does cavity method / BP converge on this graph?** Short loops in the constraint graph may break the locally-tree-like assumption.

---

## See also

- [`tunneling-theory.md`](tunneling-theory.md) — detailed theoretical frameworks
- [`roadmap.md`](roadmap.md) — action plan
- [`experiment-log.md`](experiment-log.md) — full experiment history
- [`formal-problem-statement.md`](formal-problem-statement.md) — decomposition theorem, LP formulation
- [`novel-mathematical-machinery.pdf`](novel-mathematical-machinery.pdf) — innovation prize frameworks
