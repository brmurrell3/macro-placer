# Approach: Polyhedra Navigation

Last updated: 2026-04-23

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
- ClusterScreener: multi-tier pre-projection pruning (Zobrist dedup, displacement floor, HPWL bound, axis crowding) — rejects ~20-40% of moves at ~1-10μs each before expensive projection+surrogate
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

### A. Congestion-Aware LP Objective --- KILLED

Tested as SP4 (6 variants: net weighting, separation margins, McCormick area, dual-informed targeting, real-proxy feedback, LP-navigate-reweight). Best result: McCormick area at 1.4918 (noise). LP-level congestion modifications are washed out by 50s of navigation — the navigator re-solves LP each iteration, erasing starting conditions.

### B. Fix the Surrogate --- KILLED

Tested as SP3 (8 variants: online calibration, delta ranking, top-k verify, rank aggregation, PinRUDY, pairwise ranking). Best result: top_k_20 at 1.4919 (noise). Real bottleneck is not ranking quality — only 1-7 candidates are genuinely better per benchmark run. The search space is nearly exhausted at current navigation scale.

### C. Hierarchical Topology Decomposition (UNTESTED — most promising remaining)

Flat O(N^2) pair flips can't make coordinated global changes. Group topology (200+ pairs) creates LP infeasibilities. Multilevel navigation clusters macros into K=15-25 super-macros, navigates at coarse level (190 pairs, exhaustive search possible), then refines within clusters. This is the only approach that could break the congestion barrier by making large topology jumps feasible. Novel combination of multilevel paradigm with LP-based polyhedra framework.

### D. Soft-Mode Tunneling --- KILLED

Tested in Phase 4. Soft-pair flips, group cascade, sequence pairs all failed to cross the congestion barrier. The barrier is hundreds of flips wide, not 5-20.

### E. Congestion-First Initialization --- KILLED

Tested as SP1 (6 variants: spectral, RePlAce extraction, congestion-aware extraction, hMETIS, greedy, boundary attraction). All alternative inits either produce terrible density or map to the same topology after extract_assignment(). SDF's analytical spreading is genuinely hard to beat.

### F. Sequence Pair as Topology Representation --- KILLED

Tested: 3205/3250 transpositions LP-feasible, 14 overlap-free, 0 improvements. Multi-step (10-500 transpositions) all worse on congestion.

### G. Cheap-Signal Cluster Screening --- KILLED

Tested as Miftari experiments. Cheap LP dual signals predict HPWL changes (rho=0.86, 42700x speedup) but LP-HPWL doesn't predict proxy (rho=-0.001). HPWL and density anti-correlate; congestion dominates proxy and is uncorrelated with HPWL. Any HPWL-based topology ranking is blind to the objective.

### H. Incremental Real-Proxy Evaluator (PROPOSED)

Replace GridSurrogate (within-benchmark ρ=0.17) with an incremental evaluator that returns exact proxy cost by caching per-net HPWL, per-cell density, and per-cell congestion accumulators and updating only what a move touches. Motivated by Vedu Mallela's Partcl submission ("Incremental CD", 300× speedup on real objective).

**What it solves:** Eliminates surrogate ranking error entirely. If exact signal still finds only 1-7 improving moves per benchmark, the diagnosis shifts definitively from "bad signal" to "bad move set" — removing a confound for all future experiments.

**What it doesn't solve:** The congestion barrier (§5). Local moves still can't reach distant congestion basins regardless of eval accuracy. But faster exact eval enables more candidates per iteration, cheaper restarts, and tractable tunneling experiments.

**Risk:** Matching `compute_proxy_cost` bit-for-bit requires reimplementing PlacementCost C++ internals (density grid, RUDY, top-k order statistics) in Python. Order statistics (top-10% density, top-5% congestion) don't decompose incrementally — need full reduce after each move (~10K cells, likely fine). SP3 overnight results suggest surrogate accuracy isn't the binding constraint, so the payoff may be diagnostic rather than score-improving.

**Effort:** 3-5 days. See [evaluation.md](evaluation.md) for full design.

---

## See Also

- [problem.md](problem.md) -- formal mathematical formulation
- [theory.md](theory.md) -- tunneling frameworks and theoretical backing
- [results.md](results.md) -- experiment history and per-benchmark data
- [roadmap.md](roadmap.md) -- action plan
- [evaluation.md](evaluation.md) -- navigator eval pipeline (screener + proposed incremental evaluator)
