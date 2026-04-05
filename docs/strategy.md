# Macro Placement Challenge 2026 — Strategy

Last updated: 2026-04-04

**Goal:** Beat RePlAce baseline (1.4578 avg proxy cost) on 17 IBM benchmarks.
**Deadline:** May 21, 2026 (~7 weeks remaining).
**Prize:** $49K+. Innovation prize ($4K) rewards novel approaches.
**Hardware:** AMD EPYC 9655P (16 cores) + RTX 6000 Ada 48GB. 1-hour timeout per benchmark.

---

## 1. Where We Stand

| Entry | Avg Proxy | Gap to RePlAce | Notes |
|-------|-----------|----------------|-------|
| RePlAce (target) | 1.4578 | — | Analytical placer baseline |
| **Polyhedra Navigation (our best)** | **1.4867** | **-2.0%** | Graduated; search not plateauing |
| SDF v5 | 1.5002 | -2.9% | Stalled; used as init for polyhedra |
| Will's SA seed | 1.5338 | -5.2% | Previous best submission |

### The critical discovery (31 experiments)

> Across SDF (10 variants) and Optimal Transport (12 variants), **every continuous
> optimization loop is inert.** The final placement equals `greedy_legalize(initial_positions)`
> exactly. The greedy legalizer does 100% of the work — and degrades proxy cost 10-15%
> from the (overlapping) initial positions.

This killed 4 of our 6 original hypotheses (SDF, OT, variational, neural ODE). All share
the same paradigm: continuous optimization → overlap relaxation → one-shot legalization.
The optimizer contributes nothing; the legalizer irreversibly commits to a suboptimal arrangement.

**Polyhedra navigation is the only approach that avoids this trap** — it stays in feasible
(non-overlapping) space throughout, making the legalization bottleneck structurally impossible.

---

## 2. Leading Hypothesis: Polyhedra Navigation

### 2.1 The decomposition theorem

The feasible region of non-overlapping placements is a **union of convex polyhedra**. Each
polyhedron corresponds to a fixed assignment of pairwise spatial relations (for each macro pair
(i,k): is i left/right/above/below k?). Within any single polyhedron, the placement problem
is a **linear program** — solvable exactly in polynomial time.

The full problem decomposes into:
1. **Navigation** (NP-hard): choose which polyhedron (= which assignment of L/R/A/B relations)
2. **Evaluation** (polynomial): solve the LP within that polyhedron for exact optimal positions

This is fundamentally different from all standard EDA approaches. No existing work explicitly
decomposes placement into "which polyhedron" and "where within it."

### 2.2 Why this approach has structural advantages

| Standard pipeline (RePlAce, DREAMPlace) | Polyhedra navigation |
|---|---|
| Relaxes non-overlap → one smooth convex basin | Feasible region = union of exponentially many convex cells |
| Legalization is one-shot, irreversible, loses 10-15% | Every visited state is legal by construction |
| Gradient of relaxed objective (heuristic) | LP dual variables (exact marginal cost of each constraint) |
| Continuous optimization only | Discrete (topology) + continuous (LP), formally decomposed |
| Fast search, lossy legalization | Slower search, zero legalization tax |

**The bet:** their legalization tax (10-15%) exceeds our search inefficiency. If we can
search effectively, we win on solution quality by construction.

### 2.3 Diagnosis: why search is stalled

Three compounding bottlenecks:

| Bottleneck | Evidence | Impact |
|---|---|---|
| Search budget (~20 candidates/benchmark) | `compute_proxy_cost` ~1s; 30s nav budget | Can't explore the discrete space |
| Single-pair flips too local | 0.1% improvement after 5 experiments | Trapped in SDF's topology basin |
| LP optimizes HPWL only | 30% WL slack unusable due to density | Duals guide toward WL, not proxy cost |

**The implementation is operating exclusively at the most expensive evaluation level.**
This is the equivalent of pointing a telescope at individual planets without first checking
whether their star could support life.

---

## 3. The Fix: Hierarchical Multi-Fidelity Search

### 3.1 Core insight

Evaluate **neighborhoods of polyhedra** cheaply before evaluating individual polyhedra
expensively. A neighborhood = a set of topologies sharing the same coarse structure (e.g.,
"group A is left of group B") but differing in fine-grained pair assignments. If the coarse
structure is bad, every topology in that neighborhood is bad — skip them all instantly.

### 3.2 The multi-fidelity cascade

```
Level 0: Group topology sweep    →  "Which coarse arrangements are geometrically feasible?"
Level 1: WL/net-cut screening    →  "Does this arrangement have acceptable wirelength bound?"
Level 2: Surrogate evaluation    →  "Does this fine-grained assignment have good estimated proxy?"
Level 3: LP solve                →  "What are the optimal positions within this topology?"
Level 4: Full proxy evaluation   →  "Confirm: does this beat our current best?"
```

Each level is 100-1000x more expensive than the previous. The filtration ratio at each
level eliminates the vast majority of candidates before they reach an expensive evaluation.

### 3.3 Filtration math

```
Level 0: 100,000 group topologies screened      → 1,000 pass geometric filter    (~0 cost)
Level 1: 1,000 screened by WL/net-cut           → 100 pass                       (~1-10μs each)
Level 2: 100 expanded + surrogate evaluated     → 10 pass                        (~0.1-1ms each)
Level 3: 10 LP solves                           → 5 promising                    (~1-10ms each)
Level 4: 5 full proxy evaluations               → 1 accepted                     (~1s each)

Total: 100,000 neighborhoods explored, 5 expensive evaluations.
Time:  ~100K × 1μs + 1K × 10μs + 100 × 1ms + 10 × 10ms + 5 × 1s ≈ 5.2s per round
```

Compare to today: 20 expensive evaluations, 0 cheap screenings, ~20s per round.

In 30 seconds we complete ~5 rounds = **500,000 group topologies surveyed** vs 20 today.

### 3.4 What's implemented vs what's needed

| Level | Component | Cost | Status |
|-------|-----------|------|--------|
| **0: Group topology** | `HierarchicalGrouper` — spectral clustering on netlist, geometric filter | ~0 (μs) | **Not implemented** |
| **1: WL/net-cut screen** | `CoarseEnumerator` — HPWL lower bound + net cut count | ~1-10μs | **Not implemented** |
| **2: Surrogate** | `GridSurrogate` — incremental density grid + RUDY congestion | ~0.1-1ms | **Not implemented** (another Claude instance may be building this) |
| **3: LP solve** | `LPSolver` — HPWL LP via HiGHS + dual extraction | ~1-10ms | **Implemented** (used once per benchmark currently) |
| **4: Full proxy** | `Navigator._eval_proxy` — compute_proxy_cost() | ~1s | **Implemented** (this is ALL we do today) |

### 3.5 How the search loop changes

**Current (flat, single-pair):**
```
init from SDF → solve LP once → get duals
repeat: rank pairs by |dual| → flip → project → eval_proxy(~1s) → accept/reject
(20 candidates/benchmark, ~0.1% improvement)
```

**Target (hierarchical cascade):**
```
init from SDF → spectral cluster into k groups → solve LP → get duals

repeat:
    Level 0: enumerate N group topologies, filter by geometry     → ~N/10
    Level 1: compute WL bound + net cut for survivors             → ~N/100
    Level 2: expand to macro-level, evaluate with surrogate       → ~N/1000
    Level 3: LP solve for top candidates                          → positions + duals
    Level 4: eval_proxy on LP-optimized positions                 → accept best
```

### 3.6 New modules needed

| Module | Level | What it does |
|--------|-------|-------------|
| `HierarchicalGrouper` | 0 | Spectral clustering on netlist hypergraph → k groups (8-16). Adjacency matrix A[i,j] = shared nets; normalized Laplacian; bottom-k eigenvectors; k-means. Output: group assignments + bounding boxes. |
| `CoarseEnumerator` | 0-1 | Enumerate group-level L/R/A/B assignments. Filter 0: geometric feasibility (can groups tile canvas?). Filter 1: WL lower bound from group separations. Filter 2: net cut count. |
| `TopologyExpander` | 1→2 | Expand group topology into macro-level assignment. Keep intra-group pairs from SDF; re-assign only inter-group pairs. For k=10 groups: 45 group pairs → 500-2000 macro pairs change, rest stay fixed. |
| `GridSurrogate` | 2 | Incremental density grid (32x32 or 64x64). RUDY-style congestion on grid edges. HPWL from net bounding boxes. Composite: `wl + 0.5*density + 0.5*congestion`. Target: <1ms, rank correlation >0.7 with true proxy. |

### 3.7 Key insight: most of the polyhedron stays fixed

When we change a group topology, only inter-group pair assignments change. For k=10 groups,
that's ~45 group pairs affecting ~500-2000 macro pairs out of 30,000+. The vast majority of
the polyhedron is unchanged. This is what makes the hierarchy efficient — coarse moves change
the global structure while preserving good local structure from SDF.

---

## 4. Theoretical Extensions (contingency + innovation prize)

These ideas extend the polyhedra framework. They're ranked by actionability — Tier 1 could
be implemented within the competition timeline if the core cascade works.

### Tier 1: Directly actionable

- **Density-aware candidate ranking**: Blend LP duals (wirelength signal) with surrogate
  congestion/density deltas. `score = α|dual| + β·Δcongestion + γ·Δdensity`. Requires
  GridSurrogate. Could replace pure dual-magnitude ranking in the cascade.

- **Tropical geometry**: HPWL is a tropical polynomial in the (max,+) semiring. Tropical
  gradient descent (Talbut & Monod, 2024) achieves classical convergence on tropically convex
  problems. Could replace log-sum-exp smoothing in LP objective.

- **Disjunctive programming (Balas)**: P-split hierarchy (Kronqvist et al., 2025) provides
  tighter LP relaxations than Big-M. "Full optimal big-M lifting" is exact for axis-aligned
  rectangles in 2D.

- **Fisher-Rao natural gradient**: Fisher-Rao gradient flows achieve linear convergence for
  LPs. Natural gradient on assignment distributions has block-diagonal Fisher matrix —
  computationally trivial. Could replace greedy search at Level 1.

### Tier 2: Structural insights

- **Cavity method / survey propagation**: Maps to 4-state Potts model. Predicts phase
  transitions in solution space. At 43-53% area utilization (our benchmarks), likely near
  clustering transition — explains why local search traps.

- **Lifting / complexification**: Adding a synthetic third dimension during search makes
  separation constraints easier to satisfy (more room to maneuver). Gradually anneal z→0
  to recover 2D placement. Directly mitigates low feasibility rates between polyhedra.
  **Contingency if hierarchical cascade stalls due to <1% group-level feasibility.**

- **Stratified Morse theory**: Feasible region is stratified. Morse inequalities bound local
  minima count by Betti numbers. Persistent homology reveals basin structure.

### Innovation prize narrative

The polyhedra navigation framework is absent from EDA literature. The composition of:
Benders decomposition + LP duals as discrete gradients + multi-fidelity cascade + spectral
grouping + tropical geometry connections gives a rich theoretical story regardless of score.

---

## See also

- [`roadmap.md`](roadmap.md) — phased action plan with verifiable goals
- [`experiment-log.md`](experiment-log.md) — full experiment history and per-benchmark results
- [`profiling-plan.md`](profiling-plan.md) — measurement methodology
- [`formal-problem-statement.md`](formal-problem-statement.md) — decomposition theorem, LP formulation
- [`polyhedra-navigation.pdf`](polyhedra-navigation.pdf) — primary implementation reference
- [`novel-mathematical-machinery.pdf`](novel-mathematical-machinery.pdf) — innovation prize frameworks
