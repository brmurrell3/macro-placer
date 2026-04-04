# Macro Placement Challenge 2026 — Strategy

Last updated: 2026-04-04

**Goal:** Beat RePlAce baseline (1.4578 avg proxy cost) on 17 IBM benchmarks.
**Deadline:** May 21, 2026 (~7 weeks remaining).
**Prize:** $49K+. Innovation prize rewards novel approaches.
**Hardware:** AMD EPYC 9655P (16 cores) + RTX 6000 Ada 48GB. 1-hour timeout per benchmark.

---

## 1. Where We Stand

### Current scores

| Entry | Avg Proxy | Notes |
|-------|-----------|-------|
| RePlAce (target) | 1.4578 | Analytical placer baseline, dominates all benchmarks |
| **SDF v5 (our best)** | **1.5002** | 0.0002 above graduate threshold, 2.9% behind RePlAce |
| Will's SA seed | 1.5338 | Previous best submission |
| SA baseline | 2.1362 | Weak |

### What we tried

**SDF Density (10 variants):** Best result 1.5002. WL is excellent (0.05-0.08), congestion is the bottleneck (1.3-2.5 range, weight 0.5). Tried: two-phase, force-directed init, SA post-processing, soft macro opt, density variance. None improved beyond v5.

**Optimal Transport (12 variants):** All results equal the legalize-only baseline (1.2775 fast). The entire gradient descent optimization loop is **inert** — legalization does all the work. OT spreading actively hurts when forced (-8 to -12%). Macros are treated as points, destroying netlist-aware locality.

### The critical discovery

> The optimization loop in all our placers has zero effect. The final placement equals `greedy_legalize(initial_positions)`. The greedy legalizer degrades proxy cost 10-15% from (overlapping) initial positions. **The path to improvement is better legalization, not better continuous optimization.**

This is the single most important finding from 22 experiments. It means:
- Gradient-based approaches (SDF, OT, diffusion, variational, neural ODE) that produce overlapping solutions requiring legalization all hit this same wall
- The legalizer is a one-shot, irreversible commitment to a specific non-overlapping arrangement
- Any approach that stays in feasible (non-overlapping) space throughout avoids this problem entirely

---

## 2. The Big Idea: Polyhedra Navigation

The formal problem structure gives us a fundamentally different approach. This is the most developed idea and the strongest candidate for the competition.

### The decomposition theorem

Given a fixed assignment of all pairwise relations (for each macro pair (i,k): is i left/right/above/below k?), the remaining problem is a **linear program** — solvable exactly in polynomial time.

The full placement problem decomposes into:
1. **Navigation** (NP-hard): choose which polyhedron (= which assignment of L/R/A/B relations)
2. **Evaluation** (polynomial): solve the LP within that polyhedron

### Three-level architecture

```
Level 1 (Coarse Topology):   min_{tau}     F(tau)           [search over group-pair relations]
Level 2 (Fine Assignment):   min_{sigma}   f*(sigma)         [given tau, optimize intra-group pairs]
Level 3 (Continuous Solve):  min_{p}       f(p)             [given sigma, solve LP + refine]
```

- **Level 3** is an LP (~30K variables, ~90K constraints for ibm01). Solvable in 1-10ms. Warm-starting after a single pair flip: sub-millisecond.
- **Level 2** is local search with dual-variable-guided proposals. LP duals tell you which constraints are most expensive to flip.
- **Level 1** uses macro grouping (spectral clustering on netlist) to reduce the search space from 4^{30,000} to 4^{45} ~ 10^{27}.

### Why this is different from everything else

| Standard pipeline | Polyhedra navigation |
|---|---|
| Relaxes non-overlap (overlapping solutions allowed) | Always in feasible space (zero overlaps by construction) |
| Legalization is one-shot, irreversible | Polyhedron choice is revisable at any time |
| Gradient of relaxed objective (heuristic) | LP dual variables (exact gradient information) |
| Continuous optimization only | Discrete + continuous, formally decomposed |

### Key unknowns (must answer empirically)

1. **Continuous slack magnitude**: How much better is LP-optimal HPWL within RePlAce's polyhedron vs. RePlAce's actual HPWL? If large → LP alone helps. If small → navigation is where the wins are.
2. **Feasibility of neighbors**: What fraction of single-flip neighbors are feasible? If >10% → local search works. If <0.1% → need non-local moves.
3. **Density/congestion under LP**: Does LP-optimal placement produce reasonable density? Or pathological clustering requiring heavy post-refinement?
4. **LP warm-start speedup**: What's the actual cold-start vs. warm-start ratio on these benchmarks?

### Implementation roadmap

| Phase | Week | What | Key deliverable |
|-------|------|------|-----------------|
| 1: Proof of concept | 1 | Modules 1+2: extract assignment from RePlAce, solve LP | Continuous slack measurement |
| 2: Flat navigation | 2 | Modules 3+4: neighbor generation, greedy descent | Cost-vs-iteration curves |
| 3: Hierarchical | 3 | Module 5: spectral grouping, coarse search, pruning cascade | Flat vs. hierarchical comparison |
| 4: Tuning | 4 | All 17 benchmarks, calibrate schedules, handle edge cases | Final submission |

### The five modules

1. **Assignment Extractor**: Legal placement -> dict mapping each pair (i,k) to {L,R,A,B}
2. **LP Solver**: Assignment sigma -> optimal positions + dual variables (HiGHS)
3. **Neighbor Generator**: Current sigma + duals -> ranked candidate flips. P(flip pair (i,k)) proportional to |d_{ik}|^alpha
4. **Navigator**: Search loop — propose, evaluate (warm-start LP), accept/reject (Metropolis)
5. **Hierarchical Grouper**: Spectral clustering -> macro groups -> coarse/fine decomposition

### Pruning cascade (for Level 1)

Each coarse assignment tau passes through increasingly expensive filters:
1. Geometric feasibility (~0 cost): can groups physically fit?
2. Wirelength lower bound (~1us): min possible WL given group separations
3. Net cut cost (~10us): how many nets cross group boundaries?
4. Area packing feasibility (~100us): can macros fit in implied regions?
5. Relaxed LP bound (~1ms): solve tiny LP with groups as super-macros

Target: 1000:1 filtration ratio. Most candidates die at filters 1-3.

---

## 3. Hypothesis Status

### Tried and characterized

| Hypothesis | Status | Verdict |
|---|---|---|
| SDF Density | **stalled at 1.5002** | WL is good, congestion is the bottleneck. Optimizer is inert — all results = legalize(init). No clear path to break 1.50. |
| Optimal Transport | **unlikely** | OT spreading hurts when activated. Treats macros as points. Same inert-optimizer problem. |

### Not yet tried (from original 6)

| Hypothesis | Risk | Notes |
|---|---|---|
| Diffusion generative | Medium | Could be good for diverse initialization. 2-3 week implementation. Post-hoc legalization still needed (same wall). |
| Variational / least-action | High | Elegant theory. JKO scheme + OT. Very research-y. Same continuous-optimization paradigm. |
| Neural ODE adjoint | High | Requires differentiable pipeline. Adjoint through 1000+ steps may be unstable. |
| CBO (consensus swarm) | Medium-low | Simple to implement (2-3 days). Dimension-independent convergence in theory. But non-overlap constraint has no natural encoding. |

### The paradigm problem

Four of the six original hypotheses (SDF, OT, variational, neural ODE) share the same paradigm: continuous optimization with overlap relaxation, followed by legalization. **Our experiments show the optimizer does nothing and legalization does all the work.** These approaches all hit the same wall.

Only two escape this: **diffusion** (generates complete solutions, but still needs legalization) and **CBO** (swarm, but can't encode non-overlap). Neither is clearly better than the polyhedra approach.

---

## 4. Theoretical Landscape (for Innovation Prize)

The "Novel Mathematical Machinery" document maps the placement problem to 7 deep mathematical frameworks. These are ranked by actionability:

### Tier 1: Directly actionable

- **Tropical geometry**: HPWL is literally a tropical polynomial in the (max,+) semiring. Tropical gradient descent (Talbut & Monod, 2024) achieves classical convergence on tropically convex problems. Could replace log-sum-exp smoothing in Level 3.
- **Disjunctive programming (Balas)**: P-split hierarchy (Kronqvist et al., 2025) gives a principled middle ground between weak Big-M and expensive convex hull. "Full optimal big-M lifting" is exact for axis-aligned rectangles in 2D. Could tighten LP relaxations.
- **Information geometry / Fisher-Rao**: Fisher-Rao gradient flows achieve linear convergence for LPs. Natural gradient on assignment distributions (block-diagonal Fisher matrix) is computationally trivial. Could replace SA at Level 1.

### Tier 2: Structural insights

- **Cavity method / survey propagation**: Maps to 4-state Potts model. Predicts phase transitions in solution space. At 43-53% area utilization, benchmarks likely sit near the clustering transition — explaining why local search gets trapped.
- **Stratified Morse theory**: Feasible region is a stratified space. Morse inequalities bound local minima count by Betti numbers. Persistent homology reveals basin structure.
- **Complexification**: Extending to C^{2N} makes the feasible region path-connected (complex hyperplane complement is always connected). Homotopy continuation can trace paths between optima.

### Tier 3: Theoretical but impractical for competition

- **Random matrix theory**: Spectral analysis of constraint interaction matrices
- **Kolmogorov complexity sampling**: Bias toward compressible assignments (DeepMind, Feb 2026)

### The narrative for Innovation Prize

The polyhedra navigation framework is itself novel — no existing work explicitly decomposes placement into "which polyhedron" (NP-hard) and "where within it" (LP-solvable). The composition of LP solving + dual-guided MCMC + hierarchical grouping + pruning cascade is new. Connecting it to Benders decomposition, tropical geometry, and the cavity method gives a rich theoretical story.

---

## 5. Recommended Priority

### Primary path: Polyhedra Navigation
- Highest theoretical upside
- Avoids the inert-optimizer trap entirely (always in feasible space)
- Modular (each week's work strictly improves on the last)
- Strong innovation prize case
- 4-week implementation plan with clear milestones

### Secondary: SDF v5 as fallback
- Already at 1.5002 (beats Will's seed by 2.2%)
- Could be submitted as-is while developing polyhedra approach
- Congestion reduction is the only lever left — unclear how to pull it

### Not recommended
- Returning to OT, variational, neural ODE (all hit the same wall)
- Diffusion (2-3 weeks, still needs legalization, unlikely to beat polyhedra)
- CBO (can't encode constraints, would be a curiosity)

---

## 6. Reference Documents

| Document | What it contains | Status |
|---|---|---|
| `formal-problem-statement.md` | Precise math formulation, decomposition theorem |
| `hypothesis-status.md` | Per-variant experiment tracking (live) |
| `novel-mathematical-machinery.pdf` | 7 deep math frameworks — tropical, spin glass, etc. (innovation prize) |
| `polyhedra-navigation.pdf` | Full polyhedra navigation writeup — **primary implementation reference** |
| `rl-assessment.pdf` | Debunks RL for placement (Cheng, Kahng et al.) |
| `archive/hierarchical-polyhedra-framework.pdf` | Earlier version of polyhedra spec (unique detail on surrogates/Benders) |
| `archive/novel-solutions.md` | Original 6 hypotheses — SDF/OT experimentally dead |
