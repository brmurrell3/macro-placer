# Tunneling Between Basins: Theoretical Framework

Last updated: 2026-04-05

---

## The Problem, Precisely

We have a graph G = (V, E) where:
- **Vertices** = polyhedra (topologies σ, each a pairwise L/R/A/B assignment)
- **Edges** = adjacency (two polyhedra share a codimension-1 face, i.e., differ by one pair flip)
- **Energy** E(σ) = LP optimal cost within polyhedron σ (or real proxy cost of LP solution)

We're at vertex σ_SDF with E(σ_SDF) = 1.715. We know vertices exist with
congestion 43% lower (the LP experiment proved this). But every neighbor of
σ_SDF has equal or higher energy. Classical local search (walking downhill on G)
is stuck.

**The question: how do we reach a distant low-energy vertex without abandoning
the graph (feasibility)?**

This is not a search efficiency problem — we tried 3200 neighbors (sequence
pairs), 500-step random walks, and group-level restructuring. The basin around
σ_SDF is genuinely a local minimum in all directions we can probe. We need
to cross a barrier.

---

## Framework 1: Complexification — Dissolving the Walls

### The theorem

In R^{2N}, the feasible region F = ∪ P_σ is **disconnected**. The polyhedra
are separated by hyperplane walls {x : x_j - x_i = w_i/2 + w_j/2}. No
continuous path within F connects different polyhedra without crossing a wall.

In C^{2N}, the same hyperplanes become **complex codimension-1** (real
codimension 2). The complement of a complex hyperplane arrangement is always
**path-connected** (Zariski, 1962). Therefore:

> For any two feasible placements x_A, x_B ∈ F, there exists a continuous
> path γ: [0,1] → C^{2N} with γ(0) = x_A, γ(1) = x_B, and γ(t) satisfying
> all separation constraints for every t.

The path goes "around" the walls through the imaginary dimensions, not through
them. At the endpoints, Im(γ) = 0 (real placements). In between, Im(γ) ≠ 0
(complex coordinates — not physical, but mathematically valid).

### What the imaginary part means

Think of x_i = a_i + ib_i where a_i is the real position and b_i is a
"virtual displacement." The separation constraint for pair (i,j) in
direction L becomes:

  Re(x_j - x_i) ≥ (w_i + w_j)/2  OR  the complex constraint is satisfied

The imaginary part b_i creates "room" that doesn't exist in real space.
Two macros that overlap in real projection can be separated in the complex
plane. This is exactly the "3D lifting" idea from strategy.md — the
imaginary dimension is a synthetic z-axis.

### How to traverse the path algorithmically

**Homotopy continuation** (Sommese & Wampler, Bertini):

1. Reformulate non-overlap as complementarity: for each pair (i,j), introduce
   slack variables s_k ≥ 0 for each direction, with ∏_k s_k = 0
2. In C^N, this system has finitely many solutions (the isolated optima of
   each polyhedron)
3. Define a homotopy H(x,t) that deforms one system (at t=0, easy to solve)
   into the target system (at t=1, our problem)
4. Track solution paths as t varies

**The practical version** (parametric LP):

As we continuously vary a parameter t that interpolates between topology σ_A
and σ_B, the LP optimal solution traces a **piecewise-linear path** through
the feasible region. At each breakpoint, the LP basis changes (a constraint
becomes active/inactive). This is classical parametric LP theory.

The key insight: we don't need to solve in C^{2N}. The parametric LP gives us
a real-valued path. The "complexification" guarantees the path exists — the
parametric LP finds it by tracking basis changes.

**What parameter to vary?** The separation constraints themselves. For each
pair (i,j) that differs between σ_A and σ_B, introduce a parameter t_{ij}
that continuously rotates the constraint from direction A to direction B.
As t varies, the LP solution moves continuously through real space, passing
through boundary polyhedra (where the constraint is tight from both sides).

### Connection to our problem

The interpolation experiment (alpha blending SDF ↔ LP positions) failed because
it moved in **position space**, not in **constraint space**. The parametric LP
moves in constraint space — it doesn't interpolate positions, it rotates
constraints and lets the LP find new optimal positions at each step.

---

## Framework 2: Least Action — The Mountain Pass

### The variational principle

Between two basins (SDF basin at E=1.715, unknown low-congestion basin),
there exists a **mountain pass** — the lowest saddle point connecting them.
The mountain pass theorem (Ambrosetti & Rabinowitz, 1973) guarantees this
saddle exists under mild conditions.

The **least action path** γ* minimizes the maximum energy along the way:

  γ* = argmin_γ max_{t ∈ [0,1]} E(γ(t))

This is the minimax path problem on graph G. The saddle height

  E_saddle = min_γ max_t E(γ(t))

determines the minimum barrier that must be crossed.

### Computing the mountain pass

On the discrete graph G:
- The minimax path from σ_A to σ_B is the path that minimizes the maximum
  vertex energy along the way
- This is solvable by modified Dijkstra: instead of summing edge weights,
  take the max vertex weight along each path, and minimize over paths
- Complexity: O(|V| log |V|) if we could enumerate vertices

But |V| = 4^{N(N-1)/2} is astronomical. We can't enumerate.

### Local approximation: the saddle search

Instead of finding the global mountain pass, find **local saddle points**
near the current basin. A saddle point on the energy landscape is a topology
σ_saddle where:
- E has a local minimum in some directions (the "valley" directions)
- E has a local maximum in other directions (the "pass" directions)

The **dimer method** (Henkelman & Jónsson, 1999) finds saddle points by:
1. Start from a local minimum
2. Find the direction of lowest curvature (the "softest" mode)
3. Move uphill in the softest direction, downhill in all others
4. Converge to the nearest saddle point

For our discrete graph, the analog:
1. Start from σ_SDF
2. Identify which pair flips have the **smallest** cost increase (softest modes)
3. Flip those pairs (go uphill) while allowing the LP to re-optimize (downhill
   in the continuous direction)
4. After crossing the saddle, descend into the new basin

**This is different from our current navigation** because current navigation
only accepts downhill moves. Saddle search deliberately goes uphill in the
softest direction.

### Connection to LP duals

The LP dual variable for pair (i,j) tells us how much the cost would
**decrease** if we relaxed that constraint. The softest mode (smallest cost
increase for a flip) corresponds to pairs where:
- The dual is small (constraint not very active) — flipping costs little
- AND the new direction's dual would be small too — not creating a new
  expensive constraint

We already rank by duals, but we only flip pairs with LARGE duals (most
expensive constraints). For saddle search, we should flip pairs with SMALL
duals — the easiest constraints to change. Then the LP re-optimizes within
the new topology, potentially finding a lower-energy basin on the other side.

---

## Framework 3: Quantum Tunneling — The Path Integral

### The Potts model

The doc establishes the exact mapping: each macro pair (i,j) carries a
**4-state Potts spin** σ_{ij} ∈ {L, R, A, B}. The energy of a spin
configuration is E(σ) = LP optimal cost. The Boltzmann distribution is:

  p(σ) ∝ exp(-β E(σ))

At high temperature (β → 0): all topologies equally likely (exploration).
At low temperature (β → ∞): concentrates on the global optimum (exploitation).

### Why classical SA fails

Classical simulated annealing flips one spin at a time, accepting uphill
moves with probability exp(-β ΔE). For our problem:
- Each flip changes the LP optimal cost by ΔE
- At the right temperature, SA would accept uphill moves to cross barriers
- But **evaluating ΔE requires an LP solve** (~seconds), so we can only
  try ~30 flips per benchmark in the time budget
- The barrier between basins is ~100+ flips wide
- SA can't cross a 100-step barrier with 30 total evaluations

### Quantum tunneling

In quantum mechanics, a particle can tunnel through a barrier it cannot
classically cross. The tunneling amplitude is:

  T ∝ exp(-1/ℏ ∫ √(2m(V(x) - E)) dx)

The integral is over the classically forbidden region (where V > E). The
key: tunneling probability depends on **barrier width × barrier height**.
Tall but narrow barriers are easy; short but wide barriers are hard.

For our problem:
- The "particle" is the current topology σ
- The "potential" V(σ) is the LP cost E(σ)
- The "barrier" between basins is the ridge of high-cost topologies separating
  them
- Tunneling amplitude depends on how many high-cost topologies must be
  traversed

### The Feynman path integral formulation

The transition amplitude from σ_A to σ_B is:

  K(A → B) = Σ_{paths γ: A→B} exp(-β Σ_t E(γ(t)))

This sums over ALL paths, weighted by Boltzmann factors. Paths through low
barriers contribute exponentially more than paths through high barriers.

The dominant path is the **instanton** — the saddle-point of the action
functional S[γ] = Σ_t E(γ(t)). This is exactly the mountain pass from
Framework 2!

### Practical connection: the transfer matrix

For a 1D path of T steps: σ_0 → σ_1 → ... → σ_T, the path weight is

  W = exp(-β Σ_t E(σ_t))

The transfer matrix T_{σ,σ'} = exp(-β E(σ)) × [σ and σ' are adjacent]
encodes single-step transitions. The T-step transition is:

  K(A → B; T) = (T^T)_{A,B}

The eigenvalue decomposition of T reveals:
- The largest eigenvalue λ_0 corresponds to the equilibrium (ground state)
- The gap λ_0 - λ_1 determines the mixing time (how fast SA converges)
- Small gap = hard barrier = slow tunneling

**For quantum annealing**: replace the classical transfer matrix with a
quantum Hamiltonian H = V(σ) + Γ Σ_{ij} X_{ij} where X_{ij} is the
flip operator for spin (i,j) and Γ is the transverse field strength.
At large Γ: quantum fluctuations enable tunneling. Anneal Γ → 0 to
settle into the ground state.

We can't simulate quantum mechanics efficiently on a classical computer.
But the **structure** revealed by the path integral is useful:

> The paths that contribute most to tunneling go through the **softest
> modes** — the spin directions where V changes least. These are the
> saddle points from Framework 2.

---

## Synthesis: What the Three Frameworks Agree On

All three frameworks point to the same algorithmic insight:

1. **Complexification**: The path between basins exists, goes through
   constraint space (not position space), and passes through codimension-1
   boundaries where one constraint rotates between directions.

2. **Least action**: The optimal path crosses at the **saddle point** — the
   topology where flipping specific pairs costs the least. This is the
   mountain pass.

3. **Quantum tunneling**: The dominant tunneling path is the **instanton**,
   which follows the softest modes — pairs with the smallest cost to flip.

**The unified algorithm:**

```
1. At current topology σ, solve LP to get duals
2. Identify the SOFTEST pairs: those where |dual| is SMALL 
   (flipping costs little) — NOT the hardest pairs (large duals)
3. Flip the softest pair, re-solve LP
4. Accept EVEN IF cost increases (barrier crossing)
5. After crossing the saddle (cost starts decreasing), switch to greedy descent
6. Check if the new basin has better congestion
```

This is the **opposite** of our current navigation strategy, which flips
pairs with the LARGEST duals (most expensive to maintain). That strategy
optimizes within a basin. Saddle search flips the SOFTEST pairs to cross
between basins.

### Why this might work for congestion

The LP duals reflect HPWL costs. Soft pairs (small duals) are constraints
that don't matter much for wirelength. But they might matter a lot for
congestion (which the LP doesn't optimize). By flipping soft-for-wirelength
pairs, we change the topology in ways that don't affect wirelength much
but might reorganize the spatial structure enough to change congestion.

The SDF topology has many "don't care" pairs — constraints that are slack
in the LP solution. These represent degrees of freedom that wirelength
optimization ignores. Congestion might be hiding in these degrees of freedom.

---

## What to Test First

### Quick validation: soft-pair flipping

Before implementing any of the heavy machinery, test the core hypothesis:

1. Solve LP for σ_SDF, get duals
2. Sort pairs by |dual| ASCENDING (softest first)
3. Flip the softest 5-20 pairs, accepting cost increases
4. Re-solve LP, evaluate proxy
5. Does congestion change more than when flipping the HARDEST pairs?

If soft-pair flips can move congestion where hard-pair flips cannot, the
theory is validated and we proceed to the full saddle search.

### Parametric LP continuation

1. Pick the softest pair (i,j), currently direction L
2. Set up parametric LP: at t=0, constraint is L; at t=1, constraint is R
3. For t = 0.0, 0.1, ..., 1.0: solve LP with the rotated constraint
4. Track how positions move, how proxy cost changes
5. Is there a t_saddle where cost peaks then decreases?

If the parametric LP shows a saddle, we've found the mountain pass.

### Cavity method marginals

1. Build the factor graph: one variable per pair, one factor per macro
   (connecting all pairs involving that macro)
2. Run belief propagation: compute marginal probabilities p(σ_{ij} = d)
   for each direction d
3. Identify "soft" spins: p ≈ 0.25 for all directions (undetermined)
4. Identify "frozen" spins: p ≈ 1.0 for one direction (locked)
5. Only flip soft spins — these are the degrees of freedom that don't
   create infeasibility

---

## Risk Assessment

| Approach | Implementability | Theoretical grounding | Risk |
|----------|-----------------|----------------------|------|
| Soft-pair flipping | Easy (1 day) | Medium (heuristic) | May not move congestion either |
| Parametric LP | Medium (3 days) | Strong (exact path tracking) | May be too slow for 500+ macros |
| Cavity method / BP | Hard (1-2 weeks) | Very strong (exact for trees) | Short loops in constraint graph may break BP |
| Full complexification | Very hard (Bertini) | Perfect | Not practical in competition timeline |
| Quantum annealing | N/A (need QC) | Perfect | Not available |

**Recommended order**: Soft-pair flipping (1 day, cheap test) → Parametric
LP (if soft-pair works) → Cavity method (if parametric LP reveals structure).
