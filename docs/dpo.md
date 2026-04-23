# Differentiable Proxy Optimization (DPO)

Last updated: 2026-04-23

---

## 1. The Objective Mismatch

Every approach we have tried optimizes a proxy of the proxy cost. The
competition scores placements by

$$f(p) = \text{WL}(p) + 0.5\,D(p) + 0.5\,C(p)$$

where WL is normalized HPWL, D is top-10% grid density, and C is ABU-5%
routing congestion. Our polyhedra navigator optimizes HPWL via LP. RePlAce
optimizes an electrostatic potential. Both optimize *surrogate objectives*
that are only loosely correlated with f.

The empirical evidence is stark:

| Signal | Correlation with f(p) | Source |
|--------|----------------------|--------|
| LP-HPWL | rho = -0.001 | Miftari experiments, Apr 16 |
| LP-HPWL vs congestion | rho = 0.072 | ibm01, 24 topologies |
| LP-HPWL vs density | rho = -0.536 | ibm01, anti-correlation |
| GridSurrogate (within-benchmark) | rho = 0.17 | Phase 2 profiling |
| RePlAce electrostatic potential | no direct measure | -- |

Congestion is 66.5% of proxy cost. HPWL is blind to congestion. Every
gram of LP optimization effort is wasted on a signal that does not move the
objective. This is not a tuning problem -- it is a structural defect.

**DPO eliminates the mismatch by differentiating through f(p) directly.**

---

## 2. What DPO Is

Represent all macro positions as a single tensor $p \in \mathbb{R}^{N
\times 2}$ with `requires_grad=True`. Build a differentiable computation
graph that approximates $f(p)$, including all three components. Run
gradient-based optimization (Adam, L-BFGS) on p, starting from SDF init.
Use a penalty term for overlaps, annealed over the course of optimization.
Legalize the result with existing robust projection.

This is not new for HPWL+density -- DREAMPlace (Lin et al., DAC 2019) does
exactly this with log-sum-exp HPWL and electrostatic density. What is new:

1. **Congestion is included in the gradient.** DREAMPlace does not
   differentiate through congestion. It evaluates congestion post-hoc
   or uses it as a heuristic weight. We make RUDY congestion fully
   differentiable, so the gradient of f tells every macro exactly how
   to move to reduce routing hotspots.

2. **The objective IS the competition metric.** RePlAce's electrostatic
   potential is a different function from the proxy cost. We differentiate
   through the same formula the competition uses to score placements.

3. **Top-k aggregation is differentiable.** The density and congestion
   costs use order-statistic aggregation (top-10%, top-5%). We handle
   this via PyTorch's native `topk`, which passes gradients through the
   selected elements.

---

## 3. Making Each Component Differentiable

### 3.1 Wirelength: Log-Sum-Exp HPWL

The standard smooth HPWL approximation (Naylor et al., 2001; used by
all analytical placers):

$$\text{HPWL}_j \approx \gamma \ln\!\sum_{i \in S_j} e^{x_i/\gamma}
+ \gamma \ln\!\sum_{i \in S_j} e^{-x_i/\gamma}
+ [\text{same for } y]$$

As $\gamma \to 0$, this recovers exact HPWL. The smoothing parameter
$\gamma$ is annealed during optimization (starting large for smooth
landscape, decreasing for accuracy). DREAMPlace uses
$\gamma \in [0.01W, 0.001W]$ where W is canvas width.

Pin positions are affine functions of macro centers:
$\text{pin}_{ij} = p_i + \text{offset}_{ij}$, so HPWL gradients propagate
directly to macro positions.

Normalized by $(W + H) \cdot M$ where M is total net count, matching
PlacementCost.get_cost().

**Gradient interpretation:** $\nabla_{p_i} \text{WL}$ points toward the
centroid of each net's bounding box that macro i participates in. Macros
at bounding box extremes receive the strongest pull.

### 3.2 Density: Differentiable Grid Overlap

For each macro i and grid cell (r,c), the overlap area is:

$$a_{i,rc} = \max(0, \min(x_i^+, x_{rc}^+) - \max(x_i^-, x_{rc}^-))
\cdot \max(0, \min(y_i^+, y_{rc}^+) - \max(y_i^-, y_{rc}^-))$$

where $x_i^\pm = x_i \pm w_i/2$ are the macro edges and $x_{rc}^\pm$
are the cell edges. This is piecewise bilinear in $p_i$ -- smooth almost
everywhere, with kinks only when a macro edge aligns exactly with a cell
boundary.

PyTorch's `torch.clamp(min=0)` and `torch.min`/`torch.max` provide
subgradients at the non-smooth points. The density of cell (r,c) is:

$$\rho_{rc} = \frac{1}{A_\text{cell}} \sum_i a_{i,rc}$$

This is a simple linear combination of overlap areas, fully differentiable.

**Top-10% aggregation.** `torch.topk(rho.flatten(), k)` returns the k
largest values with gradients flowing through the selected elements. The
density cost is:

$$D(p) = 0.5 \cdot \text{mean}(\text{topk}(\rho, \lceil 0.1 \cdot RC \rceil))$$

PyTorch's topk is natively differentiable: in the backward pass, gradients
are routed to the positions of the selected elements. Elements not in the
top-k receive zero gradient. This is a straight-through selection --
exact in the forward pass, approximate at the selection boundary in the
backward pass.

**Gradient interpretation:** $\nabla_{p_i} D$ pushes macro i away from
the densest grid cells. Only macros overlapping top-10% cells receive
gradient signal. Macros in sparse regions are invisible to the density
gradient.

### 3.3 Congestion: Differentiable RUDY

RUDY (Rectangular Uniform wire densitY) estimates routing demand by
spreading each net's routing uniformly across its bounding box. For net j
with bounding box $[x_j^{\min}, x_j^{\max}] \times [y_j^{\min},
y_j^{\max}]$:

**Step 1: Smooth bounding box.** Use the same log-sum-exp as HPWL:

$$x_j^{\max} \approx \gamma \ln \sum_{i \in S_j} e^{x_i/\gamma}, \quad
x_j^{\min} \approx -\gamma \ln \sum_{i \in S_j} e^{-x_i/\gamma}$$

**Step 2: Per-cell demand.** In PlacementCost, RUDY distributes demand
uniformly across all grid cells covered by the bounding box:

$$h_j(r,c) = \frac{1}{n_{\text{cols}} \cdot c_H}, \quad
v_j(r,c) = \frac{1}{n_{\text{rows}} \cdot c_V}$$

where $n_{\text{cols}}, n_{\text{rows}}$ are the number of grid cells
spanned and $c_H, c_V$ are routing capacities per cell.

The discrete "cell is inside or outside bbox" test is non-differentiable.
We replace it with continuous fractional overlap between the net bounding
box and each grid cell -- the same bilinear overlap computation used for
density. This gives smoothly varying demand per cell.

**Step 3: Aggregate.** Total congestion per cell is the sum of H and V
routing congestion across all nets:

$$\kappa_{rc} = \sum_j (h_j(r,c) + v_j(r,c))$$

**ABU-5% aggregation.** Same as density:

$$C(p) = \text{mean}(\text{topk}(\kappa, \lceil 0.05 \cdot 2RC \rceil))$$

(Factor of 2 because H and V are concatenated in PlacementCost.)

**Gradient interpretation:** $\nabla_{p_i} C$ tells macro i how to
move to reduce congestion in the most congested cells. If macro i has
pins in nets whose bounding boxes overlap the top-5% most congested
cells, the gradient pushes i to *shrink those bounding boxes* (move
toward net centroids) or *shift them away from hotspots*. This is the
signal that HPWL-based optimization entirely lacks.

### 3.4 Overlap Penalty

For hard macro pairs (i,k), the overlap area is:

$$O_{ik} = \text{ReLU}\!\left(\tfrac{w_i+w_k}{2} - |x_i - x_k|\right)
\cdot \text{ReLU}\!\left(\tfrac{h_i+h_k}{2} - |y_i - y_k|\right)$$

Total overlap penalty:

$$\mathcal{P}(p) = \sum_{i < k} O_{ik}$$

This is fully differentiable (ReLU and abs have subgradients everywhere).
For N=537 macros, the pairwise computation is ~144K pairs -- a single
vectorized 537x537 matrix operation, trivially fast on CPU or MPS.

**Gradient interpretation:** $\nabla_{p_i} \mathcal{P}$ pushes macro i
away from all macros it overlaps with, proportional to the overlap depth
in each axis. The gradient direction is the minimum-displacement
separation direction.

### 3.5 Canvas Bounds

Enforce via clamping after each gradient step:

$$x_i \leftarrow \text{clamp}(x_i, w_i/2, W - w_i/2)$$

Or include as a soft penalty. Clamping is simpler and sufficient.

---

## 4. The Composite Loss

$$\mathcal{L}(p) = \text{WL}(p) + 0.5\,D(p) + 0.5\,C(p) + \lambda(t)\,\mathcal{P}(p)$$

where $\lambda(t)$ is annealed during optimization:

- **Phase 1 (exploration):** $\lambda$ small, $\gamma$ large.
  The landscape is smooth, overlaps are cheap. The optimizer finds a
  good region of position space, ignoring hard feasibility.

- **Phase 2 (refinement):** $\lambda$ increases, $\gamma$ decreases.
  The landscape sharpens, overlaps become expensive. The optimizer
  separates macros while maintaining low proxy cost.

- **Phase 3 (legalization):** $\lambda \to \infty$ or switch to
  hard projection. Zero overlaps achieved via existing robust
  projection (cascade repair + overlap repair).

This graduated scheme is a classical **penalty continuation method**
(Bertsekas, 1982). The key property: each phase's solution is a warm
start for the next, and if the penalty schedule is slow enough, the
optimizer tracks the constrained optimum through the penalty landscape.

---

## 5. Comparison to Current Approaches

### 5.1 vs Polyhedra Navigation

| Dimension | Polyhedra Navigation | DPO |
|-----------|---------------------|-----|
| **What it optimizes** | HPWL (via LP) | Proxy cost f(p) directly |
| **Correlation with f** | rho = -0.001 | rho = 1.0 (by construction) |
| **Macros moved per step** | 1-5 pairs flipped | ALL macros simultaneously |
| **Congestion signal** | None (LP is blind) | Direct gradient of C(p) |
| **Overlap handling** | Feasible by construction | Penalty + legalization |
| **Steps per benchmark** | 3-20 (LP-limited) | 1000+ (gradient is cheap) |
| **Landscape** | Discrete (topology graph) | Continuous (smooth loss) |

Polyhedra navigation's strength is guaranteed feasibility -- every
intermediate state is overlap-free. Its weakness is the objective
mismatch: it optimizes HPWL, which is uncorrelated with the actual
objective. DPO inverts this tradeoff: it optimizes the right objective
but must manage feasibility via penalties.

The overnight sweep (22 experiments) proved that no LP-level modification
can bridge the objective mismatch. Surrogate fixes, alternative inits, and
LP reformulations are all washed out by navigation because navigation
itself optimizes the wrong thing.

### 5.2 vs RePlAce / DREAMPlace

| Dimension | RePlAce/DREAMPlace | DPO |
|-----------|-------------------|-----|
| **Density model** | Electrostatic potential (Poisson) | Grid overlap (matches scorer) |
| **Congestion** | Not in gradient (post-hoc eval) | In gradient (RUDY) |
| **Objective** | Smooth HPWL + electrostatic | Smooth proxy cost |
| **Legalization** | One-shot, loses 10-15% | Graduated penalty + robust projection |
| **Overlap during opt** | Allowed (penalty) | Allowed (penalty) |

RePlAce's electrostatic model is a smooth approximation of density, but
it is *not* the same function as top-10% grid density. The electrostatic
potential penalizes density everywhere, while top-10% only penalizes the
worst cells. This mismatch means RePlAce spreads macros uniformly when
the objective only cares about hotspots.

More critically, RePlAce has no congestion gradient. DREAMPlace has
explored congestion-aware placement (RUDY-based net weighting), but the
congestion signal enters as a *weighting on HPWL*, not as a direct
gradient of congestion cost. The gradient of "congestion-weighted HPWL"
is not the same as the gradient of congestion itself.

DPO differentiates through the actual congestion cost. The gradient
tells each macro: "moving you by dx in direction v will change the top-5%
congestion by this much." This is fundamentally more informative than
"your nets are congested, so we'll weight your HPWL more."

### 5.3 vs Simulated Annealing

SA with real proxy evaluation optimizes the right objective but makes
random moves. DPO also optimizes the right objective but uses gradient
information to make informed moves. With N=500 macros and 1000 degrees
of freedom, gradient-guided search is exponentially more efficient than
random perturbation.

SA's advantage is that it naturally handles the non-convex landscape
via stochastic acceptance. DPO can be combined with SA for a hybrid:
gradient descent for fast convergence, SA for escaping local minima.

---

## 6. Theoretical Foundations

### 6.1 Graduated Optimization (Continuation Methods)

The penalty annealing schedule $\lambda(t)$ is an instance of
graduated optimization (Hazan, Levy & Shalev-Shwartz, 2016), itself
rooted in continuation methods from numerical analysis (Allgower &
Georg, 1990). The idea: solve a sequence of progressively harder
problems, using each solution to warm-start the next.

The convergence theory requires that the penalty landscape has no
spurious local minima at low penalty -- which is plausible when
starting from SDF init, since the initial placement is already
near-feasible and in a good basin.

Mobahi & Fisher (2015) formalize this as Gaussian smoothing of the
objective: convolving f with a Gaussian of decreasing variance creates
a sequence of smoothed objectives $f_\sigma$ that progressively
reveal the true landscape. The single-loop variant (SLGH, Iwakiri et
al., NeurIPS 2022) eliminates the double-loop overhead.

In our setting, both $\gamma$ (HPWL smoothing) and $\lambda$ (overlap
penalty) serve as continuation parameters. The optimization trajectory
is a path through the family of smoothed problems:

$$f_{\gamma,\lambda}(p) = \text{WL}_\gamma(p) + 0.5\,D(p)
+ 0.5\,C_\gamma(p) + \lambda\,\mathcal{P}(p)$$

### 6.2 Connection to Optimal Transport

The density component is fundamentally an optimal transport problem
(see theory.md section 4d). Given macro mass distribution $\mu_0$ and
target uniform density $\mu_\text{target}$, minimizing density cost
is equivalent to finding a transport map that reduces peak congestion.

RePlAce implicitly solves this via the Poisson equation -- its
electrostatic potential $\phi$ satisfying $\nabla^2\phi = \rho -
\rho_\text{target}$ is the Brenier potential, with optimal transport
map $T(x) = x - \nabla\phi(x)$.

DPO's density gradient is a discretized version of the same transport
force, but applied to the *actual* grid density metric (top-10%) rather
than the smooth electrostatic approximation. The key difference:
DPO's gradient is zero for macros not overlapping top-10% cells (sparse
transport), while RePlAce's electrostatic force is nonzero everywhere
(dense transport). Sparse transport is more efficient because it only
moves macros that matter.

### 6.3 Why Congestion Gradients Are Informative

The congestion cost C(p) depends on net bounding box geometry. Moving
macro i affects the bounding boxes of all nets containing i. The
gradient $\nabla_{p_i} C$ aggregates these effects:

$$\frac{\partial C}{\partial x_i} = \sum_{j : i \in S_j}
\frac{\partial C}{\partial \text{bbox}_j} \cdot
\frac{\partial \text{bbox}_j}{\partial x_i}$$

The first factor ($\partial C / \partial \text{bbox}_j$) captures how
net j's bounding box contributes to the top-5% congested cells. Nets
whose bounding boxes overlap congestion hotspots have large derivatives.

The second factor ($\partial \text{bbox}_j / \partial x_i$) is nonzero
only if macro i is at the extreme of net j's bounding box (i.e., i
determines the max or min coordinate). With log-sum-exp smoothing,
all macros in the net contribute, with weight proportional to how close
they are to the extreme.

**The result:** macros that are (a) at the boundary of nets that (b)
route through congested cells receive a strong gradient signal to move
inward, shrinking those nets' bounding boxes and reducing demand on
congested cells. This is precisely the "coordinated multi-macro move"
that our polyhedra navigator cannot make -- it moves all contributing
macros simultaneously, by a coordinated amount, in the right direction.

### 6.4 The Top-k Gradient and Hotspot Focusing

The top-k selection creates a natural focusing mechanism. In early
optimization (large $\gamma$), many cells have similar density/congestion,
so the top-k set is unstable and gradients are diffuse. As $\gamma$
decreases and the landscape sharpens, the top-k set stabilizes around
the true hotspots, and gradients concentrate on the macros contributing
to those hotspots.

This is related to the hard-mining effect in machine learning: focusing
gradient updates on the hardest examples (Shrivastava, Gupta & Girshick,
2016). In placement, the "hardest examples" are the most congested cells,
and focusing on them is precisely what the proxy cost formula demands.

### 6.5 Connections to Our Theory Stack

DPO complements rather than replaces the polyhedra theory (theory.md):

- **Complexification (section 2a):** DPO's overlap penalty implicitly
  allows temporary passage through infeasible (overlapping) states,
  analogous to complexification's imaginary-dimension detour. The
  penalty parameter $\lambda$ controls how "expensive" this detour is
  -- at $\lambda=0$ the walls dissolve completely, at $\lambda=\infty$
  they are impenetrable.

- **Mountain pass (section 2b):** Graduated optimization naturally
  traverses saddle points. At high smoothing, the landscape has fewer
  local minima, and the optimizer settles into a basin that may be
  different from the SDF basin. As smoothing decreases, the solution
  refines within the selected basin. The continuation path crosses
  barriers that are impassable in the unsmoothed landscape.

- **Barrier crossing (section 1):** The polyhedra navigator's 100+
  flip-wide barrier becomes a smooth hill in DPO's relaxed landscape.
  By allowing overlaps (at a cost), DPO can continuously deform the
  placement through configurations that are infeasible in the hard-
  constraint formulation, reaching basins that topology-local search
  cannot access.

---

## 7. Why 1.22 Is Reachable

The swap+LP experiment proved that topologies exist with 44% lower
congestion than SDF. But those topologies have 180% higher density
because the LP (optimizing HPWL) ignores density. DPO optimizes
density and congestion *jointly*. The gradient of the full proxy cost
naturally balances these competing objectives:

- WL gradient: pull macros toward net centroids (clustering)
- Density gradient: push macros away from hotspots (spreading)
- Congestion gradient: reshape net bounding boxes (redistributing)

The equilibrium of these three forces is a placement that
simultaneously minimizes all three components -- something no
single-objective optimizer can achieve.

Our current best (1.49) loses to RePlAce (1.46) primarily on
congestion. RePlAce itself likely loses to 1.22 on all three
components. The 1.22 leader is almost certainly optimizing the full
proxy cost or something close to it.

**Ceiling estimate.** If DPO achieves:
- Congestion reduction from "SDF-like" to "swap+LP-like" (44% lower)
  while maintaining SDF-like density: congestion component drops from
  ~0.99 to ~0.55, saving ~0.22 on proxy cost (with 0.5 weight).
- Density improvement of 10-20% through gradient-guided spreading:
  saves ~0.03-0.06.
- Wirelength stays comparable: saves 0-0.02.

Total savings: ~0.25-0.30 from our 1.49, reaching ~1.19-1.24. This
puts 1.22 within the ceiling.

---

## 8. Technical Risks and Mitigations

### 8.1 Smooth approximation fidelity

The smooth HPWL and RUDY computations don't exactly match PlacementCost.
Mitigation: anneal $\gamma$ to small values so the smooth objective
converges to the true objective. Verify final placement with exact
compute_proxy_cost.

### 8.2 Non-convexity and local minima

The proxy cost is non-convex. Gradient descent may find a local minimum
worse than SDF init. Mitigation: (1) start from SDF (already a good
basin), (2) use multiple random restarts, (3) combine with SA for
local refinement.

### 8.3 Overlap elimination

Penalty methods don't guarantee zero overlaps. Mitigation: use our
existing robust projection (cascade repair + direct overlap repair)
as a final step. This has proven zero-overlap on all 17 benchmarks.

### 8.4 Congestion model mismatch

PlacementCost uses Steiner-tree routing for 3+ pin nets, not pure RUDY.
The smooth RUDY model may not match exactly. Mitigation: calibrate RUDY
against PlacementCost on initial placement, add correction factors. Use
exact compute_proxy_cost for final verification.

### 8.5 Gradient magnitude balancing

WL, D, and C operate on different scales. Gradient magnitudes may be
dominated by one component. Mitigation: normalize gradients per
component, or use Adam optimizer (which adapts learning rates per
parameter).

---

## 9. Architecture Sketch

```
SDF init (3s)
    |
    v
[DPO Phase 1: Exploration]         gamma = 0.01*W, lambda = 0.1
    Adam optimizer, 200 steps        All macros moved simultaneously
    Landscape is smooth, overlaps cheap
    |
    v
[DPO Phase 2: Refinement]          gamma = 0.001*W, lambda = 10.0
    Adam or L-BFGS, 300 steps       Landscape sharpens
    Overlaps become expensive
    |
    v
[DPO Phase 3: Sharpening]          gamma = 0.0001*W, lambda = 1000.0
    L-BFGS, 200 steps              Near-exact HPWL, near-zero overlaps
    |
    v
[Legalization]                      Robust projection (cascade + direct)
    Zero overlaps guaranteed
    |
    v
[Optional: SA refinement]          Single-macro moves, real proxy eval
    Polish with exact signal         50-200 improving moves
    |
    v
Final placement
```

Estimated runtime: 10-30s per benchmark (gradient steps are O(N*M) where
N=macros, M=nets, with vectorized PyTorch ops). Fits within 60s budget
with room for SA refinement.

---

## 10. What Makes This Novel

1. **First differentiable placement through the actual competition
   metric.** DREAMPlace differentiates HPWL + electrostatic density.
   We differentiate WL + top-10% density + ABU-5% RUDY congestion --
   the exact formula used for scoring.

2. **Congestion in the gradient.** No prior work includes a differentiable
   RUDY congestion model in the placement gradient. Congestion-aware
   DREAMPlace (Lu et al., 2020) uses congestion as net weights on HPWL,
   not as a direct gradient of congestion cost.

3. **Top-k aggregation for placement.** Using differentiable top-k to
   focus gradient signal on density/congestion hotspots. Related to
   hard-example mining in ML, but novel in the placement context.

4. **Penalty continuation from a feasible init.** Starting from SDF
   (feasible, good density) and using graduated penalty to traverse
   infeasible configurations on the way to a better basin. The overlap
   penalty serves the same role as complexification's imaginary
   dimension -- it temporarily dissolves the walls between polyhedra.

5. **Hybrid with robust projection.** The final legalization uses
   proven zero-overlap repair, combining the best of both worlds:
   DPO's global optimization with polyhedra navigation's feasibility
   guarantees.

---

## See Also

- [problem.md](problem.md) -- formal problem statement with objective
- [theory.md](theory.md) -- polyhedra theory, barrier analysis, tunneling
- [approach.md](approach.md) -- current polyhedra navigation architecture
- [results.md](results.md) -- empirical evidence for objective mismatch
