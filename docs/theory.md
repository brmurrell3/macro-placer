# Theoretical Foundations

Last updated: 2026-04-13

## 1. The Barrier Problem

We have a graph G = (V, E) where:
- **Vertices** = polyhedra (topologies sigma, each a pairwise L/R/A/B assignment)
- **Edges** = adjacency (two polyhedra share a codimension-1 face, i.e., differ by one pair flip)
- **Energy** E(sigma) = LP optimal cost within polyhedron sigma (or real proxy cost of LP solution)

The feasible region is a union of exponentially many convex polyhedra, one per disjunctive assignment of pairwise left/right/above/below relations. Given any fixed assignment, the problem collapses to a linear program. This "LP inside combinatorics" architecture is the defining structural feature.

The real feasible region F = union of P_sigma is **disconnected** -- the polyhedra are separated by hyperplane walls {x : x_j - x_i = w_i/2 + w_j/2}. No continuous path within F connects different polyhedra without crossing a wall. Local search within one polyhedron cannot reach other polyhedra without crossing infeasible (overlapping) regions.

Classical simulated annealing flips one spin at a time, accepting uphill moves with probability exp(-beta * Delta_E). But evaluating each flip requires an LP solve (~seconds), limiting us to ~30 evaluations per benchmark in the time budget. The barrier between basins is ~100+ flips wide. SA cannot cross a 100-step barrier with 30 total evaluations. We tried 3200 neighbors (sequence pairs), 500-step random walks, and group-level restructuring. The basin around the current minimum is genuinely a local minimum in all directions we can probe.

**The question: how do we reach a distant low-energy vertex without abandoning feasibility?**

This is not a search efficiency problem -- it is a structural problem. We need to cross a barrier, not search more efficiently within the current basin.

Three independent mathematical frameworks -- from algebraic geometry, variational calculus, and quantum statistical mechanics -- converge on the same answer. They are not competitors but complementary views of a single phenomenon.


## 2. Three Frameworks for Barrier Crossing

### 2a. Complexification -- Dissolving the Walls

**The theorem.** In R^{2N}, the hyperplane arrangement disconnects the feasible region into exponentially many polyhedra. In C^{2N}, the same hyperplanes become complex codimension-1 (real codimension 2). The complement of a complex hyperplane arrangement is always path-connected (Zariski, 1962). Therefore:

> For any two feasible placements x_A, x_B in F, there exists a continuous
> path gamma: [0,1] -> C^{2N} with gamma(0) = x_A, gamma(1) = x_B, and
> gamma(t) satisfying all separation constraints for every t.

The path goes "around" the walls through imaginary dimensions. Think of x_i = a_i + ib_i where a_i is the real position and b_i is a "virtual displacement." The separation constraint for pair (i,j) in direction L becomes:

    Re(x_j - x_i) >= (w_i + w_j)/2  OR  the complex constraint is satisfied

The imaginary part b_i creates "room" that does not exist in real space. Two macros that overlap in real projection can be separated in the complex plane. This is a synthetic z-axis -- the imaginary dimension lifts macros off the 2D canvas to route around each other, then sets them back down at the destination polyhedron.

**Homotopy continuation is computationally intractable at scale.** The approach reformulates non-overlap as complementarity: for each pair (i,j), introduce slack variables s_k >= 0 for each direction, with prod_k s_k = 0 (at least one must be active). In C^N, this system has finitely many solutions (the isolated optima of each polyhedron). A homotopy H(x,t) deforms one system (at t=0, easy to solve) into the target system (at t=1, our problem), and solution paths are tracked as t varies.

In practice, PHCpack, Bertini 2.0, and HomotopyContinuation.jl operate on systems of 4 to 20 variables. The fundamental bottleneck is the number of solution paths, which grows exponentially: ten quadratics yield 1,024 paths; twenty quadratics yield over one million. For 500 variables with 100K constraints, the path count is astronomically beyond reach. The closest work is HCBB for mixed-integer nonlinear programming, handling ~8,600 continuous variables but only 13 binary variables -- the discrete component is trivial. No application of homotopy continuation to combinatorial optimization at this scale exists in the literature.

**Parametric LP is the practical residue of complexification.** Instead of tracking paths through C^{2N}, rotate a single separation constraint parametrically (from direction L to direction R) and track the LP optimal basis as it changes. This is classical single-parameter parametric LP, well-developed since Gass & Saaty (1955). For right-hand-side parameterization, the solution is piecewise affine, and the number of basis changes along a single-parameter path is bounded. For constraint-matrix changes (which is what rotating a separation direction entails), the theory is harder -- the feasible region shape changes -- but each basis change requires only a single LP pivot, potentially staying within the evaluation budget.

Kenefake & Pistikopoulos (2024) provide a parallel combinatorial algorithm for multiparametric programming. The primary application domain remains Model Predictive Control, where the parametric LP solution is precomputed as an explicit piecewise-affine map. For our problem, a single-parameter sweep rotating one macro pair's separation constraint crosses polyhedron boundaries while tracking the optimal basis -- a computationally tractable operation costing O(number of basis changes) LP pivots per sweep.

**The key insight**: the interpolation approach of blending positions failed because it moved in **position space**, not **constraint space**. Parametric LP moves in constraint space -- it rotates constraints and lets the LP find new optimal positions at each step.

**What parameter to vary?** The separation constraints themselves. For each pair (i,j) that differs between topology sigma_A and sigma_B, introduce a parameter t_{ij} that continuously rotates the constraint from direction A to direction B. As t varies, the LP solution moves continuously through real space, passing through boundary polyhedra where the constraint is tight from both sides.

**Gaussian homotopy** (Mobahi & Fisher, 2015) offers another practical angle. The Single-Loop variant (SLGH, Iwakiri et al., NeurIPS 2022) eliminates the expensive double-loop structure of classical graduated optimization. However, the zeroth-order variant requires O(d) function evaluations per step -- at d = 500, this exceeds the evaluation budget in a single iteration. The closest VLSI application is **solution-space smoothing** (Dong et al., IEEE GLSVLSI 2003), which solves simplified instances first and uses solutions to guide harder instances -- the continuation idea applied to floorplanning, tested on MCNC benchmarks.

**Simulated lifting** (Shen et al., arXiv 2602.05887, February 2026) is the most intriguing discrete analog of complexification. Over-parameterization via tensor lifting converts local minima into strict saddle points. Their Simulated Oracle Direction mechanism projects escape directions from the lifted space back to the original space *without actually performing the lift*. This is conceptually identical to complexification -- enlarging the space to improve the landscape, then projecting back -- but is currently developed only for matrix sensing, not combinatorial optimization. Adapting simulated lifting to LP-based combinatorial problems would require formalizing the tensor lifting for the disjunctive constraint structure, a non-trivial theoretical extension.


### 2b. Mountain Pass / Least Action

**The variational principle.** Between two basins there exists a mountain pass -- the lowest saddle point connecting them. The mountain pass theorem (Ambrosetti & Rabinowitz, 1973) guarantees this saddle exists under mild conditions. The least action path gamma* minimizes the maximum energy along the way:

    gamma* = argmin_gamma max_{t in [0,1]} E(gamma(t))

On the discrete graph G, the minimax path from sigma_A to sigma_B minimizes the maximum vertex energy along any flip-sequence connecting them:

    E_saddle = min_gamma max_t E(gamma(t))

This saddle height determines the minimum barrier that must be crossed. Modified Dijkstra (replacing sum with max) or minimum spanning tree methods solve this in near-linear time on explicit graphs (Hu, 1961), but the graph has 4^{N(N-1)/2} vertices -- enumeration is impossible. On the implicit exponential graph, no off-the-shelf algorithm exists. The LP-dual-guided greedy approach is a principled heuristic: at each step, it flips the pair minimizing barrier height, using duals as a local gradient.

**Local saddle search via the dimer method.** Instead of finding the global mountain pass, find local saddle points near the current basin. A saddle point on the energy landscape is a topology sigma_saddle where E has a local minimum in some directions (the "valley" directions) and a local maximum in others (the "pass" directions). The **dimer method** (Henkelman & Jonsson, 1999) finds saddle points by: (1) starting from a local minimum, (2) finding the direction of lowest curvature (the "softest" mode), (3) moving uphill in the softest direction, downhill in all others, and (4) converging to the nearest saddle point.

**Gentlest Ascent Dynamics (GAD)** (Weinan E & Xiang Zhou, 2011) is the most conceptually aligned formalization. GAD constructs a dynamical system whose stable fixed points are exactly index-1 saddle points: it tracks the lowest eigenmode of the Hessian, reverses the force component along it (ascending the softest direction), and follows force perpendicular to it (descending in all other directions). Extensions include Constrained GAD (Liu, Xie & Yuan, 2022) and High-index Saddle Dynamics (Yin et al., 2019-2025). GAD typically requires 50-200 force evaluations in chemical systems.

**The discrete analog requires only 2 LP solves per step:** solve the LP to obtain dual variables, identify the pair with the smallest |dual| (the "softest mode"), flip that pair's L/R/A/B assignment, and re-solve. With ~30 LP evaluations this allows ~15 steps -- enough to climb over a moderate saddle and descend into a new basin. **This is different from current navigation** because current navigation only accepts downhill moves; saddle search deliberately goes uphill in the softest direction. The algorithm has two phases:

- **Phase 1 (ascent):** Flip soft pairs (small |duals|) sequentially. Cost rises, but slowly -- climbing the gentlest slope toward the saddle.
- **Phase 2 (descent):** Once cost begins decreasing after a flip, the saddle is crossed. Switch to greedy descent (flip the pair whose flipping reduces cost most).

**Connection to LP duals.** The LP dual variable for pair (i,j) tells us how much cost would decrease if we relaxed that constraint. The softest mode (smallest cost increase for a flip) corresponds to pairs where the dual is small -- flipping costs little. Our current navigation flips pairs with the LARGEST duals (most expensive constraints), which optimizes within a basin. Saddle search flips the SOFTEST pairs to cross between basins.

The **Nudged Elastic Band** (NEB) method and the **string method** are related continuous saddle-search algorithms from computational chemistry. NEB discretizes the path into images connected by springs, optimizing intermediate images to find the minimum energy path. The string method evolves a curve by the force perpendicular to the curve, converging to the minimum energy path without spring forces. Both require gradient information that is available via LP duals in our setting.

The Elastic String Algorithm (More & Munson, 2002, Argonne National Lab) is the most direct continuous implementation of the Mountain Pass Theorem -- it optimizes breakpoints of a piecewise-linear path to minimize maximum elevation. For discrete spaces, the LP-dual-guided approach is more practical since it naturally handles the combinatorial structure.


### 2c. Quantum Tunneling / Potts Model

**The Potts model.** Each macro pair (i,j) carries a 4-state Potts spin sigma_{ij} in {L, R, A, B}. The energy of a spin configuration is E(sigma) = LP optimal cost. The Boltzmann distribution is p(sigma) proportional to exp(-beta * E(sigma)). At high temperature (beta -> 0): all topologies equally likely (exploration). At low temperature (beta -> infinity): concentrates on the global optimum (exploitation).

**Why classical SA fails on this specific problem.** Classical SA flips one spin at a time, accepting uphill moves with probability exp(-beta * Delta_E). For our problem, each flip changes the LP optimal cost by Delta_E. At the right temperature, SA would accept uphill moves to cross barriers. But evaluating Delta_E requires an LP solve (~seconds), so we can only try ~30 flips per benchmark in the time budget. The barrier between basins is ~100+ flips wide. SA cannot cross a 100-step barrier with 30 total evaluations.

**Quantum tunneling also fails for wide barriers.** In quantum mechanics, a particle can tunnel through a barrier it cannot classically cross, with tunneling amplitude T proportional to exp(-1/hbar * integral sqrt(2m(V(x) - E)) dx). The key: tunneling probability depends on barrier width times barrier height. Tall but narrow barriers are easy; short but wide barriers are hard. The Crosson-Harrow FOCS 2016 result proves SQA achieves exponential speedup over classical SA only for thin, high barriers where barrier width zeta and height alpha satisfy alpha + zeta < 1/2. For our ~100+ flip-wide barriers, the instanton action S proportional to width times height yields an astronomically small tunneling amplitude exp(-S). Nannini et al. (2023) confirm: "quantum annealing only performs well exploring functions with high and narrow peaks, while classical annealing is better in overcoming flat and wide energy-profile barriers." No results from 2022-2026 change this picture. SQA is definitively the wrong tool for our barrier geometry.

**The Feynman path integral.** The transition amplitude from sigma_A to sigma_B is:

    K(A -> B) = sum_{paths gamma: A->B} exp(-beta * sum_t E(gamma(t)))

This sums over ALL paths, weighted by Boltzmann factors. Paths through low barriers contribute exponentially more than paths through high barriers. The dominant path is the **instanton** -- the saddle-point of the action functional S[gamma] = sum_t E(gamma(t)). This is exactly the mountain pass from Framework 2. Smelyanskiy et al. (2016) developed instanton calculus for quantum spin models, confirming that QMC escape rates match quantum tunneling rates via dominant instantonic paths. However, their mean-field approach requires permutation symmetry (fully connected models), and no algorithmic implementation exists for finding instantons in general discrete combinatorial problems. The concept remains a theoretical organizing principle.

The transfer matrix T_{sigma,sigma'} = exp(-beta * E(sigma)) * [sigma and sigma' are adjacent] encodes single-step transitions. For a 1D path of T steps sigma_0 -> sigma_1 -> ... -> sigma_T, the path weight is W = exp(-beta * sum_t E(sigma_t)). The T-step transition amplitude is K(A -> B; T) = (T^T)_{A,B}.

The eigenvalue decomposition of T reveals:
- The largest eigenvalue lambda_0 corresponds to equilibrium (ground state)
- The gap lambda_0 - lambda_1 determines mixing time (how fast SA converges)
- Small gap = hard barrier = slow tunneling

This spectral gap directly connects to the evaluation cascade in Section 5: methods that can estimate the transfer matrix's spectral gap cheaply (via Cheeger inequalities on the constraint graph) can predict before search begins whether a particular barrier will be crossable within the evaluation budget.

**For quantum annealing**: replace the classical transfer matrix with a quantum Hamiltonian H = V(sigma) + Gamma * sum_{ij} X_{ij} where X_{ij} is the flip operator for spin (i,j) and Gamma is the transverse field strength. At large Gamma, quantum fluctuations enable tunneling. Anneal Gamma -> 0 to settle into the ground state. We cannot simulate quantum mechanics efficiently on a classical computer, but the structure revealed by the path integral is useful: the paths that contribute most to tunneling go through the softest modes -- the saddle points from Framework 2.

**Discrete Schrodinger Bridge methods** -- DASBS (Guo et al., February 2026), DDSBM (Kim et al., ICLR 2025), SDDS (Sanokowski et al., ICLR 2025) -- learn a continuous-time Markov chain transport from a simple prior to a Boltzmann target. DASBS has been tested on 4-state Potts models on 16x16 lattices (256 variables), directly relevant to our formulation. The fatal constraint is training cost: all methods require 10^5 to 10^6 energy evaluations during neural network training, prohibitive when each evaluation costs an LP solve.

**Population Annealing (PA)** emerges as the practical winner from this framework. PA maintains a population of R replicas cooled simultaneously: at each temperature step, replicas are resampled proportionally to Boltzmann weight ratios, then each undergoes local MCMC sweeps. Wang, Machta & Katzgraber (2015) benchmarked PA on 3D Edwards-Anderson spin glasses (up to N = 512) and found it significantly more efficient than SA and comparable to parallel tempering. PA overcomes wide barriers by maintaining diversity at high temperatures where barriers are surmountable. GPU implementations (Barash, Weigel & Janke, 2017-2019) handle thousands of Potts spins.

The LP-evaluation budget constrains PA: with R = 100 replicas, 30 temperature steps, and 5 sweeps per temperature, each sweep proposing N = 100 single-pair flips, you need ~1.5 million LP solves. Feasible only if incremental LP re-solves (warm-starting from the previous basis after a single constraint flip) cost milliseconds. If each LP solve truly costs seconds, PA requires restricting to very small populations (R = 5-10) with coarse temperature schedules.

**Kolmogorov complexity sampling** (Dingle & Hutter, Entropy, February 2026) provides an intriguing theoretical result: optima of combinatorial problems defined by simple objective functions tend to have low Kolmogorov complexity. They propose sampling according to algorithmic probability P(x) proportional to 2^{-K(x)}. Since Kolmogorov complexity is uncomputable, practical implementation uses compression proxies (Lempel-Ziv complexity, gzip length). For our 4-state Potts assignment, a "simple" solution might exhibit regular patterns in separation directions. Implementation is trivial (compute compression length of candidate assignments, add a bias term to Metropolis acceptance), but the payoff is speculative -- no empirical validation exists for placement-type problems.


### 2d. Unified Insight

All three frameworks point to the same algorithmic principle:

1. **Complexification**: The path between basins exists, goes through constraint space (not position space), and passes through codimension-1 boundaries where one constraint rotates between directions.

2. **Least action**: The optimal path crosses at the saddle point -- the topology where flipping specific pairs costs the least. This is the mountain pass.

3. **Quantum tunneling**: The dominant tunneling path is the instanton, which follows the softest modes -- pairs with the smallest cost to flip. The instanton is the saddle-point of the Feynman path integral action, which is exactly the mountain pass -- the bridge between Frameworks 2 and 3.

The three frameworks are not competitors. They converge on a single operational principle: **identify soft degrees of freedom via LP duality, and navigate through them.** This is the dimer method's "softest mode," the mountain pass theorem's "lowest saddle," and statistical mechanics' "unfrozen spins" -- all the same object viewed through different lenses.

**The unified algorithm: flip SOFT pairs, not HARD pairs.**

```
1. At current topology sigma, solve LP to get duals
2. Identify SOFTEST pairs: those where |dual| is SMALL
   (flipping costs little) -- NOT the hardest pairs (large duals)
3. Flip the softest pair, re-solve LP
4. Accept EVEN IF cost increases (barrier crossing)
5. After crossing the saddle (cost starts decreasing), switch to greedy descent
6. Check if the new basin has lower cost
```

This is the opposite of the current navigation strategy, which flips pairs with the LARGEST duals (most expensive to maintain). That strategy optimizes within a basin. Saddle search flips the SOFTEST pairs to cross between basins.

**Why this might work for congestion.** LP duals reflect HPWL costs. Soft pairs (small duals) are constraints that don't matter much for wirelength but might matter a lot for congestion (which the LP doesn't optimize). By flipping soft-for-wirelength pairs, we change topology in ways that don't affect wirelength much but might reorganize spatial structure enough to change congestion. The SDF topology has many "don't care" pairs -- constraints that are slack in the LP solution -- representing degrees of freedom that wirelength optimization ignores.

**Validation experiments for the unified insight:**

1. **Soft-pair flipping (quick test).** Solve LP, sort pairs by |dual| ascending (softest first). Flip the softest 5-20 pairs, accepting cost increases. Re-solve LP, evaluate proxy. Does congestion change more than when flipping the hardest pairs? If soft-pair flips move congestion where hard-pair flips cannot, the theory is validated.

2. **Parametric LP continuation.** Pick the softest pair (i,j), currently direction L. Set up parametric LP: at t=0, constraint is L; at t=1, constraint is R. For t = 0.0, 0.1, ..., 1.0, solve LP with the rotated constraint. Track proxy cost. Is there a t_saddle where cost peaks then decreases? If so, the mountain pass exists.

3. **Cavity method marginals.** Build the factor graph: one variable per pair, one factor per macro (connecting all pairs involving that macro). Run belief propagation: compute marginal probabilities p(sigma_{ij} = d) for each direction d. Identify "soft" spins (p ~ 0.25 for all directions -- undetermined) and "frozen" spins (p ~ 1.0 for one direction -- locked). Only flip soft spins.

| Approach | Implementability | Theoretical grounding | Risk |
|----------|-----------------|----------------------|------|
| Soft-pair flipping | Easy (1 day) | Medium (heuristic) | May not move congestion either |
| Parametric LP | Medium (3 days) | Strong (exact path tracking) | May be too slow for 500+ macros |
| Cavity method / BP | Hard (1-2 weeks) | Very strong (exact for trees) | Short loops in constraint graph may break BP |
| Full complexification | Very hard (Bertini) | Perfect | Not practical in competition timeline |


## 3. Literature Grounding

### Three independent literatures support the soft-mode insight

The synthesis connecting LP dual magnitudes to GAD's softest mode, to backbone theory's unfrozen variables, to survey propagation's joker spins appears in no single paper, though all ingredients are individually well-established.

**Reduced cost fixing** (established in integer programming): variables with large reduced costs are provably fixable at their LP values. The contrapositive -- variables with small reduced costs are flexible and should be explored -- is the same insight. Hougardy & Schroeder (2024, Mathematical Programming Computation) used reduced-cost-based edge elimination for TSP, reducing graphs to under 3n edges.

**Backbone theory** (Slaney & Walsh, 2001; Zeng et al., 2012): backbone variables are frozen in all optimal solutions. Non-backbone (soft) variables define the search space for exploration. Backbone-guided extremal optimization (BGEO) for MAX-SAT operationalizes exactly this principle.

**Survey propagation's joker variables**: in the spin glass theory of random K-SAT, "joker" or "white" variables are not frozen in any solution cluster. Backtracking Survey Propagation (Marino, Parisi & Ricci-Tersenghi, Nature Communications 2016) finds solutions by assigning joker variables first -- the easy variables, not the hard ones.

**The LP-dual / belief-propagation connection is rigorous, not merely analogical.** Max-product belief propagation operates as block coordinate ascent on the LP dual (Sontag et al., UAI 2008; Sanghavi et al. for weighted matching). LP dual variables *are* BP messages in the zero-temperature limit. The soft-mode insight thus connects to replica symmetry breaking: in the RSB phase, the solution space shatters into clusters, and the LP relaxation becomes loose -- fractional LP optima represent superpositions across clusters. Dual variables with small magnitude signal variables that are unresolved between clusters: precisely the joker/soft variables.

This provides the deepest justification for the unified algorithm: flipping small-dual pairs is equivalent to exploring the dimensions of the solution space where clusters disagree -- exactly the directions where barrier crossings between clusters are possible.

### What actually works for VLSI placement today

The empirical landscape provides important context. **Simulated annealing with good move operators remains the gold standard for macro placement.** The TILOS-AI-Institute MacroPlacement benchmark (updated January 2026) shows that an improved SA baseline with multithreading and go-with-the-winners resampling achieves up to 26% better proxy cost than Google's Circuit Training (AlphaChip) RL approach, using only one-quarter of the resources. SA wins 16 of 17 HPWL comparisons on ICCAD04 benchmarks. Markov's reassessment (CACM 2024) confirms that ML-based techniques lag behind analytical placers.

DREAMPlace (Lin et al., DAC 2019) dominates standard-cell placement via GPU-accelerated analytical optimization. **Hybro** (arXiv 2402.18311, 2024) directly addresses the local-minima problem by alternating between DREAMPlace gradient optimization and black-box perturbation -- the closest published system to what we are building. For the ~30-evaluation budget, **Bayesian optimization over combinatorial structures** (BOCS, Baptista & Poloczek, ICML 2018; AutoDMP, Agnesina et al., ISPD 2023) is designed for expensive black-box discrete optimization. AutoDMP already wraps DREAMPlace with multi-objective BO for macro placement parameter tuning.

Adaptive Parallel Tempering with non-local cluster moves (Nature Communications 2025) establishes the strongest known classical baseline for spin-glass-type problems, outperforming both SQA and D-Wave quantum annealers on 3D Ising glasses. Simulated Bifurcation (Goto et al., Science Advances 2021) scales to one million binary variables on Max-Cut, making it the largest-scale physics-inspired optimizer demonstrated, though it operates on QUBO formulations rather than LP-based objectives.

### Maturity assessment

| Method | Maturity | Demonstrated scale | Failure mode | Effort |
|---|---|---|---|---|
| LP-dual-guided discrete GAD | **Novel** (no prior work) | Untested | Soft-pair flips may lead to infeasible regions; saddle may be far | 2 weeks |
| Parametric LP sweeps | **Textbook** (theory since 1955) | 10K+ constraints (MPC) | Many basis changes may exceed budget; A-matrix changes harder | 2 weeks |
| Population Annealing | **Established** (benchmarked 2015+) | 512 spins (3D Ising) | Requires cheap incremental LP; wide barriers need high T | 3 weeks |
| Solution-space smoothing | **Established** (VLSI, 2003) | MCNC benchmarks | Relaxed solutions may not guide to good full solutions | 1 week |
| Bayesian surrogate (BOCS) | **Established** (ICML 2018) | 50-100 binary variables | Surrogate inaccurate in 500-D; pairwise features miss higher-order | 1 week |
| Discrete Schrodinger Bridges | **Recent/speculative** | 256 Potts variables | Needs 10^5+ evaluations; LP cost prohibitive | Not feasible |
| Homotopy continuation (C^{2N}) | **Textbook** but **infeasible** at scale | 20 variables max | Exponential path count | Not feasible |
| SQA / quantum tunneling | **Established** but **wrong regime** | 1000+ qubits (D-Wave) | Wide barriers defeat tunneling exponentially | Not recommended |

### The dual-guided discrete GAD algorithm

The three frameworks converge on a single implementable algorithm:

1. **Discrete GAD with LP dual guidance (Framework 2 core).** At a local minimum sigma_0, solve the LP and extract dual variables. Rank all macro pairs by |dual|. Flip the pair with smallest |dual| to get sigma_1. Re-solve LP. If cost increased, continue flipping soft pairs (ascending toward the saddle). When cost begins decreasing, switch to greedy descent. This uses ~2 LP solves per step, fitting ~15 steps within the 30-evaluation budget. The Mountain Pass Theorem guarantees a saddle exists; GAD theory says softest-mode ascent finds it most efficiently; reduced cost fixing confirms small duals identify flexible variables.

2. **Parametric LP sweeps (Framework 1 residue).** For each macro pair at the local minimum, perform a single-parameter parametric LP sweep rotating its separation constraint (L->R or A->B). Track the optimal basis through boundary crossings. This probes the energy landscape along "rotation paths" between neighboring polyhedra at a cost of one LP pivot per basis change -- potentially much cheaper than a full LP solve. Pairs whose sweeps reveal decreasing-cost trajectories are promising flip candidates.

3. **Population Annealing with warm-start LP (Framework 3).** If incremental LP re-solves can be done in milliseconds, implement PA with a small population (R = 10-30) and coarse temperature schedule. The LP-dual-guided move selection serves as the MCMC proposal distribution, biasing flips toward soft pairs.

4. **Solution-space smoothing (Framework 1 continuation).** Start with a relaxed problem (subset of macros, looser constraints), solve to optimality, add macros or tighten constraints incrementally. Each relaxed problem is itself an LP. This is the practical VLSI implementation of Gaussian homotopy, tested on MCNC benchmarks by Dong et al. (2003).

5. **Bayesian surrogate over assignment space.** Build a sparse Bayesian linear regression surrogate with pairwise monomial features (following BOCS, Baptista & Poloczek, ICML 2018) from all LP evaluations accumulated during search. Use the surrogate to predict which distant assignments might have low cost, and spend remaining LP evaluations verifying the best surrogate predictions. Use LP dual information to construct informative priors -- pairs with large duals should have fixed features in the surrogate. The primary risk is surrogate inaccuracy in 500-D; pairwise features may miss higher-order interactions that determine LP cost.

The **highest-risk, highest-reward** component is the discrete GAD -- it has no precedent and could either find dramatically better solutions or fail to escape if the local minimum is surrounded by a uniformly high barrier with no soft direction. The **lowest-risk** component is Population Annealing, proven on similar problems but potentially bottlenecked by LP-solve cost. The **most underappreciated** is parametric LP sweeps, which exploit classical LP theory to probe the landscape at sub-LP cost per step.

The 30-evaluation budget is the binding constraint. It rules out all sampling-heavy methods and demands that every LP solve be maximally informative, which is precisely what dual-guided navigation provides: each solve reveals not just a cost but a complete sensitivity map of the local landscape.

The genuinely novel contribution of this synthesis is connecting LP dual magnitudes to GAD's softest mode, to backbone theory's unfrozen variables, to survey propagation's joker spins. This connection appears in no single paper, though all ingredients are individually well-established. If validated empirically, the LP-dual-guided discrete GAD could be a significant methodological advance for escaping local minima in structured combinatorial optimization -- not just VLSI placement but any problem where the objective decomposes into an LP over combinatorial assignments.


## 4. Deep Mathematical Connections

The macro placement problem sits at a remarkable intersection of mathematical structures -- tropical, statistical-mechanical, topological, transport-theoretic -- that have developed independently but converge on this single problem. The feasible region is a union of exponentially many convex polyhedra (one per disjunctive assignment of pairwise L/R/A/B relations), and given any fixed assignment the problem collapses to a linear program. This "LP inside combinatorics" architecture creates bridges to several deep mathematical frameworks. What distinguishes these approaches from standard heuristics is that they engage with the problem's specific mathematical structure -- the union-of-polyhedra geometry, the tropical objective, the Potts-like spin system -- rather than treating it as a generic black-box optimization.

Three directions stand out for immediate theoretical impact: tropical geometry provides the algebraically natural language for HPWL optimization; complexification via numerical algebraic geometry offers the only known principled method for connecting the disconnected real feasible regions; and the Fisher-Rao gradient flow combined with natural gradient updates provides a complete framework for the hybrid discrete-continuous structure. The deeper frameworks -- cavity methods, persistent homology -- offer structural insights (phase transition locations, landscape complexity bounds) that guide algorithm design even before they yield direct algorithms.


### 4a. Tropical Geometry

The half-perimeter wirelength HPWL = (max_i x_i - min_i x_i) + (max_i y_i - min_i y_i) is **literally a tropical polynomial** in the (max, +) semiring. The placement objective lives natively in tropical algebraic geometry. The feasible polyhedra under each disjunctive assignment tropicalize into tropical polyhedra, and the full feasible region forms a tropical polyhedral complex.

Three developments make this actionable:

1. **Tropical central path** (Allamigeon, Gaubert, Joswig): characterizes the piecewise-linear limit of interior-point LP methods via the tropical geometry of the normal fan, providing combinatorial lower bounds on LP complexity. For placement, the combinatorial types of optimal placements are encoded in the secondary fan (normal fan of the state polytope) of the parametric LP as wirelength weights vary. This means one can understand how optimal placements change combinatorially as net weights are perturbed.

2. **Tropical gradient descent** (Talbut & Monod, arXiv:2405.19551, 2024): achieves convergence rates matching classical gradient descent on problems with tropical convexity but not classical convexity -- directly applicable since HPWL is tropically convex.

3. **Tropical attention** (Hashemi et al., arXiv:2505.17190, 2025): neural architectures in tropical projective space that universally approximate tropical circuits while preserving sharp polyhedral decision boundaries.

The theoretical gap requiring new work is a **tropical theory of disjunctive programming** -- expressing the 4-term non-overlap disjunction in tropical semiring operations and characterizing the tropical variety of optimal solutions as cost vectors vary. Current tropical LP algorithms have exponential internal representations in high dimensions, so scaling to the 2N-dimensional placement space (with N ~ 500 macros) demands exploiting the sparse, factored structure of pairwise constraints. The work on ideal formulations for rectangle packing (arXiv:2601.15252) develops automated methods to prove that certain MBLP formulations project exactly onto the convex hull -- connecting the disjunctive programming hierarchy to the tropical framework.


### 4b. Cavity Method / Spin Glass Theory

The placement problem maps naturally onto a factor graph where each macro pair (i,j) carries a 4-state Potts spin sigma_{ij} in {L, R, A, B}. The partition function integrates over all spin configurations weighted by LP-optimal cost. This is the structure where the cavity method and survey propagation (Mezard, Parisi, Zecchina -- *Science*, 2002) have produced algorithmic breakthroughs.

**Replica symmetry breaking (RSB) and clustering.** Survey propagation computes distributions over the clustered solution space under 1RSB. The union-of-polyhedra structure maps directly: each polyhedron is one spin configuration, and clusters of nearby polyhedra correspond to the RSB cluster hierarchy. The phase transition phenomenology from random CSP theory (Krzakala, Montanari et al., *PNAS*, 2007) predicts that as the ratio of total macro area to canvas area increases, the solution space undergoes a cascade:

    single connected cluster -> exponentially many clusters (clustering transition)
    -> condensation into few dominant clusters -> infeasibility threshold

**For the ICCAD04 benchmarks at 43-53% area utilization, the problem likely sits near or beyond the clustering transition**, explaining why local search gets trapped. At the clustering transition, the solution space fragments into exponentially many disconnected clusters. Local moves cannot bridge between clusters, but the soft/joker variables identified by survey propagation provide the inter-cluster bridges.

The critical theoretical development needed is **hybrid discrete-continuous belief propagation** -- message-passing that marginalizes over LP optima. Each factor-to-variable message must encode the parametric LP optimal value as a function of adjacent variable assignments. The spatial embedding creates short loops (nearby macro triples have correlated constraints), violating the locally-tree-like assumption. Region-graph methods (Kikuchi approximations) or loop-corrected BP would be necessary. No application of cavity methods to geometric placement with disjunctive constraints has been published.

Tree decomposition and survey propagation both fail on the full problem for a fundamental reason: K_n has treewidth n-1, and the cavity method assumes locally tree-like structure that K_n entirely lacks. However, after conditioning on a backdoor set of 30-50 critical pairs (see Section 5d), the residual constraint graph may have bounded treewidth, enabling exact methods on the tree decomposition.


### 4c. Stratified Morse Theory

The feasible region F = union of P_sigma is a union of polyhedra with shared faces -- a **stratified space** (Goresky & MacPherson, 1988). The strata are: interiors of full-dimensional polyhedra (top strata), shared codimension-1 faces (where one disjunctive assignment changes), shared codimension-2 faces, and so on.

Morse theory on this stratified space decomposes the local Morse data at each critical point into tangential data (within the stratum) and normal data (transverse to it). On each polyhedron P_sigma, the HPWL objective is piecewise-linear, so the LP optimum sits at a vertex or face -- giving at most one critical point per polyhedron. The **Morse inequalities then bound the total number of local minima below by the Betti numbers of F.** Since F is a union of chambers in a hyperplane arrangement, its Betti numbers are computable from the intersection lattice via the Orlik-Solomon algebra and the Mobius function.

**Persistent homology** of the objective filtration F_c = {x in F : Objective(x) <= c} reveals at which cost levels connected components (H_0), tunnels (H_1), and voids (H_2) appear and disappear. Long-lived H_0 features indicate well-separated basins of attraction for local search; short-lived features indicate noise. This topological fingerprint could guide multi-start strategies by identifying the number and relative depth of distinct solution clusters before expensive optimization begins.

The Salvetti complex provides a finite CW-complex homotopy equivalent to the complexified arrangement complement, with cells corresponding to valid disjunctive assignments. Since F is a union of chambers in a hyperplane arrangement, all Betti numbers are computable from the intersection lattice via the Orlik-Solomon algebra and the Mobius function. The practical question is whether these algebraic invariants can be computed efficiently for the ~20K-144K pairwise constraints in our benchmarks.


### 4d. Other Connections

**Optimal transport.** The density minimization component is fundamentally an OT problem. Given N macros with total area A distributed with some density mu_0, the target is uniform density mu_target = A / |Canvas|. The ePlace/DREAMPlace framework already performs approximate OT implicitly -- its electrostatic potential phi solving the Poisson equation is the Brenier transport potential where the optimal map is T(x) = x - nabla*phi(x).

Making this connection explicit unlocks three advances. First, **entropy-regularized OT via Sinkhorn iterations** (Cuturi, NeurIPS 2013) provides a differentiable, efficiently computable density-spreading mechanism with controllable regularization strength, replacing ad-hoc smoothing in DREAMPlace. Second, the **Benamou-Brenier formulation** interprets global placement as a Wasserstein gradient flow: placement evolves along the geodesic in Wasserstein space from concentrated initial positions toward uniform density, with wirelength acting as potential energy -- giving a principled dynamics replacing heuristic annealing of density weights. Third, **semi-discrete OT** (discrete point masses transported to continuous target) naturally produces Laguerre tessellations (power diagrams) that partition the canvas into macro "territories" -- a placement-aware Voronoi decomposition encoding congestion information. The missing piece is **constrained OT preserving rectangular geometry and non-overlap** -- standard OT does not enforce that transported regions remain axis-aligned rectangles. Developing this "geometric OT" for placement would require integrating the disjunctive structure into the transport framework, potentially via multi-marginal OT formulations where each net's routing demand is a separate marginal.

**Information geometry.** The Fisher-Rao metric provides a principled optimization framework for the hybrid discrete-continuous structure. Muller & Montufar et al. (arXiv:2403.19448, 2024) proved Fisher-Rao gradient flows achieve **linear convergence for linear programs**, with rate depending on LP geometry. Parameterizing the probability distribution over disjunctive assignments as a product distribution p(sigma_{ij} = k | theta) for k in {L,R,A,B} gives a natural exponential family. The Fisher information matrix is block-diagonal (N(N-1)/2 blocks of 3x3), making the natural gradient tilde{nabla}theta = F(theta)^{-1} nabla_theta E[cost] computationally trivial. Adding KL regularization KL(p | q) yields a Boltzmann distribution p*(sigma) ~ q(sigma) exp(-beta * LP-cost(sigma)) -- reconnecting to the statistical physics picture and providing a principled annealing schedule.

**Disjunctive programming hierarchies.** The most mature applicable optimization theory comes from Balas's disjunctive programming. For the macro placement non-overlap constraints -- each a 4-term linear disjunction -- the convex hull of each pairwise disjunction can be computed in the extended variable space. A recent result (Springer LNCS, 2024) proves that the "full optimal big-M lifting" convex hull formulation is exact for axis-aligned rectangles when d <= 2, directly matching placement geometry.

The **P-split formulation hierarchy** (Kronqvist et al., *Mathematical Programming*, 2025) provides a principled middle ground between the weak Big-M relaxation and the expensive convex hull formulation. At level P=1, one recovers Big-M; at the top level, one recovers the convex hull. Computational results on 344 instances show that intermediate P-split levels achieve node counts comparable to the full convex hull while reducing solution time by an order of magnitude. For placement, this means one can progressively tighten the LP relaxation by applying sequential convexification to the most constraining macro pairs first.

The **SDP relaxation** of Anjos & Liers (2012) provides global bounds on VLSI floorplanning for up to ~100 facilities using mixed-integer SOCP-SDP formulations. Combining these with the Lasserre hierarchy's guaranteed convergence (O(1/r) rate, finite convergence generically per Nie 2014) and sparse variants like TSSOS that exploit pairwise constraint structure could yield certifiably optimal solutions for small-to-medium instances.

**Diffusion models.** Lee et al. (ICML 2025) replace sequential RL with simultaneous denoising of all macro positions, achieving competitive HPWL and >35% congestion reduction. DiffPlace (arXiv:2510.15897) adds constrained manifold diffusion to enforce non-overlap during denoising. DiffUCO (ICML 2024) applies diffusion to unsupervised combinatorial optimization. The theoretical gap is substantial -- no approximation guarantees exist -- but empirical results suggest diffusion processes can sample effectively from the multimodal union-of-polytopes landscape.

**Sequence pair representation.** The sequence pair (Murata et al., 1996) resolves the enumeration problem by construction -- each pair of permutations (Gamma+, Gamma-) maps to a feasible placement, giving (n!)^2 elements rather than 4^{n(n-1)/2} raw assignments. The adjacency graph on sequence pairs (via adjacent transpositions) is connected with diameter Theta(n^2), ensuring reachability. The practical path forward combines compact representations with the theoretical machinery above: tropical gradient descent or Fisher-Rao flows for continuous optimization within each polyhedron, survey propagation or diffusion for navigating between polyhedra, and disjunctive programming bounds for certification.

**Competition context.** The Partcl/HRT Macro Placement Challenge 2026 evaluates on 17 ICCAD04 IBM benchmarks (246-537 macros, 7K-16K nets, 43-53% area utilization) with proxy cost = 1.0 * WL + 0.5 * Density + 0.5 * Congestion, subject to a 1-hour runtime on an AMD EPYC 9655P with NVIDIA RTX 6000 Ada 48GB. The RePlAce analytical placer baseline dominates SA across all benchmarks (15-55% lower proxy cost). Top-7 submissions undergo full OpenROAD flow evaluation on NG45 designs.

The development time constraint (~50-60s per benchmark on M3 Max) rules out methods requiring thousands of function evaluations and demands that every LP solve be maximally informative. The hardest constraint is ~30 LP evaluations per benchmark -- this eliminates all sampling-heavy methods (Population Annealing, Discrete Schrodinger Bridges, neural diffusion) and points toward surrogate-model-based or dual-guided strategies as the only viable approaches at competition scale. The evaluation cascade (Section 5) addresses this directly by expanding the effective budget through cheap screening.

The competition also imposes a generality constraint: the placer must work well across all 17 IBM benchmarks (246-537 macros, varying connectivity structures) without per-benchmark tuning, and may be evaluated on hidden NG45 commercial designs. This rules out parameter tuning or benchmark-specific heuristics.


## 5. Computational Approaches

The theoretical frameworks in Sections 2-4 identify *what* to do (navigate through soft degrees of freedom at saddle points). This section addresses *how* to do it cheaply enough to fit within the ~30 LP-evaluation budget per benchmark. The core insight: most candidate assignments can be screened or bounded without full LP solves, expanding the effective search budget by orders of magnitude.


### 5a. LP Sensitivity Bounds (Miftari) --- TESTED, KILLED

> **Empirical status (Apr 16):** Tested on ibm01. Cheap dual signals predict
> LP-HPWL (rho=0.86, 42700x speedup), but LP-HPWL has **zero correlation with
> proxy cost** (rho=-0.001). The chain `cheap signal -> LP-HPWL -> proxy`
> breaks at the second link. See [results.md](results.md) "Miftari" section.
>
> This invalidates the entire class of LP-value-based cluster elimination:
> branch-and-bound with LP relaxation, Lagrangian bounds from duals, MCMC
> sampling weighted by LP cost, and partial-commitment LP bounding. All
> optimize a function (HPWL) uncorrelated with the actual objective (proxy =
> WL + 0.5*density + 0.5*congestion). Congestion is 66.5% of proxy cost and
> is a structural property of the topology, not the LP solution.
>
> Any viable bounding strategy must include density and congestion terms —
> which are non-convex (top-k order statistics) and not naturally available
> from LP solves. The incremental real-proxy evaluator ([evaluation.md](evaluation.md))
> is the only proposed path to cheap exact proxy signal.

The most directly applicable result for cheap evaluation is the 2024-2026 work of Miftari, Derval et al. (arXiv:2410.14443, v3 February 2026) on sensitivity analysis for linear modifications of the LP constraint matrix. Their framework studies f(lambda) = min c^T x subject to (A + lambda*D)x <= b, where D encodes constraint-matrix perturbations -- precisely what happens when switching a separation direction changes specific rows.

Their **linear Lagrangian bound** uses dual variables pi* from a solved LP to cheaply bound the optimal value as constraints change, requiring only matrix-vector products rather than full LP solves. Their companion paper (arXiv:2501.04151, January 2025) extends this with **Neumann series expansions**: if ||(lambda+delta)*E_B||_inf < 1, the basis inverse can be approximated and objective deviation bounded analytically.

**Practical recipe:** Solve one "center" LP per cluster, extract the optimal dual vector pi*, then for each nearby LP differing in k separation directions, check whether pi* remains dual feasible. If so, pi*^T b_new is a valid lower bound -- computed in microseconds. When the bound exceeds the best known solution, the entire cluster is pruned without any additional LP solve. The expected failure mode is basis instability: when switching a direction affects a tight (active) constraint, the bound becomes vacuous.

**Surrogate duality** (Muller et al., Mathematical Programming 2022) adds another layer: aggregating m constraints into K weighted constraints yields bounds strictly tighter than K=1 Lagrangian relaxation.

**Don't-care pairs reduce the effective search space.** LP sensitivity analysis via dual values identifies macro pairs whose assignment does not affect LP cost: pairs with zero dual values for all assignment constraints can be fixed arbitrarily, reducing the effective search space from 4^m to 4^{m'} where m' << m. This echoes "safe screening rules" from sparse optimization (Ndiaye et al., arXiv:1611.05780), adapted from eliminating LASSO features to eliminating combinatorial variables.

**Minimal viable experiment for LP sensitivity bounds:** Solve one LP. Record dual variables. For 1,000 nearby assignments (differing in 1-10 pairs), compute the Miftari linear Lagrangian bound and dual-feasibility-based bound. Compare both to actual LP solutions. This characterizes the "LP landscape smoothness" and directly reveals whether cluster pruning is viable.


### 5b. Ejection Chains

The core barrier -- that group flips of 200+ pairs create circular constraint chains causing LP infeasibility -- is precisely the problem ejection chains were designed to handle. Ejection chains (Glover & Rego, Annals of Operations Research 2010) create a reference structure that is *deliberately infeasible*, allowing a sequence of moves to propagate before feasibility is restored.

For the disjunctive assignment: starting from a "pivot pair" whose direction change has the largest potential impact, the change propagates through dependent pairs via BFS/DFS on the constraint graph. Each dependent pair's direction is flipped to maintain local consistency. At depths 1, 5, 10, 20, trial solutions are constructed by repairing unassigned pairs using fast constraint propagation, and the LP is evaluated only for the most promising trial. The key innovation over random multi-pair flips is **sequential construction**: rather than flipping 200 pairs simultaneously, the chain builds the new assignment incrementally, maintaining a feasibility-restoring escape route at every step.

**Adaptive Large Neighborhood Search (ALNS)** provides the complementary destroy-and-repair framework. ALNS maintains a portfolio of destroy and repair operators, adaptively selecting among them based on past performance. The BALANCE framework (Phan et al., 2023) uses Thompson sampling for adaptive operator selection, achieving 50% improvement in operator choice quality over static selection.

For macro placement, the "destroy" operator removes 5-20% of pair assignments (in a spatially coherent region identified by spectral clustering), and the "repair" operator uses constraint propagation plus a fast heuristic before LP warm-starting. Multiple destroy operators can be maintained: random destruction, worst-cost destruction (remove pairs contributing most to cost via LP duals), and cluster destruction (remove all pairs within a spectral cluster). The key practical insight across all methods in this category: **navigate through infeasible intermediate states using cheap heuristic evaluation, reserving LP solves for feasibility-restored trial solutions**.

The **STP decomposability theorem** (Dechter, Meiri & Pearl, AIJ 1991) provides essential support: for a fixed discrete topology, the continuous placement is exactly a Simple Temporal Problem -- a conjunction of difference constraints solvable in O(n^3) by shortest paths. The decomposability theorem guarantees that any partial assignment satisfying shortest-path constraints extends to a full solution, meaning topology can be built incrementally.

**Two-Dimensional Parallel Tempering** (arXiv:2506.14781, 2025) extends replica exchange along a second penalty-strength axis in addition to temperature, with feasibility-based swaps transferring configurations from soft-constraint replicas to hard-constraint ones. This directly addresses the constraint satisfaction barrier, demonstrating O(N^s) gap improvement in time-to-solution. The non-reversible DEO swap schedule of Syed et al. (arXiv:2102.07720, JRSS-B 2022) provides the optimal annealing protocol, with round-trip rate converging to (2+2*Lambda)^{-1}.

**Minimal viable experiment for ejection chains:** From the current local minimum, select the pair with highest LP sensitivity (from dual values). Flip its direction, propagate through 5-50 dependent pairs using BFS, repair via constraint propagation at each depth, evaluate LP at depths 1, 5, 10, 20, 50. Track the best trial solution versus depth. This directly tests whether sequential propagation avoids the infeasibility that simultaneous multi-pair flips create.


### 5c. Spectral Decomposition

The constraint adjacency graph is L(K_n) -- the line graph of K_n -- a strongly regular graph with parameters (n(n-1)/2, 2(n-2), n-2, 4). Its adjacency spectrum has exactly three distinct eigenvalues: **2n-4** (multiplicity 1), **n-4** (multiplicity n-1), and **-2** (multiplicity n(n-3)/2). The n-1 eigenvectors at eigenvalue n-4 encode "macro-level" coupling, while the overwhelming majority at -2 represent generic pair interactions. The Laplacian algebraic connectivity is n, and higher-order Cheeger inequalities (Lee, Oveis Gharan & Trevisan, 2014) confirm that the natural k-way partition has k <= n parts, corresponding to individual macros.

The practical value lies in the **weighted** constraint graph where weights reflect LP dual values and current assignment sensitivity. Computing the top-20 Laplacian eigenvectors (routine with ARPACK for 20K-144K vertices) reveals which groups of macro pairs can be flipped quasi-independently. Spectral clustering on this embedding identifies natural "flip neighborhoods" that minimize inter-group constraint coupling.

Practical expander decomposition algorithms (Gottesburen, Parotsidis & Probst Gutenberg, ESA 2024) identify bottleneck cuts in near-linear time O~(m). Applied to the effective constraint graph weighted by constraint tightness, these partition the problem into subproblems connected by narrow bottlenecks. The spectral gap of subgraphs induced by proposed flip sets directly predicts feasibility: if the gap drops below a threshold, circular constraint chains become likely.

**Matrix sketching** provides cheap LP bounds. Cekirge, Gay & Woodruff (APPROX/RANDOM 2025) give multipass linear sketches for LP-type problems with space polynomial in dimension and polylogarithmic in 1/epsilon. A Clarkson-Woodruff sparse sketch compresses the constraint matrix from O(n^3) rows to O(n^2 * polylog) rows, yielding (1+epsilon)-approximate LP bounds in time O(nnz(A)) -- seconds rather than the minutes of a full LP solve.

**Minimal viable experiment for spectral decomposition:** Compute the weighted Laplacian of the constraint graph using LP dual values as edge weights. Extract top-20 eigenvectors via ARPACK (routine for 20K-144K vertices). Apply k-means clustering on the spectral embedding. Test whether resulting clusters align with macro groups that can be optimized independently by solving separate LPs for each cluster. Compare total LP cost (sum of cluster LPs) to full LP cost. If the gap is small (<5%), hierarchical decomposition is viable and each cluster can be optimized independently, yielding massive speedups.

The spectral approach also connects to the soft-mode insight: eigenvectors corresponding to small Laplacian eigenvalues identify loosely-coupled macro groups whose internal pair assignments can change without significantly affecting other groups. These correspond to the "soft directions" for barrier crossing -- reorganizing assignments within a loosely-coupled cluster is less likely to create the cascading infeasibilities that prevent multi-pair flips.


### 5d. Benders Decomposition

**Logic-Based Benders Decomposition (LBBD)**, comprehensively treated in Hooker's 2024 Springer monograph, provides the principled framework for accumulating pruning information across evaluations. The key idea: decompose the problem into a master problem that selects L/R/A/B assignments and LP subproblems that optimize continuous positions given the assignments. Each LP subproblem generates Benders cuts -- valid inequalities in the assignment-variable space -- that prune future master problem solutions. After solving several subproblems, the accumulated cuts yield valid lower bounds for *any* assignment, including unvisited ones.

This is qualitatively different from surrogate models: Benders cuts are exact (provably valid bounds), not statistical estimates. Disjunctive Benders Decomposition (arXiv:2506.03561, June 2025) enhances this by generating valid inequalities for the convex hull of the Benders reformulation, tested on large-scale facility location instances.

**Backdoor variables** (Ganian, Ramanujan & Szeider, STACS 2017) bridge the gap between intractable full problems and tractable substructures. A strong backdoor set B is a subset of pair-assignment variables such that fixing B makes the remaining problem tractable. If |B| = 30-50 critical pairs can be identified (via LP dual sensitivity analysis or fractional relaxation), the residual constraint graph after conditioning on B may have bounded treewidth, enabling exact dynamic programming on the tree decomposition. The identification of backdoor variables connects directly to the soft-mode insight: pairs with large LP duals (frozen/backbone variables) are natural backdoor candidates, while pairs with small duals are the search space.

**AD^3** (Alternating Directions Dual Decomposition; Martins et al., JMLR 2015) offers the most scalable MAP inference framework. It decomposes the factor graph into overlapping tractable subproblems solved in parallel, with ADMM consensus at O(1/epsilon) convergence. Bethe-ADMM (Fu et al., arXiv:1309.6829) extends this to 7+ million variables. For n=537 macros with ~144,000 pairwise factor variables of domain size 4, AD^3 is well within computational reach. The main challenge is specifying factor potentials: these must approximate LP objective contributions for each pair assignment, estimable from dual-based bounds or surrogate models.

The FOCS 2024 breakthrough on efficient fractional hypertree width approximation (Korchemna et al., arXiv:2409.20172) makes practical hypertree decomposition feasible for the first time. While the full constraint hypergraph has prohibitive width (K_n has treewidth n-1), a sparsified backbone -- identified via spectral decomposition or LP sensitivity -- may have manageable width, enabling exact optimization on the backbone with heuristic extension to remaining pairs. This connects to the backdoor variable concept: the backbone pairs define the hard structure, and exact methods on the backbone combined with heuristic assignment of remaining pairs could yield provably good solutions.

**Graphs of Convex Sets** (GCS; Marcucci et al., SIAM J. Optimization 2024) formalizes exactly the "union of convex sets" structure from robotics motion planning. GCS produces a perspective-based convex relaxation that is empirically very tight -- often exact -- enabling solution via a single convex program followed by depth-first rounding. Direct application is impossible at our scale (4^{20,000} polyhedra vs. GCS's demonstrated 2,500 vertices), but the implicit search extensions GCS* (arXiv:2407.08848) and IxG (arXiv:2410.08909) interleave graph search with local convex optimization, exploring only a fraction of the graph.

**Dead-End Elimination** (DEE) from protein side-chain packing (OSPREY software, Donald Lab; Gainza et al. 2012) provides a complementary pruning paradigm. DEE proves that a rotamer r_i at position i can be eliminated if its minimum possible contribution (over all other positions' rotamers) exceeds another rotamer's maximum possible contribution. Adapted to macro placement: for pair (i,j), constraint direction d can be eliminated if choosing d always leads to higher LP cost than direction d', regardless of all other assignments.

Computing these bounds requires pairwise interaction estimates -- analogous to the precomputed residue-residue energies in protein design. The challenge is that protein interactions are local (each residue interacts with ~10-50 neighbors) while macro placement coupling is global through the LP. **Sparsification via spectral clustering** bridges this gap: after decomposing the constraint graph into weakly-coupled clusters, pairwise interaction bounds become computationally tractable within each cluster, enabling DEE-style pruning on a per-cluster basis.

**Complementarity formulation** from multi-body contact mechanics (Yang et al., IEEE Trans. Robotics 2024; Pang et al., MIT/Tedrake Lab 2023) represents non-overlap constraints as complementarity conditions (gap >= 0, force >= 0, product = 0). Interior-point methods for the resulting linear complementarity problem (LCP) implicitly navigate between discrete contact modes along a central path, providing smooth traversal of the discrete topology space. The LCP central path connects to the complexification framework: interior-point methods follow a path through the interior of the feasible region that smoothly transitions between polyhedra, avoiding the hard combinatorial boundary crossings that trap local search. The complementarity LP provides a heuristic for generating candidate discrete assignments: solve the smoothed problem, then snap the complementarity variables to discrete modes.

**Evaluation cascade architecture.** These computational approaches can be layered into a multi-fidelity cascade that screens candidates at sub-millisecond cost before committing to any LP solve:

- **Level 0 (nanoseconds)**: Zobrist hashing for duplicate detection. Assign 4 random 64-bit numbers per macro pair (one per direction). Hash of full assignment = XOR of all table entries. Incremental update on single-pair flip: one XOR operation. Maintain a transposition table of all evaluated assignments.

- **Level 1 (microseconds to milliseconds)**: Structural feasibility filters. Cycle detection in the induced precedence graph via Bellman-Ford in O(V+E). Edge-finding and not-first/not-last constraint propagation rules from disjunctive scheduling (Dorndorf, Pesch & Phan-Huy, AI 2000) in O(n^2). These immediately reject assignments creating infeasible precedence cycles without any LP computation.

- **Level 2 (microseconds)**: Dual-based cluster bounds (Miftari) plus surrogate model screening. Use the Lagrangian bounds from previously solved LPs. Supplement with a Random Forest surrogate (SMAC3-style) trained on all LP evaluations to date. Features include aggregate assignment statistics, graph-theoretic properties of the precedence graph, and spectral features from the constraint graph Laplacian. Reject candidates whose predicted LP cost exceeds the best known solution.

- **Level 3 (50-150ms)**: Partial LP evaluation. Solve LP with 10-20% of constraints (strategically selected), or run 10-50 simplex iterations with early termination. The constraint propagation-based LP bounding method of Dlask & Werner (JAIR 2024) provides an alternative: applying arc consistency to complementary slackness conditions yields LP bounds "often not far from the global optimum" at linear space complexity. Only candidates surviving this filter proceed to full LP.

- **Level 4 (seconds)**: Full LP solve with warm-starting from the nearest previously solved LP. Multi-fidelity Knowledge Gradient (Wu & Frazier, UAI 2019; implemented in BoTorch) selects which candidate to evaluate at which fidelity level, jointly optimizing information gain per unit cost.

This cascade expands the effective evaluation budget from ~30 full LP solves to thousands of screened candidates. The budget allocation: 5 random initial evaluations to train the surrogate, 5 partial LP evaluations to calibrate fidelity correlation, then 20 full LP evaluations guided by MFKG.

**Three insights cut across all computational approaches.**

First, **the evaluation bottleneck is solvable**: the Miftari dual bounds, constraint propagation filters, and partial LP fidelity levels can screen candidates 100-10,000x faster than full LP solves, expanding the effective evaluation budget from ~30 to thousands of candidates.

Second, **the infeasibility barrier is an engineering problem, not a theoretical one**: ejection chains and ALNS destroy-and-repair traverse infeasible intermediates by design, transforming the "wide barrier" from an obstacle into a structured search space.

Third, **the constraint graph's spectral structure provides the natural decomposition for free**: the L(K_n) eigenvalue spectrum analytically identifies macro-level groupings, while weighted spectral clustering reveals which flip neighborhoods are quasi-independent.

The most speculative ideas with strong mathematical connections are GCS perspective relaxation adapted to implicit graph search (from robotics, ~2 years from maturity for this scale) and DEE-style pruning of constraint directions (from protein design, requiring novel pairwise interaction bound computation). The most immediately implementable are Zobrist hashing + constraint propagation cascade (days), LP warm-starting with dual bounds (3-5 days), and ejection chains with spatially-coherent neighborhoods (1 week).


---

## See Also

- [approach.md](approach.md) -- what we do and why
- [roadmap.md](roadmap.md) -- implementation plan
- [problem.md](problem.md) -- formal problem statement
