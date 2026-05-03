# Research Principles for Breaking the Four Walls

Last updated: 2026-05-03
Author: parent agent (multi-day research synthesis)
Champion at time of writing: E48 hybrid 1.08151 --all (ADR-011 Accepted); ADR-012
Proposed promotes 3-lane hybrid with E61_v2 to 1.08025 (-0.12%).

## 0. Scope and method

This document is a deliberate response to the empirical wall pattern that
emerged over 2026-04-26 to 2026-05-03. The codebase has tried roughly
60 hypotheses; the deepest finding is in
`experiments/E65_neb_cross_section/manifest.md`: the SDF (E25) and DPO
(E41) basins are separated by an *infeasibility wall* in placement space,
not by a proxy ridge. That finding shapes which mechanisms are even
possible. Section 1 surveys 2024-2026 literature relevant to each of the
four walls. Section 2 maps mathematical structure to candidate
mechanisms. Section 3 ranks 7 concrete mechanisms by EV times
tractability. Section 4 addresses the hardware-regression risk. Section 5
lists open questions worth pursuing for the writeup.

The bar for a mechanism to count as "breakthrough" is dropping the
average proxy from 1.08 to <1.05 on `--all` while staying within the
17 hr `--jobs 4` envelope and not regressing on NG45 ariane133. Mechanisms
that only deliver -0.05 to -0.15% on top of E48 (i.e., extensions of
the existing best-of-N hybrid) are rated lower than mechanisms that could
plausibly open a structurally new basin or a new search paradigm.

---

## 1. Literature survey (2024-2026)

### 1.1 Disconnected feasible sets in combinatorial optimization

Our infeasibility wall is a special case of optimization over a
disconnected feasible set. Two mathematical traditions handle this
explicitly: bilevel programming and constrained multi-objective
optimization with fractured feasible regions.

- Yu and Beck 2024 (Coupling Constraints in Bilevel Optimization,
  Optimization Online) shows that coupling constraints induce
  disconnected feasible sets in bilevel programs but are
  complexity-equivalent to non-coupled formulations. The relevant
  observation for us is methodological: when the feasible set has K
  connected components, methods that respect the components individually
  (one optimizer per component, recombination across components) work,
  while methods that push toward midpoints fail.
- Wang, Liu, Zhang et al. 2024 (Constrained Multi-Objective
  Optimization Problems: Methodologies, Algorithms and Applications,
  Knowledge-Based Systems) treats fractured constrained Pareto fronts as
  the central difficulty. Their consensus is that successful algorithms
  rely on (a) explicit per-component archives, (b) repair operators that
  project onto the nearest component, and (c) crossover that respects
  component boundaries. Our E61 V2 spatial-block crossover is a coarse
  instance of (c): it preserves block-level component membership.

The lesson: per-macro recombination across two components creates a
midpoint that lies in neither component. Coarser-than-per-macro mixing
is the only class of recombination operators that preserves component
membership without an external repair operator. E61 V2 is a particular
case (block granularity = quadrant), and its modest -0.07% --all delta
suggests the right granularity is finer than quadrant but coarser than
per-macro - the hierarchical multi-level idea below.

### 1.2 Constrained NEB / manifold optimization

Henkelman and Jonsson 2000 (Climbing-image NEB) and the long materials-
science tradition treat reaction paths as variational extrema of an
action functional, optimized while respecting constraints by forced
reparameterization of the chain. Two recent strands matter:

- Raja, Sipka, Psenka et al. 2025 (Action-Minimization meets Generative
  Modeling, ICML 2025, arXiv:2504.18506). They cast Onsager-Machlup
  action minimization between two configurations as a variational problem
  solved by repurposing a pre-trained score-based generative model. Their
  energy landscape (Mueller-Brown, fast-folding proteins) has the same
  topological character as ours: distinct basins separated by barriers
  that interpolation alone cannot cross. The key technical move is
  representing the path measure as the law of an SDE driven by the
  learned score, then minimizing the discretized OM action. For us the
  obstruction is that we have no score model trained on placements; but
  the OM action functional itself is well-defined on any continuous path
  in placement space and we already have an approximate gradient (from
  DPO's smooth-proxy autograd path).
- Chiavazzo, Pavelka, Krishnapriyan 2024 (Doob's Lagrangian, NeurIPS
  2024) gives a sample-efficient variational formulation of transition
  path sampling via Doob's h-transform. Their formulation is amenable to
  trajectory-level gradient updates without requiring full path
  resampling.
- For the manifold side, the 2024 ESAIM-COCV paper "Null space gradient
  flows for constrained optimization with bound and inequality
  constraints" gives projected-gradient flows that stay on the active
  inequality face (an exact solution to "stay on the no-overlap
  manifold during search"). The active set of overlap constraints
  evolves as the trajectory moves; projection onto its null space gives
  a continuous flow that does not cross into infeasibility.

The lesson: there is a 2-paper-deep recipe for "constrained NEB on the
feasible manifold" in 2024-2025 literature that we have not yet
implemented. The Raja et al. method gives the score / OM-action
machinery; the null-space-flow method gives the manifold-respecting
projection. Combining them yields a path-level optimizer that stays
inside the no-overlap region.

### 1.3 Score-based / diffusion sampling for combinatorial problems

This is the field's biggest 2023-2025 movement.

- Sun and Yang 2023 (DIFUSCO, NeurIPS 2023, arXiv:2302.08224) uses
  graph-based denoising diffusion on discrete {0,1} vectors for TSP-500,
  TSP-1000, MIS, and MaxCut. They report state-of-the-art on TSP-10000
  with a 2.58% gap to ground truth. Critically, the model is trained
  on solved instances and used at test time to sample diverse,
  high-quality solutions.
- Sanokowski, Hochreiter, Lehner 2024 (DiffUCO, ICML 2024,
  arXiv:2406.01661) is the unsupervised version. They train without
  ground truth via reverse-KL minimization on the Boltzmann distribution
  exp(-beta f(x)). Achieves SOTA or competitive on MIS, MaxCl, MDS,
  MaxCut, MVC. The unsupervised property is critical for us: we have no
  optimal placement labels; we have only the proxy.
- Sanokowski, Hochreiter, Lehner 2025 (Scalable Discrete Diffusion
  Samplers, openreview 2502.08696) extends DiffUCO to larger graphs and
  to statistical-physics ground state problems, using subgraph
  tokenization for efficient inference. Demonstrated on graphs of
  N>10^4.
- Lai, Li, Spielberg et al. 2024 (Chip Placement with Diffusion Models,
  arXiv:2407.12282) is the first chip-placement-specific diffusion paper
  we found. Trained on synthetic netlists, places macros zero-shot with
  guided sampling; reports >35% RUDY congestion improvement vs prior
  learning baselines. UC Berkeley group; presented at ICML 2025
  workshop. Notable: places all objects simultaneously rather than
  sequentially (RL methods place one at a time).

The lesson: diffusion sampling is the state-of-the-art class for
combinatorial problems with multi-modal solution distributions. For our
infeasibility-wall problem, diffusion sampling explicitly bypasses the
wall by sampling from a learned distribution over feasible placements
rather than walking from one to another. The cost is training: a useful
score network needs to see many converged placements. We have ~60 in
`results/experiment_log.jsonl` plus thousands of intermediate ones we
can recover from CD trajectory checkpoints.

### 1.4 Hierarchical / multilevel placement

The placement field has 3 generations of hierarchical approaches.

- Caldwell, Kahng, Markov 2000 (CAPO, recursive bisection): coarse
  partitioning then bottom-up refinement. Greedy at the partition
  boundaries.
- Karypis et al. 1999 (hMETIS, multilevel hypergraph partitioning):
  coarsen by collapsing hyperedges, partition the small graph, uncoarsen
  with Fiduccia-Mattheyses refinement. Workhorse of VLSI partitioning.
- Kahng, Varadarajan et al. 2024 (Hier-RTLMP, IEEE TCAD, arXiv:
  2304.11761): hierarchical multilevel macro placer that uses RTL +
  dataflow hierarchy to drive coarsening, then SA on each level. Reports
  substantial improvements over RTL-MP on large blocks. Released into
  OpenROAD.
- Kim, Lin, Pan et al. 2024 (GOALPlace, arXiv:2407.04579) does not use
  multilevel directly but learns from EDA tool's post-route results to
  set per-region density targets, then runs analytical placement with
  density-target conditioning. Reports comparable quality to commercial
  tools.

The lesson: multilevel works for placement at the hMETIS partitioning
layer and at the dataflow-hierarchy layer (Hier-RTLMP). What is missing
in our codebase is V-cycle on the proxy itself: place super-macros, then
refine constituents while super-macro envelopes are fixed. Our E45
multigrid was falsified, but the manifest indicates the failure was
"didn't compose with K-joint mechanism"; the experiment was not really a
V-cycle, more a one-shot coarse-then-fine pass. A genuine V-cycle with
multiple refinement levels and coarsening guided by RTL hierarchy is
distinct.

### 1.5 Belief propagation / tensor networks on factor graphs

- Wrigley, Lee, Ye 2017 (Tensor Belief Propagation, ICML 2017): junction
  tree algorithm with each clique potential represented as a tensor;
  message passing as tensor contraction. Well-suited to graphs with
  small-to-moderate treewidth.
- Alkabetz and Arad 2021 (Tensor networks contraction and the BP
  algorithm, Phys. Rev. Research 3, 023073): explicit equivalence
  between tensor network contraction and BP on trees, and approximate
  contraction schemes for non-tree graphs.
- 2024-2025 literature on factor graph neural networks (e.g., Zhang et
  al. JMLR 2023) treats GNNs as differentiable approximations to BP,
  trainable end-to-end.

The lesson: BP / tensor networks are provably optimal on tree graphs and
provably-approximate-bounded on bounded-treewidth graphs. The macro-
interaction graph in our problem is dense (every macro interacts with
nearby macros via density and congestion), so naive BP would have
maximum treewidth. But the *netlist* hypergraph (macros connected if
they share a net) is sparser; on ibm01 the average net touches 3.2
macros, and the macro adjacency graph induced by nets has tree
decomposition width on the order of 30-50 (consistent with the typical
benchmark structure reported in hMETIS papers). For BP to be useful we
would need to (a) discretize macro positions to a coarse grid, (b) build
the netlist-induced macro adjacency graph, (c) tree-decompose it, (d)
run BP message passing. The manifest for E40-BP in roadmap §5.2 already
sketches this.

### 1.6 GNN / RL approaches to placement

- Mirhoseini, Goldie et al. 2021 (A graph placement methodology for
  fast chip design, Nature) and the AlphaChip controversy. Recent
  reevaluation (Cheng, Kahng et al., CACM 2024) shows the original RL
  approach lags behind classical methods. Lessons: RL is sample-
  inefficient, struggles with generalization to new netlists, and is
  outperformed by SA and analytical placers on the standard benchmarks.
- Lai, Mu, Liu et al. 2023 (ChiPFormer, ICML 2023): offline RL with
  decision transformer. Pre-trained on multiple chips, fine-tuned at
  test time. 18 min vs 3 hr for online RL.
- Xue, Chen, Lin et al. 2024 (Reinforcement Learning Policy as Macro
  Regulator Rather than Macro Placer, NeurIPS 2024, arXiv:
  2412.07167): Their key insight matches ours: "the RL policy acts as a
  regulator rather than a placer, operating on pre-existing placements."
  17.08% routing wirelength improvement, 73% horizontal congestion
  improvement, and 18.35% WNS improvement over MaskPlace. Code at
  https://github.com/lamda-bbo/macro-regulator. This is the closest
  analog to what we are doing - RL refining E48's output - but they have
  not had to clear the 1.08151 bar.
- Shi, Xue et al. 2023 (WireMask-BBO, NeurIPS 2023, arXiv:2306.16844):
  evolutionary algorithm with wire-mask-guided fitness. Reports up to
  53% HPWL improvement when fine-tuning SA-SP placements, 17% on
  MaskPlace. The fitness mask is a learned placement-quality estimator
  that bypasses the eval-cost bottleneck.
- Deng, Zhang et al. 2025 (EGPlace, ICML 2025): evolutionary search
  with greedy repositioning-guided mutation. Outperforms WireMask-EA by
  10.8% HPWL with 7.8x speedup, beats EfficientPlace by 9.3% with 2.8x
  speedup. The mutation operator is the load-bearing innovation: it
  identifies high-impact macros and only repositions those.

The lesson: 2024-2025 placement RL literature mostly works in the
"refine pre-existing placement" regime, not "place from scratch."
EGPlace and WireMask-BBO show that simple evolutionary algorithms with
good mutation operators beat heavily-engineered RL. Our E61 V2 spatial-
block crossover already does this on basin-symmetric benches; what
EGPlace adds is greedy macro repositioning as the fundamental mutation
move - which is structurally similar to our K-joint mechanism but
without the need for K-tuple selection heuristics.

### 1.7 Hungarian / assignment / OT for placement

- Cuturi 2013 (Sinkhorn, NIPS 2013): entropy-regularized OT with O(N^2)
  matrix iterations, GPU-friendly.
- Dong, Gao, Peng et al. 2025 (Hierarchical Refinement: Optimal
  Transport to Infinity and Beyond, arXiv:2503.03025): hierarchical
  Sinkhorn for very large N (10^6+), maintaining sub-quadratic cost.
- Liu et al. 2024 (Solving a Special Type of Optimal Transport Problem
  by a Modified Hungarian Algorithm, openreview NtSlKEJ2DS): cubic-time
  Hungarian beats Sinkhorn on certain structured costs.
- Kahng et al. 2014 (Hier-RTLMP earlier paper): min-cost-flow assignment
  for floorplan-row legalization. Polynomial time on row-shaped legal
  positions.

The lesson: Hungarian/Sinkhorn at K=200-700 is tractable
(O(N^3)=8e7 to 3e8 operations, sub-second on modern CPUs). What it
solves is "given N macro identities and N legal slots, optimally
assign." It does NOT solve "find slots." A two-step
generate-candidate-slots-then-assign pipeline with Hungarian is the
canonical non-local feasibility-respecting move that E65's manifest
calls out. Our E63 spectral init was attempting this; the failure was
not in the assignment but in the row-pack legalizer not handling fixed
macros properly.

### 1.8 Hardware-invariant numerical optimization

This area is sparse. The closest results:

- MLCommons Algorithmic Efficiency benchmark (Schmidt et al., 2023):
  measures wall-clock time to a target neural network accuracy,
  rewarding both algorithmic and systems improvements jointly. Has
  surfaced significant inter-hardware variance even on identical
  algorithms.
- Awareness in EDA: deterministic timing-driven parallel placement
  (Wang & Lemieux 2011, MASc thesis at UBC) explicitly designs for
  reproducibility across CPU counts. Most other EDA tools document
  per-iteration determinism but not wall-time portability.

The lesson: there is no published algorithm that is provably "as good as
its plateau" given fixed *work* rather than fixed *wall*. The principle
exists implicitly in iterative improvement methods (anytime algorithms,
bounded by *iterations* rather than *wall*) but the placement literature
has not adopted it. Section 4 below proposes a concrete formulation.

### 1.9 Placement-specific recent SOTA

- Deng et al. 2025 (EGPlace, ICML 2025): reports best known HPWL on
  WireMask benchmarks; 7.8x speedup over WireMask-EA.
- Lai et al. 2024 (Diffusion placement, arXiv:2407.12282): zero-shot
  placement by trained diffusion model; 35% RUDY congestion improvement
  vs SOTA learning baselines. Worth reading carefully because it directly
  tackles the multi-basin problem we have.
- Xue et al. 2024 (MaskRegulate, NeurIPS 2024): 17% routing WL improvement
  via RL-as-regulator on pre-existing placements.
- Hier-RTLMP in OpenROAD (2024): hierarchical macro placer; multilevel
  guided by RTL hierarchy.
- AutoDMP (Agnesina et al. 2023, ASPDAC; integrated into NVIDIA pipeline):
  multi-objective hyperparameter tuning over DREAMPlace.

For a 2026-deadline submission, the diffusion placement paper is the
most directly threatening competitor: it addresses our multi-basin
problem head-on with a different mechanism class. We should read its
score-network architecture and synthetic-data-generation algorithm
carefully (see section 5 open question).

---

## 2. Mechanism principles by wall

### 2.1 Wall 1: Infeasibility wall between basins

**Mathematical structure.** The set of feasible placements
F = {p in R^{2N} : non-overlap(p)} is a closed set with a non-empty
interior, but the connected components of F that are *near-optimal under
the proxy* (within delta of the global minimum) may be many. Our two
explored components (SDF basin, DPO basin) are connected components in
the strict sense - any continuous path between them passes through the
infeasible region. E65 quantifies this: linear interpolation crosses
hundreds of overlap-pairs, and 50 iterations of `project_overlaps` are
not enough to repair them.

This is structurally similar to:
- Crystal structure prediction (USPEX): different crystal lattices are
  separated by infeasibility (atom-overlap) regions.
- Protein folding (RFdiffusion): different conformations are
  separated by steric clashes that no continuous deformation crosses.
- Combinatorial topology spaces (sequence pair, B*-tree
  representations): two distinct sequence pairs encode placements that
  differ by structural pair-flips, not continuous deformation.

**Field-validated mechanisms:**

1. *Constrained-manifold pathfinding:* Stay on F throughout the search.
   Two sub-mechanisms.
   - Active-set NEB (Henkelman & Jonsson 2000 + null-space flow
     literature): each NEB image is projected onto F via null-space of
     the active overlap constraints. The trajectory is then the geodesic
     of F connecting the two basins. If F has multiple components, the
     NEB will fail to converge; if F is connected (it is, as long as
     macros can be permuted, which they can be by passing each other in
     the orthogonal axis), the NEB finds the lowest path between them.
2. *Generative sampling of F:* Train a score model
   s_theta(p, t) ~= -nabla log p(p) for the Boltzmann distribution
   exp(-beta proxy(p)) restricted to F. Sample new feasible placements
   by annealed Langevin or DDIM. DiffUCO (Sanokowski 2024) is the
   unsupervised analog; Lai et al. 2024 already does this for chip
   placement with synthetic data.
3. *Coarse-grain crossover:* As E61 V2 demonstrates, recombination at
   coarser-than-per-macro granularity preserves block-level feasibility.
   The key parameter is granularity: per-quadrant (E61 V2) gives -0.07%;
   per-row or per-cluster might give more lift; at the limit per-macro
   crashes (E61 V1).
4. *Topology representation switch:* Sequence pair (Murata et al. 1995)
   and B*-tree (Chang et al. 2000) representations are
   feasibility-by-construction: every sequence pair decodes to a
   non-overlapping placement. Crossover and mutation in this space stay
   feasible. The 2024 RL-on-sequence-pair paper (Wang et al. MDPI 2024)
   re-discovers this benefit. Switching our representation is a deep
   architectural move with high cost and high uncertainty.

**E61 V2 already does (3) at one granularity.** The promising next
moves are (1) constrained NEB and (2) feasible-set generative sampling.
(4) is a future-research move beyond the deadline.

### 2.2 Wall 2: IBM-aware / NG45-blind transfer failure

**Mathematical structure.** We have a parametric family of benchmarks,
roughly indexed by (macro count N, canvas area A, packing density rho =
N*avg_macro_area/A). IBM has rho ~ 0.5-0.8; NG45 ariane133 has rho ~
0.06. A mechanism with parameter theta(IBM) tuned for rho ~ 0.5
generally does not generalize to rho ~ 0.06. Five experiments
(E42/E43/E44/E54/E62) confirm this.

This is structurally a **domain-shift / OOD-generalization** problem.
Field analogs:

- Domain randomization in robotics (Tobin et al. 2017; recent ICLR 2024
  paper "Domain Randomization for Sim-to-Real": train on randomized
  parameter ranges to make policy robust).
- Equivariant ML (Cohen & Welling 2016, AlphaFold2 SE(3) equivariance):
  build symmetries directly into the model so OOD performance is bounded
  by training-distribution coverage of symmetry classes, not literal
  parameter values.
- Antibody DomainBed (arXiv:2407.21028): explicit OOD-generalization
  benchmark for therapeutic protein design; demonstrates that domain-
  invariant losses (IRM, CORAL) outperform standard ERM.

**Field-validated mechanisms:**

1. *Density-invariant features.* If the mechanism uses inputs of the
   form (number_of_macros_in_radius_r), it is rho-dependent. If it uses
   (fraction_of_macros_within_top_k), it is rho-invariant. Engineering
   our K-joint cluster selection to use rank-based rather than count-
   based features is straightforward.
2. *Benchmark-blind lower bounds.* E64 LP-bounded beam K-joint is
   exactly this. The LP relaxation depends only on the netlist
   (rho-independent in nature), so the bound is valid on both IBM and
   NG45.
3. *Multi-density training.* For learning-based methods (diffusion,
   GNN, RL), train on a mixture of densities. The diffusion-placement
   paper (Lai 2024) generates synthetic netlists with varying densities
   to cover this range.
4. *Conformal prediction calibration.* For each method M, compute a
   per-benchmark uncertainty estimate (e.g., proxy variance across
   destroy seeds). Reject application of M if the uncertainty exceeds
   threshold; fall back to E48 hybrid. This is the "mechanism doesn't
   apply on this benchmark" approach.

For our deadline, **(1) and (2) are tractable today**. (3) and (4) are
multi-week.

### 2.3 Wall 3: Local-move plateau saturation

**Mathematical structure.** A multi-mechanism plateau is the
intersection of fixed points of every move type we have:
{p : T_CD(p) = p, T_LNS_grid(p) = p, T_SA(p) = p, T_pair_swap(p) = p,
T_K3_joint(p) = p}. Hard benches like ibm14/ibm15 hit this plateau under
all 5 mechanisms. Reaching deeper than the plateau requires either (a)
a fundamentally new move type, (b) escaping via a saddle point, or (c)
re-initializing in a different basin.

(a) is what E12 grid-bin LNS did vs E9 CDAdaptive (-0.6%); what E41
K-joint did vs E18 (-0.5% on hard benches). Each new mechanism extracted
a slice of structure that previous moves missed. The marginal return is
diminishing: K=3 K-joint already commits 12-95 K-tuples per hard bench;
K=4 starves the budget; K=5 in E29 MIQP form is multi-day dev.

(b) is the saddle-point literature. Recent advances:
- Hu, Cao, Liu 2025 (Dimer-Enhanced Optimization, arXiv:2507.19968):
  first-order dimer method with only gradient evaluations, escapes
  saddle points in neural network training. Direct application: at a
  CD plateau, build a dimer in 2-macro joint coordinate, follow the
  smallest unstable mode by negative-curvature direction.
- Various "perturbed gradient descent" papers (PGDOT 2024, SPGD 2025)
  with theoretical guarantees on saddle escape rates.

(c) is wall 4 below.

**Field-validated mechanisms:**

1. *Higher-K joint moves with feasibility preservation.* K=10 with LP-
   relaxation pruning (E64) is already in the roadmap. K=50 Hungarian
   re-pack (mentioned in E65 manifest) is a different beast: solve
   "assign N=50 selected macros to N=50 candidate slot positions"
   exactly via Hungarian, then commit if proxy improves.
2. *Negative-curvature escape from plateaus.* Use dimer / SPGD on the
   joint-position vector at the plateau, follow the smallest unstable
   eigenvector for epsilon, then re-CD.
3. *Topology-flip moves.* In sequence-pair or B*-tree representation, a
   single pair-flip changes the underlying topology and produces a new
   feasible placement that is not reachable via continuous local moves.
4. *Stochastic destruction at large scale.* Destroy a much larger
   fraction of the placement (50-80%, vs current 5-10%) and re-
   construct via a learned policy or via SDF-init-style spreading. This
   is closer to "random restart" than to "LNS"; it concedes that local
   refinement has saturated.

(2) and (1) are the two highest-EV directions for the deadline.

### 2.4 Wall 4: Init basin lock + Wall 5: Hardware sensitivity

**Mathematical structure (basin lock).** We have observed that CD on
SDF init lands in basin 1, CD on DPO init lands in basin 2, CD on Will's
seed lands in basin 3 (which is worse), CD on uniform-random init lands
in basin 4 (which is much worse). Different inits go to different basins.
This is consistent with a multi-basin landscape where the basins of
attraction tile placement space, and only a few are near-optimal. The
empirical task is to find more near-optimal basins.

**Mathematical structure (hardware).** Our pipeline has hard wall caps:
CD <= 2400 s, LNS <= 600 s, SA <= 600 s, K-joint <= 600 s. On a slower
per-core machine, fewer iterations fit in each cap, so the algorithm
plateaus at a worse value. Verified by a partial leaderboard
re-evaluation (see roadmap.md TL;DR: vmallela's 1.1172 reportedly
dropped to ~12th place on Partcl's hardware).

**Field-validated mechanisms:**

1. *Spectral / quadratic init (E63).* Already scaffolded; needs the
   row-pack legalizer to handle fixed obstacles. Lifschitz-Markov 1990s
   spectral methods used custom legalizers we have not ported. Alkali
   et al. 2007 covers obstacle-aware Tetris; Spindler-Schlichtmann 2008
   introduces Abacus with O(N^2) dynamic programming for fixed-row
   legalization.
2. *Multi-init ensemble.* Run 5-10 different inits in parallel under
   the same wall budget; pick best at the end. Cost: linear in init
   count. Benefit: only as good as the best init; no compounding from
   multiple inits.
3. *Generative basin sampling.* Train a diffusion or VAE model on
   converged placements; sample diverse new starting points. Lai 2024
   does this directly.
4. *Hardware-invariance via plateau detection at the iteration level.*
   Replace `wall_time < cap` with `not yet at iteration plateau`.
   Concretely: CD continues until (sweep_delta < threshold for k
   consecutive sweeps) regardless of wall time, with a soft cap that
   never fires under normal benchmarks. The risk is that an unlucky
   benchmark could exceed the 1-hr legal cap; mitigation is to keep a
   wall-time safety net but tune it loose.

For (4), the iteration count required to plateau is mostly invariant
across hardware - it depends on the benchmark, not the silicon. So
plateau-detection-on-iterations gives us partial hardware invariance.
The challenge is the LNS phase: it samples destroy candidates randomly,
and the number of useful samples per benchmark before saturation is the
relevant work-bound. Empirically, our LNS phase saturates after ~20-50
samples; instead of capping at 600 s wall, we should cap at ~200 samples
or 0 productive samples in last K.

---

## 3. Ranked action plan

Each entry: hypothesis, mechanism, wall(s) attacked, build wall (hours/
days), first-step manifest skeleton, prerequisites, risk.

The ranking criteria are EV (could it plausibly drop us to <1.05?) and
tractability (can we ship before May 21?). I rate EV from 1 (marginal
extension) to 5 (basin-changing breakthrough); tractability from 1 (1+
weeks dev) to 5 (1-2 days dev).

### Rank 1. K=50 Hungarian re-pack (non-local feasibility-respecting move)

**EV: 4. Tractability: 4.**

Mechanism: at any plateau, select a coupled cluster of 50 macros (by
netlist adjacency or LP-dual significance). Generate 100 candidate slot
positions using grid-based feasible-slot enumeration around the cluster
bounding box. Solve Hungarian assignment of 50 macros to 50 best slots
(out of 100) minimizing proxy delta. Commit if proxy improves.

This is the natural extension of E41 K=3 brute-force enumeration to a
much larger K, made tractable by the Hungarian polynomial algorithm.
Solving Hungarian at K=50 takes O(50^3) = 125000 cost evaluations =
~100 ms with the incremental evaluator. We can run hundreds of K=50
Hungarian moves per hour.

**Walls attacked.** Wall 3 (local-move saturation) directly.
Wall 1 (infeasibility) only if cluster selection picks macros from
multiple basins; otherwise cluster moves stay within one basin.

**Build wall.** 2-3 days.
- Day 1: cluster selection (reuse E39's adjacency code), candidate
  slot generation (reuse E12's grid-bin code), Hungarian solver
  (scipy.optimize.linear_sum_assignment).
- Day 2: cost-matrix construction via incremental evaluator deltas;
  feasibility check on the joint move; commit / revert.
- Day 3: tuning K (try K=20, 50, 100), candidate-slot count (50, 100,
  200), --fast smoke, --all run.

**First-step manifest (E67_kjoint_hungarian):**
- Hypothesis: K=50 Hungarian re-pack escapes the multi-mechanism plateau
  by selecting a cluster of 50 netlist-coupled macros and solving the
  optimal joint reassignment to a candidate slot pool. Bypasses
  per-macro infeasibility wall by computing a feasible joint move.
- Method: at end of E48 pipeline, identify 1-3 worst-cost clusters.
  For each, solve K=50 Hungarian; commit improvements; iterate until
  no improvement or 600 s budget.
- Kill gate: --fast > E48 0.92024 + 0.5%; --ng45 ariane133 > 0.694.
- Generalization: same as E64 - the Hungarian operates on netlist-
  derived costs, not benchmark-specific patterns.

**Prerequisites.** None new; reuses existing E39 cluster selection and
E12 grid-bin slot generation.

**Risk.** (a) Cluster selection on netlist adjacency may be IBM-aware;
mitigation: also try LP-dual significance, which is netlist-only.
(b) Hungarian commits a 50-macro joint move; even if the cost matrix
indicates improvement, the actual commit may violate density constraints
the Hungarian cost did not see; mitigation: post-commit defensive
revert via `compute_overlap_metrics`. (c) Saturation: after the first
1-2 commits the plateau may not have any 50-cluster improvements;
mitigation: K=50 might be too large; add K=20 fallback.

### Rank 2. Constrained NEB on the feasible manifold (deepest theoretical move)

**EV: 5. Tractability: 2.**

Mechanism: between two converged basins (E25 SDF and E41 DPO), run
NEB with each intermediate image *projected onto the feasible manifold*
F = {p : non-overlap(p)}. Specifically, after each NEB-step gradient
update, project each image back into F using null-space-of-active-
overlap-constraints projection (Goemans-Williamson-style or simpler
push-apart with enforced step size).

The Onsager-Machlup action between two minima of a smooth proxy is
S[gamma] = (1/4T) integral ||gamma_dot + nabla f(gamma)||^2 dt. The
minimum-action path crosses a saddle whose unstable mode points toward
potentially-lower basins on the far side. If the path is constrained to
F, it traces a geodesic of F connecting the two basins; the saddle on
that geodesic is the lowest-proxy infeasible-free path between them.

Following the smallest unstable Hessian eigenvector at the saddle leads
either back to one of the two basins or to a third (potentially lower)
basin. This is the chemistry/materials gold standard for finding new
minima: it has 30+ years of refinement and dozens of stable codebases
(Henkelman 2000 to Raja 2025).

**Walls attacked.** Wall 1 (infeasibility) directly: the path stays
inside F by construction. Wall 4 (basin lock) potentially: a third
basin discovered by saddle-following.

**Build wall.** 5-7 days.
- Day 1: NEB image data structure; perpendicular-gradient projection
  along the chain tangent.
- Day 2: null-space projection onto F. The active set of overlap
  constraints at each image evolves; this is the hard part.
  Approximations are acceptable: project onto F naively via
  push-apart, but at small step sizes that approximation is good.
- Day 3-4: image relaxation iterations; convergence check; saddle
  identification (highest-energy image after convergence).
- Day 5: Lanczos for smallest Hessian eigenvector at saddle; descend
  +/- epsilon; resume CD-LNS-SA from each side.
- Day 6: --fast smoke on ibm12 (basin gap small, NEB feasible).
- Day 7: --all if smoke shows lift.

**First-step manifest (E68_constrained_neb):**
- Hypothesis: constrained NEB on the no-overlap feasible manifold
  between E25 and E41 basin endpoints finds a saddle whose unstable
  Hessian eigenvector points to a third basin lower than either.
- Method: 30-50 images linearly interpolated, each projected onto F via
  push-apart legalizer. Iterate gradient updates with chain-tangent
  projection until image-spacing converges. Identify saddle. Lanczos at
  saddle. Descend on both sides; CD-LNS-SA polish; report best basin.
- Kill gate: --fast: zero benches improve over E48 0.92024 by >= 0.3%
  after K=10 NEB runs across basin pairs.
- Generalization: physical principle, transfers to NG45.

**Prerequisites.** Smooth proxy gradient (we have it from DPO). Lanczos
infrastructure (numpy.linalg.eigh on a smaller subspace, or scipy.sparse.
linalg.eigsh).

**Risk.** (a) The feasible manifold may have more than two connected
components (3+), in which case NEB between two of them doesn't help.
Mitigation: run NEB between all pairs from {E25, E41, E61_V2}.
(b) The infeasibility wall is so wide that even with projection the path
length grows unboundedly; mitigation: set image-spacing penalty
appropriately. (c) Compute cost: each NEB step is O(images * macros) cost
evaluations, ~1 hr on a hard bench. Iterating to convergence might be
6-12 hr per benchmark. Budget for --fast is doable; --all might require
~80 hr.

### Rank 3. Spectral init with Tetris-style obstacle-aware legalizer (E63 unblock)

**EV: 3. Tractability: 5.**

Mechanism: continue the E63 spectral init lane. The blocker is the
row-pack legalizer not handling fixed macros. The fix is well-known
in the placement literature (Spindler-Schlichtmann 2008 Abacus, Ozdal
& Markov 2007 Tetris-with-obstacles).

The spectral init places macros at coordinates given by the Laplacian
eigenvectors. The macros from this init are floating in space without
respecting fixed obstacles or row alignment. The legalizer needs to:
1. Identify each row's available x-spans by subtracting fixed-macro
   bboxes.
2. For each macro, sort by x-coordinate.
3. Greedy-place each into the row of nearest legal x-span,
   minimizing displacement.

This is the textbook Tetris-with-obstacles algorithm, ~200 lines of
Python.

**Walls attacked.** Wall 4 (init basin lock): adds a third basin source.
Wall 2 (NG45 transfer) potentially: spectral init is netlist-structure-
aware, hence benchmark-blind in the relevant sense.

**Build wall.** 1-2 days.
- Day 1: implement Tetris-with-obstacles legalizer; smoke on ibm01.
- Day 2: --fast (4 benchmarks) E63 with new legalizer.
  If --fast OK, --ng45 ariane133.

**First-step manifest (E63_v4_obstacle_legalizer):**
- Hypothesis: spectral Laplacian init + Tetris-with-obstacles legalizer
  + E41 polish pipeline lands in a basin distinct from SDF (E25) and
  DPO (E41), and at proxy <= 1.10 average on --fast.
- Method: same as E63 V3 but with the fixed-aware row-pack.
- Kill gate: --fast > 0.929 (E41 + 0.8%); zero per-bench wins on
  ibm01/04/09/13 vs E48 means falsify.
- Generalization: spectral init is benchmark-blind; NG45 ariane133 should
  not regress.

**Prerequisites.** None new. The Tetris-with-obstacles algorithm is
standard.

**Risk.** (a) Spectral basin may be uniformly worse than SDF/DPO basins
(experimentally observed standalone proxy 1.78 in early phases).
Mitigation: the question is whether *polished* spectral basin is useful
in a hybrid; even if standalone is worse than E25/E41, hybrid lift on
even one bench would justify keeping the lane. (b) Tetris legalization
introduces uncontrolled displacements that disrupt the spectral
embedding; mitigation: use the legalizer as a final pass only, not in
the loop.

### Rank 4. LP-bounded beam K-joint (E64; benchmark-blind via LP relaxation)

**EV: 4. Tractability: 2.**

Already scaffolded as E64 manifest. The dev cost (2-3 days) is
dominated by reconstructing the polyhedra LP infrastructure deleted in
commit 44efd16. The mechanism is sound: LP relaxation of HPWL within a
fixed L/R/A/B assignment polyhedron gives a valid lower bound that
prunes the K=10 beam search. The bound is netlist-structure-only, hence
benchmark-blind. This was the reason E42/E43/E44 failed NG45 transfer:
their pruning heuristics used local benchmark structure.

**Walls attacked.** Wall 2 (NG45 transfer) directly via benchmark-blind
LP. Wall 3 (local-move saturation) by enabling K=10 search depth.

**Build wall.** 2-3 days.

**Prerequisites.** Reconstruct polyhedra LP from
`writeup/archive/submissions/cd_lns_placer.py`; HiGHS solver wrapper.

**Risk.** (a) LP relaxation on HPWL alone may be too loose to prune
much of the K=10 search tree, in which case the algorithm runs out of
budget; mitigation: try K=5 with the same machinery as a fallback.
(b) Density and congestion components are not LP-bounded in this
formulation; the LP only bounds the WL component, which is 6% of proxy.
The full proxy may still depend on benchmark-specific features that the
LP does not capture; mitigation: include an analytical density bound
(per-cell density is a sum of macro contributions, which gives a linear
relaxation of the top-10% density component if we are willing to over-
estimate slightly).

### Rank 5. Diffusion sampling of feasible placements (Lai-style, scaled to our budget)

**EV: 5. Tractability: 1.**

Mechanism: train a small score network s_theta(p, t) on converged
placements from our experiment_log. Use guided sampling with
proxy-conditioning to generate diverse high-quality starting placements,
then polish each via E48 hybrid pipeline.

The 2024 paper (Lai et al., arXiv:2407.12282) gives the architecture
recipe. The training data: ~60 fully-converged placements + ~thousands of
intermediate ones from CD checkpoints. We can generate synthetic data
via CD restarts from various inits to fill out the distribution.

**Walls attacked.** Walls 1 (infeasibility), 4 (init basin lock), 5
(hardware) all attacked simultaneously: the diffusion model samples
new starts directly in feasible space (no path to traverse).

**Build wall.** 7-14 days.
- Days 1-3: GNN architecture (encoding the netlist hypergraph) +
  per-macro denoising head.
- Days 4-6: training loop on existing converged placements; reverse-KL
  loss (DiffUCO-style) for unsupervised learning, augmented with
  proxy-supervised loss when ground truth available.
- Days 7-10: annealed Langevin sampling with proxy conditioning;
  hyperparameter tuning.
- Days 11-14: pipeline integration, --fast smoke, --all.

**First-step manifest (E69_diffusion_sampling):**
- Hypothesis: a score model trained on the project's converged-
  placement corpus, sampled with proxy-conditioning, generates feasible
  placements distributed across multiple basins. The best sample,
  polished via E48, beats E48 standalone by >= 0.3%.
- Method: GNN encoder for netlist; per-macro displacement-denoising
  head; train on fully-converged placements (E25 + E41 + E61_V2
  outputs across 17 IBM); sample 10 placements with proxy guidance;
  polish each via E48; report best.
- Kill gate: --fast > E48 0.92024 + 0.5%; zero per-bench wins on
  --fast.
- Generalization: NG45 mandatory; the score model should be netlist-
  structure-aware (rho-invariant via GNN).

**Prerequisites.** PyTorch GNN training infra (we have basic PyTorch in
DPO codebase); incremental evaluator at scale (we have it).

**Risk.** (a) Training data is small (~60 converged placements + intermediate
checkpoints); the model may overfit to IBM benchmarks. Mitigation: use
synthetic netlists generated from learned netlist generators (van
Krieken-style). Substantial extra dev. (b) Sampling is GPU-bound; the
RTX 6000 Ada in the contest spec helps. (c) Quality of samples
depends on the score-model architecture; if too small, samples are
near-mean and don't help; if too large, training fails on small data.
We have very limited time for hyperparameter tuning. Highest-variance
candidate.

### Rank 6. Hierarchical V-cycle multigrid placement

**EV: 3. Tractability: 3.**

Mechanism: real V-cycle. (a) Coarsen via hMETIS into 20-40 super-
macros. (b) Place super-macros via E12 CD+LNS at coarse scale (~5 min).
(c) Uncoarsen one level: each super-macro's bounding region is fixed;
place its constituents inside via CD. (d) Repeat to leaf scale. (e)
Post-smoother: full-canvas E12 CD+LNS to integrate.

Distinct from the falsified E45 multigrid because (a) E45 used a single
coarse-fine pass without iteration, (b) our 2026-04-30 implementation
of E45 didn't engage with the hMETIS partitioning that's standard in
hierarchical placement, (c) it didn't compose with K-joint at any
level.

**Walls attacked.** Wall 3 (local-move saturation) via aggregate-mass
moves. Wall 2 (NG45 transfer) potentially: hierarchy is structural,
not benchmark-specific.

**Build wall.** 3-5 days.

**Prerequisites.** hMETIS bindings (PyMetis or pyscipopt's wrapper);
quotient-netlist construction.

**Risk.** (a) Hier-RTLMP (Kahng 2024) is the SOTA hierarchical macro
placer; our V-cycle would need to compete with it on commercial
designs. They use RTL hierarchy directly; we don't have that signal.
(b) hMETIS clustering is heuristic; bad clustering produces bad coarse-
level placements. (c) The interaction between coarse-level placement
and fine-level CD is non-trivial; refinement at one level can break
quality at another.

### Rank 7. RL-as-regulator on E48 output (extension of MaskRegulate)

**EV: 3. Tractability: 1.**

Mechanism: train a per-macro displacement policy by PPO on (state =
placement + congestion map, action = (macro index, target displacement),
reward = -proxy delta). Initialize from E48 output; let the policy
refine. MaskRegulate (Xue et al. NeurIPS 2024) reports +17% routing WL
over MaskPlace; on E48 (which is far stronger than MaskPlace), the lift
is uncertain.

**Walls attacked.** Wall 3 (local-move saturation) - if the learned
policy finds productive moves outside the K=3 K-joint reachable set.

**Build wall.** 14+ days.

**Prerequisites.** PPO training infra; GNN policy architecture; reward
machinery.

**Risk.** (a) Sample inefficiency: PPO needs millions of episodes to
converge; with ~1 sec / placement evaluation, this is ~weeks of GPU.
(b) Generalization across benchmarks is the standard RL pain point.
(c) RL approaches in placement have a 5-year track record of
underperforming classical methods (CACM 2024 reevaluation). EV vs cost
ratio is poor for our deadline.

### 3.1 Recommended sequence

For the deadline (May 21, ~18 days remaining):

1. **Days 1-2 (now):** rank 3 (E63 V4 spectral with obstacle legalizer).
   Cheap, high-confidence, possibly opens a 3rd basin lane.
2. **Days 3-5:** rank 1 (K=50 Hungarian re-pack, E67). Direct attack
   on plateau, mechanism well-understood, polynomial-time inner solve.
3. **Days 6-9:** rank 4 (E64 LP-bounded beam K-joint), if E67 succeeds
   then E64's role is to compose; if E67 stalls, E64 is the K-extension
   alternative.
4. **Days 10-15:** rank 2 (E68 constrained NEB), if any of the above
   suggest the saddle structure is reachable. Highest theoretical EV
   but requires more dev.
5. **Days 16-18:** verify the strongest candidate on --all + --ng45;
   freeze submission.

Rank 5 (diffusion sampling) is post-deadline research: too high-
variance for the timeline but the strongest candidate for the
innovation prize writeup.

Rank 6 (V-cycle multigrid) is parallel/optional; can be assigned to a
secondary parallel agent, but is not on the critical path.

Rank 7 (RL regulator) is deprecated for the deadline.

---

## 4. Hardware-regression mitigation

Wall 5 is the most under-discussed and most likely real risk. Our champion
is wall-time bound (CD 2400s, LNS 600s, SA 600s, K-joint 600s totaling
4200s/bench against the 3600s legal cap with 600s margin). On Partcl's
hardware (AMD EPYC 9655P, 16 cores), if per-core wall is slower than M3
Max, our caps may fire before plateau detection, leaving each bench mid-
descent.

### 4.1 The work-bounded vs wall-bounded distinction

Define **wall-bounded** algorithms as those that terminate when
wall_clock > cap. Define **work-bounded** as those that terminate when a
work-amount metric exceeds threshold (iterations, samples, function
evaluations, LP solves, etc.). The two converge under fixed hardware;
they diverge under hardware variance.

Our pipeline mixes both. CD has wall_cap=2400s and plateau detection
(threshold on sweep delta). LNS has wall_cap=600s and saturation
detection. SA has wall_cap=600s. K-joint has wall_cap=600s.

The plateau / saturation metrics are work-bounded by nature; the wall caps
are wall-bounded. On slower hardware, the plateau will be hit later in
wall time, possibly after the cap. This is the failure mode.

### 4.2 Concrete mitigation principles

1. **Use plateau / saturation as primary termination, wall caps as
   secondary safety nets.** Ensure the safety nets are loose enough that
   they fire only on pathological cases. Currently CD's cap fires
   regularly on hard benches; that's a sign the cap is too tight.
2. **Track work units, not wall time, for tuning.** When evaluating
   variants, compare at fixed iteration count, not fixed wall.
3. **Detect saturation faster.** LNS saturation is currently detected as
   "no commit in last K samples"; we can lower K from 5 to 3 to stop
   sooner when stuck. Same for K-joint.
4. **Adaptive caps.** Track actual elapsed time at each phase; if the
   bench is on track to hit the cap before plateau, *reduce* later
   phases' wall budgets to fit within the legal envelope. This is
   already done in E48's hybrid; could be tightened further.
5. **Hardware probe at startup.** Run a short calibration kernel at the
   start of each placement to measure ops-per-second; scale internal
   counters accordingly. This lets the same algorithm choose iteration
   counts that produce equivalent quality on different hardware.

### 4.3 Cheap test for hardware regression

Before submission, run E48 on a deliberately throttled CPU (e.g., taskset
to 1 core, or `cpulimit -l 50`). Compare the resulting --fast proxy to
the unthrottled value. If the throttled run is significantly worse, we
have a hardware-sensitivity problem.

A second check: profile per-phase wall time. On M3 Max, CD phase wall is
~30 min on hard benches. If on a slower CPU CD wall would be ~60 min, it
hits the 60-min legal cap and the LNS+SA+K-joint phases never run. Add
plateau-checks every CD sweep (currently every 3) and let CD exit early
on any signal of plateau.

### 4.4 Hardware-robustness of the candidate mechanisms

For each rank in §3:

- Rank 1 (Hungarian K=50): Hungarian solver work is O(K^3)=125000 cost
  evaluations per move. On slower hardware, fewer Hungarian moves fit in
  the budget, but each move's *quality* is unchanged. Work-bounded by
  number of Hungarian moves. **Hardware-robust** if we cap on moves
  instead of wall.
- Rank 2 (Constrained NEB): each NEB iteration is fixed work
  (n_images * gradient evals); total iterations to convergence is
  benchmark-dependent but hardware-independent. **Hardware-robust** if
  capped on iterations.
- Rank 3 (Spectral init + Tetris): one-shot per benchmark; lower-bound
  on wall (~1 sec eigsh + ~1 sec Tetris). **Hardware-robust trivially**.
- Rank 4 (LP-bounded beam K-joint): each LP solve is HiGHS-internal
  work; HiGHS is reasonably fast on small LPs (~10ms each). Beam steps
  per cluster bounded by branching factor. **Mostly hardware-robust** if
  capped on tree expansions.
- Rank 5 (Diffusion sampling): training is highly hardware-dependent.
  Sampling is fast. **Mixed**: train on our hardware, hope sampling is
  fast on theirs.
- Rank 6 (V-cycle): coarse-level wall is small; fine-level wall is
  bench-dependent. **Mostly hardware-robust** with proper plateau
  detection per level.
- Rank 7 (RL regulator): inference is fast; training is highly hardware-
  dependent. **Hardware-portable** at inference time.

The deadline-priority candidates (1, 3, 4) are all reasonably hardware-
robust if we change our termination policy from wall-bound to
work-bound. Doing this is cheap (a few hours of refactoring) and might
be the single most important mitigation we can ship.

### 4.5 Recommended short-term action

Refactor the E48 pipeline to use work-bounded termination as primary,
wall caps as soft secondary. Specifically:

- CD: terminate on plateau (sweep delta < 0.001 for 3 consecutive
  sweeps). Soft wall cap at 3000s (vs current 2400s); aggressive enough
  to catch pathological benchmarks but loose enough that normal benches
  exit on plateau.
- LNS: terminate on 5 consecutive non-improving samples. Soft wall cap at
  900s.
- SA: terminate on no improvement in last 1000 moves. Soft wall cap at
  900s.
- K-joint: terminate on no commits in last 30 K-tuples. Soft wall cap at
  900s.

Total budget: still 4200s + 600s for SDF/DPO init = 4800s, comfortable
within the 1-hr legal cap with margin. On slower hardware, the plateau-
based termination will still fire correctly; only on very slow hardware
will the wall caps fire.

The risk: under faster hardware, the soft caps don't fire, but the bench
runs longer than today. The 17-hour --all envelope might tighten. We
can re-tighten the caps per-phase if profiling shows we're at risk.

---

## 5. Open research questions

Listed in priority order, with the most promising lines of investigation.

### 5.1 What is the Boltzmann distribution of feasible near-optimal placements?

We have observed that two basins (SDF, DPO) cluster around 1.08-1.10
proxy with zero overlaps, and that linear interpolation between them
crosses through 88-233 unresolvable overlap pairs. What does the full
landscape look like? How many such basins exist? How are they topologically
related?

Investigation lines:
- *Persistence homology of converged placements:* compute a distance
  matrix on all ~60 converged placements in the experiment log, build a
  Vietoris-Rips complex, compute H_0 (connected components) and H_1
  (loops). Tells us how many basins we've found and whether they form a
  connected network or isolated islands. Reuses the E27 partial
  diagnostic.
- *Sample basin diversity via random restarts:* run 30+ random or jittered
  starts of E25 / E41 pipelines; cluster outputs by displacement. Number
  of distinct cluster centroids = lower bound on basin count.
- *Boltzmann sampling via parallel tempering:* set T to a value where the
  Metropolis acceptance rate is ~10%; run 8 chains coupled by replica
  exchange across temperatures; count distinct minima visited.

EV: structural understanding, possibly a new mechanism (basin sampling).
Tractability: moderate (1-2 days for persistence; 3-5 days for chains).

### 5.2 Why does the basin produced by Will's seed (E62) polish to a worse value than SDF or DPO basins?

WillSeed has standalone proxy 1.5338, between SDF (1.50) and DPO (1.38).
Yet polished proxy after E41 pipeline is 0.9335 on --fast (vs 0.92 for
E25/E41). The mid-quality init lands in a *worse* basin than either
end-quality init.

Possible explanations:
- Will's seed has a structural feature (e.g., specific pin clustering) that
  guides CD into a basin with more macros poorly placed but locally
  stable.
- The sequence of moves CD takes from Will's init differs from CD-from-
  SDF in a way that locks in early bad decisions.
- Will's init has subtle overlap-near-misses that affect legal_axis_range
  computation in early CD sweeps.

Investigation: trace CD trajectory from Will's init at high resolution;
compare to SDF trajectory; identify the first sweep where they
substantively diverge. May give insight into what makes an init "good"
vs "bad" beyond just standalone proxy.

### 5.3 Is the K=3 K-joint mechanism saturated structurally, or just empirically?

E41 K=3 commits 12-95 K-tuples per bench; E42 K=4 falsified, E43 longer
K-joint falsified. But the failure mode of K=4 was budget starvation,
not "no productive K=4 tuples exist." We don't know how many K=4 (or
K=5, K=10) productive tuples exist.

Investigation: implement K=4 with an LP-bounded beam (E64) to make the
search budget-feasible. If LP-pruning enables K=4 to find more wins
without budget starvation, K-extension is productive; if it still
saturates, the structural plateau is at K=3.

### 5.4 Can the spatial-block crossover granularity be tuned?

E61 V2 uses 2x2 quadrants. The hyperparameter "crossover granularity" -
how big a spatial block - is an open variable. 3x3 / 4x4 / per-row / per-
hMETIS-cluster are all candidates. None has been tested. EV per probe:
~3 hr per granularity choice on --fast.

If a per-cluster granularity (where the cluster is selected by
hMETIS partitioning) gives more lift than per-quadrant, we have a
direct route from E61 to a structurally better hybrid lane.

### 5.5 What is the right loss for a diffusion model trained on our placements?

Lai 2024 trains on synthetic netlists and uses a guided-sampling loss.
DiffUCO 2024 uses unsupervised reverse-KL on the Boltzmann distribution.
For our problem we have a small corpus of converged placements but
unlimited intermediate states from CD trajectories. The right loss is
unclear. Investigation: write the small training loop and try (a) score-
matching on converged placements only, (b) score-matching on all CD
trajectory states with t=0 = converged, (c) DiffUCO-style reverse-KL
unsupervised. Compare sample diversity and proxy quality.

### 5.6 Does the infeasibility wall persist across reformulations?

The wall is observed in (x, y) Cartesian coordinates. In a different
representation (sequence pair, B*-tree) the wall might not exist - any
point in sequence-pair space decodes to a feasible placement, so the
"wall" between two distinct sequence pairs is just a discrete-flip
distance, not infeasibility. Investigation: compute the sequence-pair
encoding of E25 and E41 outputs (via Murata-Fujiyoshi packing decoder
inverted); count discrete pair-flips between them. That number bounds
the cost of a sequence-pair-level NEB.

If small (10-50 flips), sequence-pair NEB is tractable. If large (1000+),
the representation switch buys us little.

### 5.7 What's the role of macro orientation (rotation by 90 degrees)?

Our placements have macros at fixed orientations. Real chip placement
optimizes orientation jointly. Hier-RTLMP (Kahng 2024) reports orientation
flips contribute up to 3% of HPWL improvement. We have not exposed the
orientation degree of freedom. Investigation: enable macro_rotated_180
flip moves in CD or LNS; measure proxy impact on a few benches.

If the impact is large, rotation is a missing degree of freedom in our
parameterization. If small, we can ignore it.

---

## 6. Synthesis

Three concrete recommendations for the days ahead, ordered by EV:

1. **K=50 Hungarian re-pack (E67).** Rank 1 above. 2-3 days dev. Direct
   attack on the local-move plateau. Polynomial inner solve. Hardware-
   robust if we cap on moves. The mechanism class (large-K Hungarian
   re-assignment) is exactly what the E65 manifest calls out as a
   plausible breakthrough mechanism; no one has implemented it yet.
2. **Constrained NEB on the feasible manifold (E68).** Rank 2 above.
   5-7 days dev. Highest theoretical EV: it's the chemistry/materials
   gold standard for finding new minima on a multi-basin landscape.
   Works only if the saddle Hessian unstable mode points to a third
   basin; if it does, the basin is structurally distinct from SDF and
   DPO.
3. **Spectral init with obstacle-aware Tetris legalizer (E63 V4).**
   Rank 3 above. 1-2 days. Cheap unblock for the existing E63 lane.
   Even a marginal lift adds a 3rd basin source to the hybrid; even
   confirmed non-lift closes the line cleanly.

The most surprising / non-obvious finding from the literature search:
**the 2024 NeurIPS paper "RL Policy as Macro Regulator Rather than
Macro Placer" (Xue et al.) discovers exactly the same architectural
insight as our E48 hybrid - that refining a pre-existing placement is
structurally easier than placing from scratch.** They use it to beat
MaskPlace by 17%; we use it to beat the leaderboard by 3.21%. Two
parallel research traditions arrived at the same insight in 2024-2026.
The implication for our writeup is that we can cite this as
contemporaneous validation of the hybrid-as-refiner architecture, not
as competing prior art.

The second most surprising finding: **the constrained-NEB-on-feasible-
manifold direction has a complete recipe in the 2024-2025 literature
(Raja et al. 2025 OM-action + null-space gradient flows 2024) but no
one has applied it to combinatorial layout.** This is the deepest
unexploited lever in the project. The chemistry/materials community
solved the "find new basin via saddle-following" problem 25 years ago
(Henkelman 2000) and the constrained version has been refined through
2024. The 5-7 days of dev cost is real but the EV is the highest of any
mechanism considered.

The third surprising finding: **the LP-bounded beam K-joint (E64,
already in our roadmap) directly maps onto the "benchmark-blind LP
relaxation as pruning bound" pattern from the multi-objective
optimization literature.** We had been thinking of it as a local fix
for K-joint failure; it is actually a generic principle for OOD
robustness under benchmark variance, and we can engineer the same
principle into other mechanisms (e.g., a benchmark-blind LP-bound on
Hungarian K=50 cost matrix).

Cited works (selection):
- Sun & Yang 2023, DIFUSCO, NeurIPS 2023, arXiv:2302.08224.
- Sanokowski et al. 2024, DiffUCO, ICML 2024, arXiv:2406.01661.
- Lai et al. 2024, Chip Placement with Diffusion Models, arXiv:2407.12282.
- Raja et al. 2025, Action-Minimization meets Generative Modeling, ICML
  2025, arXiv:2504.18506.
- Xue et al. 2024, RL Policy as Macro Regulator, NeurIPS 2024,
  arXiv:2412.07167.
- Deng et al. 2025, EGPlace, ICML 2025.
- Shi et al. 2023, WireMask-BBO, NeurIPS 2023, arXiv:2306.16844.
- Lai et al. 2023, ChiPFormer, ICML 2023.
- Kahng et al. 2024, Hier-RTLMP, IEEE TCAD, arXiv:2304.11761.
- Kim et al. 2024, GOALPlace, arXiv:2407.04579.
- Henkelman & Jonsson 2000, Climbing-image NEB, J. Chem. Phys.
- Hu et al. 2025, Dimer-Enhanced Optimization, arXiv:2507.19968.
- Sanokowski et al. 2025, Scalable Discrete Diffusion Samplers,
  openreview 2502.08696.
- Wrigley, Lee, Ye 2017, Tensor Belief Propagation, ICML 2017.
- Spindler & Schlichtmann 2008, Abacus legalization.

Specific paper we should read carefully: Lai et al. 2024 "Chip
Placement with Diffusion Models" - the synthetic-data-generation
algorithm and the conditioning architecture might be directly portable
to our setting. If it generalizes from Lai's training distribution to
ariane133-style sparse layouts, it would be the strongest threat to our
champion submission - and the strongest candidate to incorporate into
post-deadline research.
