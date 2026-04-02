# Macro Placement: Novel Solution Candidates

This document contains only approaches that are (a) not standard practice in the EDA community and (b) not disproven by published research. Everything that exists today or has been shown to fail is excluded.

For context on what exists today: the state of the art is electrostatic density spreading with Nesterov acceleration (ePlace/RePlAce/DREAMPlace), using log-sum-exp wirelength smoothing, FFT-based Poisson density, and Tetris-style legalization. Multi-scale coarsening, spectral initialization, force-directed models, and geometric annealing schedules are all standard. RL-based placement from scratch has been thoroughly debunked -- properly configured SA and RePlAce consistently outperform it on the ICCAD benchmarks used in this competition.

The baseline to beat: RePlAce at 1.4578 average proxy cost.

---

## Method 1: Optimal Transport Density Spreading

**Status:** Validated by one recent paper (RUPlace, DAC 2025). Not yet applied to this competition's benchmark suite.

**What changes vs. the baseline.** Replace the Poisson-equation-based density spreading in ePlace with Wasserstein-2 optimal transport. Instead of computing an electrostatic field to push overlapping macros apart, compute the minimum-cost transport plan from the current macro density to uniform target density using the Sinkhorn algorithm.

**Why this is different from what exists.** The electrostatic model produces a spreading field that has no mathematical relationship to minimum displacement. It just happens to spread things. OT computes the provably minimum-cost way to redistribute mass, meaning macros move less during spreading and more global placement structure is preserved. The gradient of the Wasserstein distance with respect to macro positions points in the direction of steepest descent in a true metric space, unlike the Poisson potential which is a heuristic proxy.

**Evidence it works.** RUPlace (Chen, Mai, Zhang, Lin -- DAC 2025) used Wasserstein distance within an ADMM-based placer and achieved $4.7\times$ congestion reduction and $7\%$ wirelength improvement vs. OpenROAD, with $3.67\times$ speedup.

**What could go wrong.** Sinkhorn is $O(M^2)$ naively for $M$ grid cells. Convolutional Sinkhorn or sliced Wasserstein approximations are needed. The regularization parameter $\epsilon$ is tricky to tune -- too large blurs out useful information, too small causes numerical instability. Unbalanced OT (total macro area $<$ canvas area) adds implementation complexity.

**Validation test.** On ibm01: replace density gradient with GeomLoss Sinkhorn divergence gradient in a minimal analytical placer. Run 1000 gradient steps. Compare proxy cost and density heatmap against the same placer using ePlace-style density. Pass criterion: proxy cost within 10% and density visually smoother.

**Implementation.** GeomLoss library (PyTorch, MIT license). Drop-in differentiable Sinkhorn divergence. Estimated 3-5 days to integrate.

---

## Method 2: SDF-Based Soft Rasterization for Density

**Status:** No prior work in EDA. Genuinely novel. Adapted from differentiable rendering (computer graphics).

**What changes vs. the baseline.** Replace the bell-shaped Gaussian smoothing used to project macro footprints onto the density grid with exact signed distance functions. For each grid point, compute the signed distance to each rectangle boundary, then pass through a sigmoid with temperature parameter $\tau$. This gives a geometrically exact soft occupancy that represents rectangles as rectangles, not blobs.

**Why this is different from what exists.** Current analytical placers approximate rectangles as Gaussian bells, losing geometric fidelity. An $8 \times 4$ rectangle and a $4 \times 8$ rectangle produce nearly identical density contributions in the Gaussian model. SDF preserves the actual shape. Temperature annealing ($\tau$ large to small) provides a natural continuation: at high $\tau$, gradients are global but smooth; at low $\tau$, the density map is sharp and geometrically exact. No FFT is needed -- the computation is embarrassingly GPU-parallel.

**Evidence it might work.** SDF-based differentiable rendering is mature in computer graphics (SoftRas, nvdiffrast, diffvg). The mathematical properties (smooth, differentiable, temperature-controllable, geometrically exact) are precisely what placement density computation needs. No evidence it has been tried in EDA, so no negative results exist.

**What could go wrong.** The $O(N \cdot M)$ complexity ($N$ macros times $M$ grid cells) may be slower than FFT-based Poisson for large grids, though for $N = 500$ and $M = 10{,}000$ this is likely fine. At large $\tau$, the sigmoid gradients may be too diffuse to provide useful directional information, making the global phase slow. May need to combine with OT for global spreading and use SDF only for refinement.

**Validation test.** Implement SDF density in PyTorch (estimated <50 lines). On ibm01, compute density grid and gradient. Verify gradient correctness with finite differences (relative error $< 10^{-4}$). Then run 1000 Adam steps on $WL + 0.5 \cdot D_{\text{sdf}}$. Pass criterion: macros spread from a clustered initial state, proxy cost decreases monotonically.

**Implementation.** Pure PyTorch. Broadcasting: macro centers as $[N,1,1,2]$ tensor, grid coords as $[1,H,W,2]$. Custom CUDA kernel optional for $5\times$ speedup. Estimated 2-3 days for basic implementation.

---

## Method 3: Diffusion Generative Model

**Status:** Validated by UC Berkeley (ICML 2025). Shows competitive results on IBM and ISPD benchmarks. Not yet applied to this specific competition.

**What changes vs. the baseline.** Instead of iterative optimization, train a neural network to generate complete placements in a single forward pass. The model learns to denoise random Gaussian noise into valid placements, conditioned on the netlist graph structure via a GNN encoder. Energy-guided sampling injects wirelength and density gradients during the denoising process.

**Why this is different from what exists.** All existing methods are iterative optimizers that search a solution space. Diffusion models learn a distribution over good placements and sample from it. This enables: generating diverse candidate solutions (sample multiple times), zero-shot transfer to new designs (no per-benchmark training), and naturally parallel evaluation of multiple candidates.

**Evidence it works.** The Berkeley paper trains on synthetic data and demonstrates zero-shot transfer to real IBM and ISPD benchmarks, outperforming existing methods on HPWL and congestion in the macro-only setting. DiffPlace (2025) extends this with energy-conditioned denoising and constrained manifold projection, achieving near-zero overlaps.

**What could go wrong.** Non-overlap enforcement remains imperfect -- all diffusion-based results require post-hoc legalization. Only 17 public benchmarks for training (though synthetic augmentation helps). The $33\times$ macro size variation in this competition's benchmarks may cause multimodal denoising distributions that are harder to learn. Training requires GPU time and hyperparameter search. May not beat well-tuned analytical methods on proxy cost alone, but could be strong for the Innovation Award or as an initialization method.

**Validation test (before training anything).** Implement energy-guided Langevin sampling without a learned model: start from random positions, iteratively update with gradient of $-(WL + 0.5 \cdot D)$ plus Gaussian noise with decaying temperature. This is the "non-learned diffusion" baseline. If this produces placements within 50% of SA baseline proxy cost, a learned version should do much better. If it doesn't, the energy landscape may be too rough for gradient-guided sampling to work.

**Implementation.** PyTorch + PyG (for GNN). Training: ~1 week. Inference: seconds per placement. Estimated total effort: 2-3 weeks including training infrastructure.

---

## Method 4: Variational / Least-Action Formulation

**Status:** No prior work in EDA or any placement context. Theoretically grounded but unimplemented. Highest novelty, highest risk.

**What changes vs. the baseline.** Reformulate the entire multi-stage pipeline (global placement, density spreading, legalization) as minimizing a single action functional over the trajectory from initial to final placement. The action combines displacement cost (Benamou-Brenier kinetic energy), wirelength, and density penalty, integrated over the optimization trajectory. The optimal schedule for the density penalty $\lambda(t)$ falls out of the Euler-Lagrange equations rather than being hand-tuned.

**Why this is different from what exists.** Current methods solve three separate problems in sequence and hand-tune the transition between them. The variational approach says these stages are artificial -- the optimal path from random to legal placement is a single curve whose properties are determined by the action functional. Key novel insight from our research: the entropy-regularized OT used in Sinkhorn and the log-sum-exp wirelength smoothing are the same mathematical object via Legendre-Fenchel duality. This structural identity appears to be genuinely new and connects two communities that don't cite each other.

**Evidence it might work.** The Bregman Lagrangian framework (Wibisono, Wilson, Jordan 2016 -- PNAS) proves that Nesterov's accelerated gradient descent (the very optimizer used in ePlace/DREAMPlace) arises from the Euler-Lagrange equation of a specific Lagrangian. The JKO scheme (Jordan, Kinderlehrer, Otto 1998) provides a practical discretization for Wasserstein gradient flows. The Benamou-Brenier formulation of $W_2$ distance is literally least-action for density transport. Wasserstein gradient flows are an active research area with recent GPU-friendly implementations.

**What could go wrong.** The JKO scheme requires an inner optimization (proximal step) at each outer step, potentially doubling computational cost. The action functional may have saddle points or degenerate Hessians that make variational optimization numerically unstable. The gap between theory and practical implementation may be too large for the 7-week competition timeline. The approach may produce theoretically elegant but practically inferior results compared to well-tuned heuristic schedules.

**Validation test.** On ibm01, simplified version: implement 50 JKO-like steps where each step minimizes $WL + \lambda \cdot D + \frac{1}{2\tau} \sum \text{squared\_displacements\_from\_previous\_step}$. Use $\tau = 0.1 \cdot \text{canvas\_width}^2$. Track the trajectory. Pass criterion: the trajectory produces a smooth, visually plausible evolution from random to organized placement, and final proxy cost is within 40% of SA. If the trajectory is chaotic or proxy cost is worse than random, the variational structure is not informative at this discretization level.

**Implementation.** PyTorch + GeomLoss for Wasserstein terms. Estimated 1-2 weeks for the simplified version. Full Neural ODE adjoint version: 3+ weeks.

---

## Method 5: Neural ODE Adjoint Schedule Optimization

**Status:** Precedent exists in EDA (Soda-PTA, ICCAD 2024, for SPICE simulation) but not for placement. Novel application.

**What changes vs. the baseline.** Treat the entire placement optimization as a continuous-time dynamical system. Parameterize the wirelength smoothing schedule $\gamma(t)$ and density penalty schedule $\lambda(t)$ as learnable functions (small neural network or polynomial). Use the adjoint method to compute $d(\text{final\_proxy\_cost}) / d(\text{schedule\_parameters})$ by integrating backward through the optimization trajectory. Optimize the schedule end-to-end for actual outcome.

**Why this is different from what exists.** Current schedules are hand-tuned or swept via grid search / Bayesian optimization (AutoDMP). The adjoint method computes exact gradients of the final result with respect to every schedule decision, at $O(1)$ memory cost. This is not searching a hyperparameter space -- it's computing the mathematically optimal schedule given the problem landscape.

**Evidence it might work.** The adjoint method is proven for ODEs (Chen et al. 2018, NeurIPS Best Paper). Soda-PTA (ICCAD 2024) successfully applied it to SPICE circuit simulation. DREAMPlace is already implemented in PyTorch, making the entire forward pass differentiable (minus legalization). The signal should exist: small schedule changes produce measurable proxy cost differences (AutoDMP showed this).

**What could go wrong.** The optimization trajectory may be chaotic (sensitive to initial conditions), making adjoint gradients noisy or uninformative. The legalization step breaks differentiability -- a differentiable surrogate is needed. Schedules optimized on the 17 IBM benchmarks may not generalize to hidden test cases. The backward integration through 1000+ placement iterations may be numerically unstable.

**Validation test.** On ibm01 only: implement 200-step placement with fixed schedules. Wrap in torchdiffeq as a Neural ODE. Compute adjoint gradient of final HPWL with respect to 5 schedule parameters. Pass criterion: gradients are non-zero, have consistent sign across 3 random seeds, and following the gradient for 10 steps reduces final HPWL by $> 1\%$. If gradients are zero or inconsistent, the signal doesn't survive the trajectory length.

**Implementation.** torchdiffeq library. Requires DREAMPlace-compatible differentiable pipeline. Estimated 1-2 weeks.

---

## Method 6: Consensus-Based Optimization (CBO)

**Status:** No prior work in placement or any spatial packing problem. Novel application of a method from applied mathematics.

**What changes vs. the baseline.** Replace gradient-based optimization with a swarm method that has unique theoretical properties. $N$ particles (candidate placements) evolve according to CBO dynamics: each particle drifts toward a consensus point (weighted average of all particles, with weights exponentially favoring lower-cost particles). Noise enables exploration. The key property: convergence is provably dimension-independent, unlike PSO or CMA-ES which degrade in high dimensions.

**Why this is different from what exists.** PSO and CMA-ES have been tried for placement and don't scale past ${\sim}100$ dimensions. CBO's dimension-independent convergence (Carrillo, Jin, Li, Zhu 2021) breaks this barrier. For 1000D placement ($500$ macros $\times$ 2 coordinates), this is a theoretically significant advantage. Polarized CBO (Bungert et al. 2024) handles multiple global minimizers, relevant when placement has several near-optimal configurations.

**Evidence it might work.** Theory is strong: SIAM Journal on Optimization (2024) proves CBO convexifies non-convex problems in the infinite-particle limit. No experimental evidence in placement specifically. The theory assumes unconstrained optimization, and non-overlap constraints are non-convex and disconnected -- this is the main gap.

**What could go wrong.** The non-overlap constraint has no natural CBO encoding. Penalty methods may not converge. The infinite-particle limit may require impractically many particles for 1000D. CBO convergence may be slower than gradient-based methods when gradients are available (which they are for the relaxed placement problem). May be a curiosity rather than a practical method.

**Validation test.** On ibm01 (reduced to hard macros only, ~246 variables in 492 dimensions): run CBO with 100 particles, 5000 iterations, consensus weight $\alpha = 100$, noise $\sigma = 0.5$ decaying to $0.01$. Use proxy cost with soft overlap penalty as objective. Pass criterion: best particle achieves proxy cost within $2\times$ of SA baseline. If best particle is worse than random after 5000 iterations, the method doesn't work for this problem structure.

**Implementation.** Pure PyTorch (CBO is simple to implement: weighted average + noise). Estimated 2-3 days.

