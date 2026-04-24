# DPO Next Steps

Last updated: 2026-04-23

DPO v3 achieves 1.4246 avg proxy (beats RePlAce by 2.3%). This document
outlines improvements to push further, organized by the two key insights:
(1) eliminate hardcoded constants, (2) exploit the full competition compute
budget (1 hour + RTX 6000 Ada 48GB + 16-core EPYC + 100GB RAM).

---

## 1. The Compute Budget We're Ignoring

We currently run in 8-60s per benchmark on an M3 Max CPU. The competition
gives us:

| Resource | Development (now) | Competition (eval) | Ratio |
|----------|------------------|--------------------|-------|
| Time | ~60s | 3600s | **60x** |
| GPU | None (CPU only) | RTX 6000 Ada 48GB | **~100x throughput** |
| CPU cores | 1 (sequential) | 16 (EPYC 9655P) | **16x** |
| RAM | 36GB | 100GB | 3x |

We are using ~1% of the available compute. The 1-hour budget exists
because the organizers expect participants to USE it. The top submissions
will absolutely saturate this budget.

### What 3600s + GPU enables

**A. Massively more optimization steps.** We currently run 200-900 steps
of DPO. With GPU acceleration (10-50x speedup on the vectorized ops) and
60x more time, we could run 100,000+ steps. This means:
- Much finer gamma annealing (smoother convergence to true HPWL)
- Deeper overlap penalty continuation (gentler schedule, fewer residual overlaps)
- More time in the "sweet spot" of each phase

**B. Multiple independent restarts.** Currently we run DPO once from SDF
init. With the compute budget, we can run 10-50 independent DPO runs with
different seeds, perturbations, or starting points, and take the best.
This is trivially parallelizable across GPU streams or CPU cores.

**C. Real proxy evaluation in the loop.** `compute_proxy_cost` takes ~50ms
per call. In 3600s we can afford ~70K evaluations. This enables:
- Periodic calibration of the smooth proxy against ground truth
- SA refinement with exact signal after DPO (thousands of moves)
- Tournament selection between candidates using exact proxy

**D. Population-based training.** Run N concurrent DPO instances with
different hyperparameters (gamma schedule, lambda schedule, lr). Every K
steps, evaluate all populations with real proxy, kill the bottom half,
and clone+mutate the top half. This automatically discovers the best
hyperparameters for each benchmark without manual tuning.

**E. GPU-accelerated DPO.** All our tensors fit easily in 48GB VRAM:
- Pin positions: [num_nets, max_pins, 2] ~ 10MB
- Grid overlap: [num_nets, R, C] ~ 100MB
- Macro positions: trivial
- Total: < 1GB even for the largest benchmarks
PyTorch CUDA ops would give 10-100x speedup on the logsumexp, clamp,
topk, and matmul operations that dominate DPO step cost.

**F. Ensemble of methods.** Run DPO, SA, and polyhedra navigation in
parallel. Each gets ~20 minutes. Take the best per benchmark. This is
a free +1-3% improvement with no algorithmic innovation — just spending
compute to hedge across approaches.

---

## 2. Eliminating Hardcoded Constants

The current DPO has these hardcoded values that should be adaptive:

### 2a. Learning rate (currently 0.5 / 0.2 / 0.1)

**Problem:** lr=0.5 means a macro moves ~0.5um per step (before Adam
adaptation). On ibm01 (canvas ~5000um), that's 0.01% of the canvas.
On ibm18 (canvas ~50000um), it's 0.001%. The optimizer's effective
step size varies 10x across benchmarks.

**Fix:** Scale lr proportional to canvas diagonal:
```python
lr_base = 0.001 * sqrt(cw**2 + ch**2)  # ~1% of canvas per step
```
Or better: normalize positions to [0,1]^2 before optimization, so lr
is scale-invariant. Then denormalize at the end.

### 2b. Gamma schedule (currently 0.01W / 0.002W / 0.0005W)

**Problem:** Gamma controls the LSE smoothing. Too large = gradients
are diffuse and inaccurate. Too small = loss landscape is spiky and
optimization gets stuck. The right gamma depends on the actual pin
position distribution, not just canvas width.

**Fix:** Set gamma relative to the average inter-pin distance within
nets:
```python
avg_span = mean([bbox_span(net) for net in nets])
gamma_start = 0.5 * avg_span  # smooth over typical net spans
gamma_end = 0.01 * avg_span   # sharp enough for accuracy
```
Anneal exponentially from gamma_start to gamma_end.

### 2c. Overlap lambda schedule (currently 1 / 50 / 500)

**Problem:** The right lambda depends on the scale of proxy cost vs
overlap area. On benchmarks with high congestion (proxy~1.7), lambda=1
is negligible. On benchmarks with low congestion (proxy~1.0), lambda=1
is significant.

**Fix:** Adaptive lambda based on loss component ratio:
```python
# After each phase, compute ratio
proxy_grad_norm = grad_norm_of(proxy)
overlap_grad_norm = grad_norm_of(overlap)
lambda_next = lambda_cur * (proxy_grad_norm / overlap_grad_norm) * target_ratio
```
Where `target_ratio` ramps from 0.1 (phase 1) to 10.0 (phase 3).
This ensures the overlap gradient is always a controlled fraction of
the total gradient.

### 2d. Phase transitions (currently fixed step counts)

**Problem:** Small benchmarks converge in 50 steps but we run 300.
Large benchmarks might need 500 but we cut them to 100.

**Fix:** Convergence-based transitions:
```python
if loss_improvement_over_last_30_steps < 0.001 * current_loss:
    transition_to_next_phase()
```
With a minimum of 50 steps per phase and a hard time budget.

### 2e. Gradient normalization across components

**Problem:** The proxy cost is WL + 0.5*D + 0.5*C, but the gradient
magnitudes of WL, D, and C can differ by 100x. The optimizer sees
mostly the congestion gradient (largest) and ignores WL (smallest).

**Fix:** Normalize each component's gradient to unit norm before
combining:
```python
wl_grad = autograd.grad(wl, pos, retain_graph=True)[0]
den_grad = autograd.grad(density, pos, retain_graph=True)[0]
cong_grad = autograd.grad(congestion, pos, retain_graph=True)[0]

# Normalize to same magnitude, then weight
total_grad = (wl_grad / wl_grad.norm() * w_wl
            + den_grad / den_grad.norm() * w_den
            + cong_grad / cong_grad.norm() * w_cong)
pos.grad = total_grad + overlap_grad
```
This is what DREAMPlace does with "adaptive net weighting" — it
ensures balanced optimization across objectives regardless of scale.

---

## 3. Congestion Model Improvement

The RUDY congestion underestimates real L-routing congestion by ~2x.
This is our biggest model mismatch and the main reason ibm01/ibm06 lag.

### 3a. Differentiable L-routing

Replace uniform RUDY distribution with L-shaped routing:
- For each 2-pin subnet: horizontal demand along source row,
  vertical demand along sink column
- Use soft cell assignment (sigmoid or softmax over grid rows/cols)
  to make the "row containing pin y" differentiable

This is more complex but matches PlacementCost's routing model more
closely. The gradient would tell macros not just "shrink the bbox"
but "move to avoid creating L-paths through congested cells."

### 3b. Hybrid: smooth RUDY + periodic real calibration

Every K steps, evaluate with `compute_proxy_cost` and compute the
ratio `real_congestion / smooth_congestion`. Use this ratio as a
multiplicative correction on the smooth congestion:
```python
congestion_corrected = congestion_smooth * calibration_ratio
```
Recompute the ratio every 50-100 steps. This keeps the gradient
direction from RUDY but scales the magnitude to match reality.

### 3c. Learned congestion surrogate

Train a small neural network (3-layer MLP or 1D conv) to predict
`real_congestion(placement)` from features of the placement
(macro positions, net bounding boxes, grid density). Train on
50-100 evaluations at the start of optimization. The network is
differentiable by construction.

This is feasible with the 1-hour budget: 100 evaluations take ~5s,
training takes ~1s, and the network can be used for the remaining
optimization.

---

## 4. Recommended Implementation Plan

### Phase A: GPU + Compute Budget (high impact, moderate effort)

1. **Move tensors to CUDA** — add `.to(device)` throughout DPO.
   Detect GPU availability, fallback to CPU for development.
2. **Increase step count 10x** when on GPU with time budget.
3. **Add multi-restart** — run N independent DPO instances (different
   seeds), take the best. Parallelize across GPU streams.
4. **Add SA refinement** — after DPO + legalization, run SA with
   real proxy evaluation for the remaining time budget.

### Phase B: Adaptive Constants (high impact, low effort)

5. **Position normalization** — normalize to [0,1]^2 for
   scale-invariant optimization. Eliminates lr/gamma scaling issues.
6. **Convergence-based phases** — replace fixed step counts with
   loss-stall detection.
7. **Adaptive lambda** — gradient-ratio-based overlap penalty
   annealing.

### Phase C: Congestion Model (medium impact, high effort)

8. **Calibration ratio** — periodic real-proxy evaluation to
   correct smooth congestion scale.
9. **Differentiable L-routing** — replace RUDY with L-shaped
   routing model.

### Expected Impact

| Change | Estimated Improvement | Effort |
|--------|----------------------|--------|
| GPU + 10x steps | +1-3% avg proxy | 1-2 days |
| Multi-restart (10 seeds) | +1-2% avg proxy | 0.5 days |
| SA refinement | +0.5-1.5% avg proxy | 1 day |
| Adaptive constants | +0.5-1% avg proxy | 1 day |
| Congestion calibration | +1-2% avg proxy | 1 day |
| Ensemble (DPO+SA+poly) | +1-2% avg proxy | 0.5 days |
| **Total (conservative)** | **+3-7% avg proxy** | **~1 week** |

A 5% improvement from 1.4246 would give ~1.35 avg proxy. Combined
with the 1-hour compute budget, reaching 1.30 or below is plausible.

---

## 5. Quick Wins (Today)

If we want immediate improvement without major refactoring:

1. **Multi-seed averaging**: Run current DPO with 3-5 seeds, take
   best per benchmark. Cost: 3-5x runtime (still under 5 min total).
   Expected: +0.5-1%.

2. **Increase steps for small benchmarks**: ibm01-ibm09 finish in
   <15s. Double their step counts. Expected: +0.5% on those benchmarks
   (we saw ibm01 go from 1.28 to 1.15 with more steps).

3. **SA polish**: After DPO, try 100-500 random single-macro
   perturbations evaluated with real proxy. Accept improvements.
   Cost: ~5-10s. Expected: +0.2-0.5%.

---

## See Also

- [dpo.md](dpo.md) --- DPO theory and architecture
- [results.md](results.md) --- DPO v3 champion result (1.4246 avg)
- [roadmap.md](roadmap.md) --- project roadmap
