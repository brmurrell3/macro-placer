# DPO Next Steps

Last updated: 2026-04-23

DPO v3 achieves 1.4246 avg proxy (beats RePlAce by 2.3%). This document
outlines improvements to push further, organized by the two key insights:
(1) eliminate hardcoded constants, (2) exploit the available compute
budget more effectively.

---

## 1. The Compute Budget

### Two regimes

**IBM benchmarks (17 known designs):** These are developed and tested
on local hardware (M3 Max). Most competitive submissions run all 17
in under 5 minutes total. Our current DPO runs in ~288s total (~17s
avg per benchmark). There is headroom to spend more time per benchmark
without exceeding reasonable limits — 60s/benchmark (17 min total) is
our current ceiling, but 2-3 min/benchmark would still be practical.

**Hidden test case (1 unknown design):** The 1-hour budget on full
competition hardware (RTX 6000 Ada 48GB, 16-core EPYC 9655P, 100GB
RAM) is for this. The hidden design may be larger/harder than any IBM
benchmark. This is where GPU acceleration and heavy compute pay off —
we can afford 100K+ optimization steps, multi-restart, SA refinement,
and population search on a single design.

### Available compute on eval hardware

| Resource | Development (M3 Max) | Competition (eval) | Ratio |
|----------|---------------------|--------------------|-------|
| Time | ~17s avg/bench | 3600s (hidden) | **~200x** |
| GPU | None (CPU only) | RTX 6000 Ada 48GB | **~100x throughput** |
| CPU cores | 1 (sequential) | 16 (EPYC 9655P) | **16x** |
| RAM | 36GB | 100GB | 3x |

### Design principle

The architecture should scale gracefully: run fast on CPU for IBM
benchmarks (the ranking metric), and automatically exploit GPU + time
budget when available for the hidden test case. This means:
- GPU support via `device = torch.device("cuda" if available else "cpu")`
- Time-budgeted optimization (run until time limit, not fixed steps)
- Multi-restart when time permits
- Same code path, different resource allocation

### What extra compute enables (scaling from IBM → hidden test case)

**A. More optimization steps.** We currently run 200-900 steps of DPO.
On IBM benchmarks, doubling to 1000-2000 steps is feasible within a
2-3 min/benchmark budget. On the hidden test case with GPU, we could
run 50,000-100,000+ steps with much finer gamma annealing, gentler
overlap penalty continuation, and more time in each phase's sweet spot.

**B. Multiple independent restarts.** Currently we run DPO once from SDF
init. Running 3-5 seeds on IBM benchmarks adds ~1-3 min total. On the
hidden test case, we can run 10-50 restarts with different seeds,
perturbations, or starting points, trivially parallelizable across GPU
streams or CPU cores.

**C. Real proxy evaluation in the loop.** `compute_proxy_cost` takes
~50ms per call. Even on IBM benchmarks, we can afford 100-200 calls
per benchmark for calibration and SA refinement. On the hidden test
case: ~70K evaluations in the full hour, enabling:
- Periodic calibration of smooth proxy against ground truth
- SA refinement with exact signal after DPO (thousands of moves)
- Tournament selection between candidates using exact proxy

**D. Population-based training.** On the hidden test case: run N
concurrent DPO instances with different hyperparameters (gamma schedule,
lambda schedule, lr). Every K steps, evaluate all populations with real
proxy, kill the bottom half, clone+mutate the top half. This
automatically discovers the best hyperparameters without manual tuning.
Not practical for IBM benchmarks (too expensive for 17 runs), but ideal
for a single high-stakes design.

**E. GPU-accelerated DPO.** All our tensors fit easily in 48GB VRAM:
- Pin positions: [num_nets, max_pins, 2] ~ 10MB
- Grid overlap: [num_nets, R, C] ~ 100MB
- Total: < 1GB even for the largest benchmarks
PyTorch CUDA ops would give 10-100x speedup on the logsumexp, clamp,
topk, and matmul operations that dominate DPO step cost. Speeds up both
IBM benchmarks (more steps in same time) and the hidden test case
(massive step budgets).

**F. Ensemble of methods.** On the hidden test case: run DPO, SA, and
polyhedra navigation in parallel. Each gets ~20 minutes. Take the best.
Free +1-3% with no algorithmic innovation — just hedging across
approaches. For IBM benchmarks, a lighter version: run DPO + quick SA
polish (5-10s extra per benchmark).

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

### Phase A: Adaptive Constants (high impact, low effort)

These improve quality on IBM benchmarks without spending more compute.

1. **Position normalization** — normalize to [0,1]^2 for
   scale-invariant optimization. Eliminates lr/gamma scaling issues.
2. **Convergence-based phases** — replace fixed step counts with
   loss-stall detection. Small benchmarks finish faster, large ones
   get more time.
3. **Adaptive lambda** — gradient-ratio-based overlap penalty
   annealing.
4. **Gradient normalization** — balance WL/density/congestion
   gradient magnitudes so the optimizer doesn't ignore small
   components.

### Phase B: Compute Scaling (high impact, moderate effort)

These exploit available compute — modest gains on IBM benchmarks,
large gains on the hidden test case.

5. **Move tensors to CUDA** — add `.to(device)` throughout DPO.
   Detect GPU availability, fallback to CPU for development.
6. **Time-budgeted optimization** — run until a time limit rather
   than a fixed step count. Automatically scales to available
   compute.
7. **Add multi-restart** — run 3-5 seeds on IBM (adds ~1-2 min),
   10-50 seeds on hidden test case. Take the best.
8. **Add SA refinement** — after DPO + legalization, run SA with
   real proxy evaluation for remaining time budget (5-10s on IBM,
   minutes on hidden).

### Phase C: Congestion Model (medium impact, high effort)

9. **Calibration ratio** — periodic real-proxy evaluation to
   correct smooth congestion scale.
10. **Differentiable L-routing** — replace RUDY with L-shaped
    routing model.

### Expected Impact

| Change | IBM Benchmarks | Hidden Test Case | Effort |
|--------|---------------|------------------|--------|
| Adaptive constants | +0.5-1.5% | +0.5-1.5% | 1 day |
| GPU acceleration | +0.5-1% (more steps) | +2-4% (massive steps) | 1 day |
| Multi-restart (3-5 / 50 seeds) | +0.5-1% | +1-3% | 0.5 days |
| SA refinement | +0.2-0.5% | +0.5-1.5% | 1 day |
| Congestion calibration | +1-2% | +1-2% | 1 day |
| **Total (conservative)** | **+2-5%** | **+5-10%** | **~1 week** |

On IBM benchmarks: 1.4246 → ~1.35-1.39 (realistic with phases A+B).
On the hidden test case: the full compute budget could push
significantly further.

---

## 5. Quick Wins (Today)

Improvements that don't require major refactoring and stay within
reasonable IBM benchmark runtime (~5 min total for all 17):

1. **Multi-seed**: Run current DPO with 3 seeds, take best per
   benchmark. Cost: 3x runtime (~15 min total). Expected: +0.5-1%.

2. **More steps on fast benchmarks**: ibm01-ibm09 finish in <15s.
   Double step counts for benchmarks finishing under 20s.
   Expected: +0.5% on those benchmarks (ibm01 went 1.28→1.15 with
   more steps in earlier experiments).

3. **SA polish**: After DPO, try 100-500 random single-macro
   perturbations evaluated with `compute_proxy_cost`. Accept
   improvements. Cost: ~5-10s per benchmark. Expected: +0.2-0.5%.

---

## See Also

- [dpo.md](dpo.md) --- DPO theory and architecture
- [results.md](results.md) --- DPO v3 champion result (1.4246 avg)
- [roadmap.md](roadmap.md) --- project roadmap
