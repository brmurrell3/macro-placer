---
id: E86
name: gpu_hessian
status: falsified
parent: E74 (CPU Hessian phase)
created: 2026-05-10
decided: 2026-05-10
champion_at_time: 1.0666 (E74 CDLNSSAHessian, ADR-012)
outcome: **FALSIFIED 2026-05-10**. Smoke on ibm01: GPU (MPS) eigvals match CPU to 1.3e-7 (PASS on numerics) BUT wall is 3.8s on both — **0.98× speedup, no benefit**. Root cause: the Hessian phase wall budget is dominated by CD polish (CPU-only Python/numpy via IncrementalProxyEvaluator), not by Lanczos eigsh. Lanczos is already fast (~5s on CPU). GPU port speeds up only the negligible part. The wall problem is the polish step; that's where to look next for speedups (Numba/Cython for IncrementalProxyEvaluator, or a different polish algorithm). Code at `code/gpu_smooth_proxy.py` retained as a working device-aware port in case future GPU phases are added.
champion_delta: 0 (no wall saved; no proxy upside)
graduated_to: null
superseded_by: null
---

# E86: gpu_hessian — port smooth-proxy autograd to MPS/CUDA

## Hypothesis

E74's Hessian phase (Lanczos `eigsh` calling `torch.autograd.functional.hvp`
repeatedly on the smooth proxy) is CPU-bound. The smooth proxy is pure
PyTorch tensor ops — fully GPU-portable. Moving the inner loop to MPS
(M3 Max) or CUDA (partcl RTX 6000 Ada) should give 5–10× speedup of the
Hessian-vector products.

This is an **enabler**, not a direct proxy lift. The wall budget saved
goes back into E84 (more cascade iterations) or E85 (more inits per
bench) inside the same 1-hr-per-bench cap.

Currently the Hessian phase takes ~10–20 min/bench on M3 Max CPU. GPU
target: 2–3 min/bench. On the largest bench (ibm12, n_free=5272) the
HVP saves the most absolute time.

## Method

Modify `experiments/E74_hessian_saddle/code/hessian_saddle.py`:

1. Add `device` parameter to `SmoothProxy.__init__`; default
   `"mps" if torch.backends.mps.is_available() else "cuda" if
   torch.cuda.is_available() else "cpu"`.
2. Move all tensor state in `SmoothProxy` (`sizes`, `half_sizes`,
   `cell_x_min`, `cell_x_max`, etc.) to `device`.
3. In `cost()`, ensure input `positions` is moved to device:
   `positions = positions.to(device)`.
4. In `hessian_vector_product`, ensure `state` and `v` are on device;
   the autograd graph runs on device end-to-end.
5. In Lanczos `matvec`, the input is numpy → tensor on CPU → device →
   HVP → device → CPU numpy. Round-tripping is cheap (~MB/sec).

Validation:
- Smoke on ibm01: confirm same eigvals (within 1e-4 tolerance) as CPU.
- Wall: confirm 5–10× speedup of the Hessian phase.
- Numerical: smooth-proxy cost must match CPU to within float32 noise.

Risk: MPS has limited op support (some torch ops fall back to CPU
silently). If `_lse_hpwl` / `_grid_density` / `_rudy_congestion` fall
back, no speedup. Mitigation: explicitly catch fall-back via
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` profiling.

## Kill gate

- Smoke ibm01: GPU Hessian eigvals differ from CPU by > 1e-3 → numerical
  bug; kill until fixed.
- Wall speedup < 2× → not worth the install complexity for partcl; kill
  (CPU stays).
- Memory blowup → MPS has 16 GB unified; ibm12 (n=2636) at fp32 could
  hit limits with the autograd graph. Mitigate via fp32 throughout
  (already default).

## Generalization check

partcl RTX 6000 Ada has CUDA — different device path, but same code path
(PyTorch device-agnostic). Validate on CUDA target before submit.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/gpu_hessian_test.py` (numerical equivalence test),
  `code/gpu_smooth_proxy.py` (device-parametric SmoothProxy).
- Parent CPU baseline: `experiments/E74_hessian_saddle/code/hessian_saddle.py`.
