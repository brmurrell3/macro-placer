---
id: E102
name: GPU Lévy LBFGS
status: blocked
hypothesis: PATH B / cascade variant exploration
date_proposed: 2026-05-14
date_decided: 2026-05-16
---

# E102 — GPU Lévy + LBFGS saddle escape

## Hypothesis

Move the Lévy heavy-tail saddle escape (E97) to GPU and switch the
ε-step from a fixed-grid sweep to LBFGS line search. The Hessian
saddle-escape primitives are torch-autograd-compatible (E74's
`SmoothProxy.hvp`); LBFGS on the smooth proxy in the soft-mode
subspace should find sharper saddle exits than fixed-step ε
perturbations. CUDA-compiled DPO patch for the smooth-proxy autograd
graph at `code/dpo_cuda_patch.py`.

## Method

- `code/gpu_lbfgs_levy.py` — driver that wraps E97 Lévy saddle escape with
  a CUDA-backed `SmoothProxy` (via `dpo_cuda_patch.py`) and runs LBFGS
  in the soft-mode subspace instead of the fixed-grid ε sweep.
- Same trigger condition as E97 (cascade plateau).
- Same per-axis legality check (cascade saddle's `find_softest_eigenvectors`
  + ±ε projection).

## Kill gate

Either of:
- `--all` ≥ 1.07820 (no improvement over PATH A post-A1 floor).
- ariane133 ≥ 0.6641 (E74 baseline).

## Generalization check

NG45 ariane133 must not regress (>+1 % vs E74 0.6641 = falsified).

## Outcome

**Blocked 2026-05-16.** GPU box (AWS g5.xlarge) blocked on quota
approval; quota poll daemon ran overnight 2026-05-15/16, did not return
green by morning. Quota approval ETA unknown.

`dpo_cuda_patch.py` compiled cleanly against a local CUDA stub but was
never validated end-to-end against `~/DREAMPlace_cpu/install`'s torch
stack. `gpu_lbfgs_levy.py` runs on CPU but the LBFGS step takes ~10×
the time of the fixed-grid ε sweep (E97), making CPU-only LBFGS
strictly slower than E97 with no quality lift.

If GPU becomes available, the spike gate to re-run is:
ibm10 single-bench `--fast` must land within 5 % of E97's cached
0.96 in ≤ 30 min wall.

Not a champion candidate; submission variant scaffold at
`submissions/cd_lns_sa_cascade_xplace_levy/placer.py` (also blocked on
GPU).
