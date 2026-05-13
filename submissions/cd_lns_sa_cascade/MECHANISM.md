# Algorithm description — for submission form

**Placer:** `CDLNSSACascadeAdaptivePlacer` in `placer_adaptive.py`
**Entry file:** `submissions/cd_lns_sa_cascade/placer_adaptive.py`
**Single global algorithm; no per-benchmark hyperparameters.**

## One-paragraph summary

A best-of cascade pipeline that combines two distinct initialization
basins (SDF + DPO) with a cascading transition-state saddle escape on
the smooth proxy. Each phase is wall-budgeted under the 60-min cap;
the cascade saddle iterates Lanczos-eigenvector perturbation + CD
polish until the smooth proxy reaches a true local minimum (smallest
eigenvalue >= 0) or budget runs out. Bench-property-tuned CD
plateau-detection (canvas area > 100k um^2 -> longer min_time + tighter
threshold) accommodates NG45 commercial designs vs IBM ICCAD04.

## Pipeline (per benchmark)

1. **Phase 1 — E25 (SDF basin polish).** SDF init (signed distance
   field — Kahng 2022) -> project_overlaps -> CD adaptive (axis-by-axis
   coordinate descent on breakpoints) -> grid-bin LNS -> SA-v2 with
   best-so-far tracking.
2. **Phase 2 — E41 (DPO basin polish, skipped if <300 s budget left).**
   DPO best-of-v2 init -> CD adaptive -> grid-bin LNS -> SA-v2 ->
   K-macro joint LNS (K=3, top-N=5 candidate enumeration).
3. **Phase 3 — Cascading Hessian saddle escape on min(E25, E41) plateau.**
   For each iteration: compute the smooth-proxy Hessian via
   `torch.autograd.functional.hvp`, find the smallest-algebraic
   eigenvector via Lanczos (scipy `eigsh` with `LinearOperator`),
   perturb +-eps along the soft mode for `eps in {0.3, 1.0, 3.0}`,
   project to feasible, CD-polish each, keep best. Iterate until
   smallest eigenvalue >= 0, or no improvement, or budget runs out.
4. **Final pick:** best-of {E25, E41, cascade} with defensive
   overlap-validated fallback chain (if cascade fails the validation,
   fall through to E41 then E25; E25 is always overlap-free by
   construction).

## Why this works

- **Two independent basins.** SDF and DPO land in structurally distinct
  basins (see E25/E41 manifests; E65 NEB cross-section showed they're
  separated by infeasibility walls, not proxy barriers — so we can't
  bridge them, but we can polish each and pick the better one).
- **Hessian saddle escape (Henkelman & Jonsson 2000 climbing-image
  NEB, dimer / gentlest-ascent).** At the polished plateau, the smooth
  proxy has multiple negative-curvature directions; perturbing along
  the smallest eigenvector and re-polishing reaches deeper local
  minima the local-move CD cannot. Cascading iterates this to depletion.
- **Wall budget enforcement.** Hard `deadline = start + 3000s` enforced
  at phase boundaries; cascade stops cleanly when remaining time drops
  below a safety threshold, returning the best legal state seen so far.

## Hardware

Validated on AMD EPYC 9655P (partcl-equivalent) + 30 cores, 216 GB,
under `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.
Each benchmark runs in ~50 min, max wall 57 min, safe under the
60-min cap. Implementation is pure Python + PyTorch + scipy; no GPU
required (PyTorch only for autograd + Lanczos HVP, all on CPU).

The internal `IncrementalProxyEvaluator` was hot-path-optimized
(see `macro_place/incremental_evaluator.py`):
- `delta_cost(macro_idx, new_xy)` — no-mutation cost peek, replaces
  the (move, current_cost, revert) probe triple in CD's `search_axis`
  and LNS destroy/reinsert probes; ~2x per probe.
- `delta_cost_axis_batch(macro_idx, axis, cur_xy, candidates)` —
  amortizes the "subtract old contributions" precompute once across K
  candidates and runs density + congestion top-K as batched torch ops
  on `[K, num_cells]` tensors; further ~2.3x on top of single-probe.
- `_net_cong_contrib_flat(net_idx)` — vectorized pin->gcell, inlined
  2-pin / multi-pin routing, no dict allocation; further ~1.2x.
- `commit(macro_idx, new_xy)` — apply move without snapshot or cost
  recompute; reserved for accept paths.

Combined: **5.36x speedup on the CD inner loop** vs the original
`move + current_cost + revert` pattern, enabling cascade to converge
multiple iterations within the 60-min cap on EPYC.

## Reproducibility

```bash
git clone <repo>
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement
uv sync
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --all --json
```

No external solvers, no GPU dependencies, no per-benchmark tuning,
no benchmark-identity dispatch.

## Hard constraints

- Zero hard-macro overlaps on every benchmark — enforced internally
  with project_overlaps + per-phase overlap-metric checks; final
  output revalidated with a defensive overlap-free fallback chain.
- Fixed macros are never moved.
- All macros stay fully within the canvas.

## Reference for the curious

- E12 -> E48 -> E74 -> E84 cascade lineage in `docs/decisions/`.
- E84 cascading saddle ADR + verification at uncapped 1.0612 IBM.
- PATH A speedup wave landed 2026-05-12 in `TODO.md` history.
