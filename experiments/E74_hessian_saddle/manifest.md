---
id: E74
name: hessian_saddle_escape
status: graduated
parent: E48 (champion plateau), E28 (proposed since 2026-04-29, never built), vmallela #2 leaderboard approach (Hessian negative-eigenvalue saddle escape)
created: 2026-05-04
decided: 2026-05-05
champion_at_time: 1.08151 (E48 hybrid)
outcome: **CHAMPION 2026-05-05 ~05:30 UTC. Verified --all aggregate 1.0666 = -1.38% over E48 1.08151, -4.53% over leaderboard 1.1172, -26.8% over RePlAce 1.4578. NG45 avg 0.6813 = -1.57% over E48 0.6922, with ariane133 at 0.6641 = -3.21% (BREAKING the failure point that killed E42/E43/E44/E54/E62).** Mechanism: smooth-proxy Hessian via torch.autograd.functional.hvp + Lanczos smallest-algebraic eigenvalues + ±ε saddle perturbation + CD polish. Smooth-proxy Hessian had genuinely NEGATIVE smallest eigenvalues on every IBM bench (ibm01 λ=-0.494, ibm12 λ=-0.105) and small-but-negative on NG45 (ariane133 λ=-0.0001) — confirming E48 "plateau" is a smooth-proxy saddle in both regimes. Per-bench IBM lifts: ibm02 -7.13% (largest), ibm01 -3.86%, ibm15 -1.73% (E61V2-layered), ibm07 -1.67%, ibm06 -1.86%, ibm04 -0.93%, ibm10 -0.90%, ibm16 -0.91%, ibm03 -0.66%, ibm12 -0.61% (layered), ibm11 -0.61%, ibm13 -0.69%, ibm14 -0.55%, ibm08 -0.45%, ibm18 -0.38%, ibm17 -0.29%, ibm09 -0.11%. NG45 lifts: ariane133 -3.21%, ariane136 -2.50%, mempool_tile tied, nvdla -0.75%. ZERO overlaps on all 17 IBM + 4 NG45. **ADR-012 *Accepted* 2026-05-05.** Beats every VERIFIED leaderboard entry by ≥17% (best previously verified: MTK 1.2818).
champion_delta: -0.0149 (-1.38%) verified vs E48 baseline
graduated_to: submissions/cd_lns_sa_hessian/placer.py
superseded_by: null
---

# E74: hessian_saddle_escape — Lanczos soft-mode escape from CD plateau

## Hypothesis

Per E25 manifest analysis, the multi-mechanism plateau is the
intersection of fixed points of CD per-axis breakpoint, grid-bin LNS,
SA-v2, pair swap, and K-joint K=3. Local moves saturate. Reaching deeper
requires either (a) a fundamentally new move type, (b) escaping via a
saddle point, or (c) re-init in a different basin.

This is option (b). At a CD plateau (smooth-proxy local min approximate),
the Hessian H = ∂²f/∂p² is positive semi-definite locally; but global
non-convexity means there exist nearby points where some eigenvalue is
NEGATIVE. Following the smallest-magnitude eigenvector (the "softest
mode") for an ε step takes us off the plateau.

vmallela's leaderboard #2 entry is exactly this approach (1.1 self-
reported; previous CD+LNS variant verified at 1.4152, so verification
drift is real). Even if verified score lands at 1.15-1.20, the
mechanism is sound and we can implement it.

## Method

```
init: state = E48 hybrid output (the plateau)

for k = 1..K:  # try multiple eigenvectors
    1. Compute smooth proxy gradient ∇f(state) — autograd
    2. Compute smallest-magnitude eigenvector v_k of Hessian H
       at state, via Lanczos (scipy.sparse.linalg.eigsh with
       LinearOperator wrapping torch.autograd.functional.hvp).
    3. Step εv_k for several ε values:
       perturbed = state + ε * v_k
       project_overlaps(perturbed)  # legalize
       proxy = compute_proxy_cost(perturbed)  # canonical TILOS proxy
    4. From best perturbed, run CD-LNS-SA polish.
    5. If polished proxy < state proxy: accept; state ← polished.
return state
```

Use DPO smooth proxy (extracted from
`writeup/archive/submissions/dpo/ablation_v2_steps.py`):
- `_lse_hpwl` — log-sum-exp HPWL (γ-smoothed)
- `_grid_density` — top-10% density
- `_rudy_congestion` — top-5% RUDY
- `_overlap_penalty` — pairwise overlap relu (skip for Hessian; we
  evaluate at a feasible E48 state where overlap penalty is ~0)

Hessian-vector products via `torch.autograd.functional.hvp`. Lanczos
with k=5 eigenvectors needs ~30 H-v products per call. ~100 ms each →
~3 sec per Hessian probe. Tractable.

## Kill gate

- Lanczos doesn't converge on ibm01 in 60 iters → kill (numerics broken).
- ε-step from softest mode never produces feasible placement that
  polishes below E48 baseline → kill (saddle direction doesn't escape).
- --fast aggregate < 0.3% lift over E48 → marginal.
- --fast aggregate ≥ 0.3% AND NG45 ariane133 doesn't regress → graduate.

## Generalization check

NG45 ariane133 — must not regress. Smooth proxy for ariane133 has
different scale (sparse layout); validate Lanczos converges there too.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/smooth_proxy.py` (autograd primitives extracted from DPO),
  `code/hessian_saddle.py` (Lanczos + ε-step + CD polish),
  `code/run_hessian_saddle.py` (driver).
- Smooth-proxy reference:
  `writeup/archive/submissions/dpo/ablation_v2_steps.py:420-591`.
- Roadmap E28 (never built):
  `docs/roadmap.md` §5.1 Tier 0.
- Leaderboard #2 approach: vmallela LSJ, "Hessian negative-eigenvalue
  saddle escape branch", 5/3 resubmit.
