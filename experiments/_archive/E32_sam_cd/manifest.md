---
id: E32
name: sam_cd
status: falsified
parent: E25
created: 2026-04-29
decided: 2026-04-29
champion_at_time: 1.0990 (E12; E25 candidate not promoted)
outcome: 0.98501 (--fast); +5.5 % vs E25 fast 0.9336; kill gate fired (gate = +5 %); zero per-bench wins. SAM-CD K=4 perturbation slowdown caused CD to hit 2400 s cap before plateau; ibm13 ended at +10.6 % vs E25.
champion_delta: +0.0514 (+5.5 %)
graduated_to: null
superseded_by: null
---

# E32: sam_cd

## Hypothesis
CD's fixed point may be a *sharp* local minimum — an artifact of the
proxy's exact form. SAM-style (Foret et al., ICLR 2021) worst-case
perturbation breakpoint scoring evaluates each candidate axis position
under the worst of K random nearby placements, rewarding *flat* basins
over sharp ones. Because flat minima are typically more robust to
proxy-form variations, this should improve OOD generalization to
NG45's slightly different proxy regime, even at potential modest cost
on the in-distribution IBM proxy.

## Method
Same pipeline as E25 (CD plateau → grid-bin LNS → SA-v2), but the CD
phase replaces `search_axis` with `sam_search_axis`. For each candidate
axis position v from `axis_breakpoints(...)`:

1. Compute base proxy at `(v, perp_val)` (move + revert).
2. Sample K=4 perturbations δ ~ uniform on disk of radius
   ρ = sam_rho_frac × canvas_diag (default sam_rho_frac = 0.01, i.e.
   1 % of canvas diagonal).
3. For each δ, compute proxy at `(v + δ_x, perp_val + δ_y)` after
   clamping to legal range. Move + revert.
4. Effective cost = max(base, perturbed_costs). Accept the v with the
   lowest max-cost (the *flattest* minimum).

LNS phase and SA-v2 phase are unchanged from E25. Hyperparams `sam_K`
and `sam_rho_frac` are constructor kwargs for sweepability. Otherwise
identical to E25 (CD ≤ 2400 s + LNS ≤ 600 s + SA ≤ 600 s).

Eval cost per CD probe is multiplied by ~(K+1) = 5×, so CD makes
fewer sweeps in the same budget. The hypothesis is that the flatter
basins this finds will generalize better, not that CD itself
converges faster.

## Kill gate
SAM-CD `--fast` proxy ≥ E25 `--fast` 0.9336 → kill. If SAM-CD
doesn't even match E25 on the in-distribution IBM proxy, the OOD
generalization claim is dead — there's no headroom to lose
in-distribution performance for OOD gain.

## Generalization check
If `--fast` passes (avg < 0.9336), run on NG45 ariane133. The SAM
hypothesis predicts BETTER OOD performance: SAM-CD's NG45 lift over
E12 must be ≥ its IBM lift. If NG45 lift < IBM lift, the OOD claim
fails — SAM is just an expensive variant with no transfer benefit.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_sam.py` (defines `CDLNSSASAMPlacer`).
- Reuses `run_lns_gridbin` and `run_sa_polish_v2` from E25 (inlined
  in `submissions/cd_lns_sa/placer.py`).
- Reuses `legal_axis_range`, `axis_breakpoints`, `_grid_lines`,
  `sdf_init`, `project_overlaps` from `macro_place.cd_core`.
- Custom: `sam_search_axis`, `_sweep_macros_sam`, `run_sam_cd_adaptive`
  (mirror of `run_cd_adaptive` with SAM-style scoring).
- Parents: E25 (CD + LNS + SA pipeline reference).
- Wall budget: ~3 hr build + ~2 hr `--fast` (K=4 perturbations
  multiplies CD eval cost by ~5×).
