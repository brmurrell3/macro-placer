# Submissions

## Tier-1 entry candidates (verified 2026-05-17)

Three placers verified on the 17-IBM `--all` gate + 4-NG45 `--ng45`
gate with zero overlaps. Final pick is a submission-day decision.

| Option | Placer | IBM `--all` | NG45 `--ng45` | Combined | External deps | Notes |
|---|---|---:|---:|---:|---|---|
| **C (NEW CHAMPION)** | [`cd_lns_sa_cascade_stacked_periphery/placer.py`](cd_lns_sa_cascade_stacked_periphery/placer.py) (`CDLNSSACascadeStackedPeripheryPlacer`) | **1.05750** | **0.68930** | **0.987** | none | Cascade (canonical) → portfolio_saddle (3 non-canonical Hessian weights) → periphery safety wrapper. E25 try/except for NG45 nvdla. |
| **B** | [`cd_lns_sa_cascade_dp_lane/placer.py`](cd_lns_sa_cascade_dp_lane/placer.py) (`CDLNSSACascadeDPLanePlacer`) | 1.06650 | **0.68086** | 0.993 | DREAMPlace (optional; falls back to A if `DREAMPLACE_ROOT` unset) | Adds DREAMPlace as a 3rd init lane; plateau picks best of {E25, E41, DP-polished}. |
| **A** | [`cd_lns_sa_cascade/placer_adaptive.py`](cd_lns_sa_cascade/placer_adaptive.py) (`CDLNSSACascadeAdaptivePlacer`) | 1.07820 | 0.68102 | 1.003 | none | PATH A post-A1 wall-safe cascade. Simpler; safest fallback. |

Option C beats Option B by **−0.85 % IBM** and **−0.6 % combined**, with
no external dependencies (no DREAMPlace required). 13/17 IBM benches
beat or tie cached cascade. NG45 nvdla used E41-only lane (E25 had 2
residual overlaps that project_overlaps couldn't clean — try/except
fallback engaged).

Both Option C and Option B beat the leaderboard reference 1.1172 by
≥4.5 % IBM; all three beat RePlAce 1.4578 by ≥26 %.

See each placer's `README.md` for the run command, algorithm
description, and per-bench verified results.

## Tier-2 ORFS (per-design strategy)

Tier-2 uses ORFS routed metrics (WNS/TNS/Area), a different objective
than Tier-1's proxy. Per-design strategy verified 2026-05-15/16
([`../docs/handoffs/2026-05-16_tier2_orfs_findings.md`](../docs/handoffs/2026-05-16_tier2_orfs_findings.md)):

| Design | Recommended | Why |
|---|---|---|
| ariane133 | Ship **without** `MACRO_PLACEMENT_TCL` | ORFS auto-place beats every cascade variant by 1.2 ns of slack. |
| ariane136 | Ship **with** cascade `MACRO_PLACEMENT_TCL` | Cascade +0.4935 ns vs auto-place +0.0457 ns. |
| mempool_tile | Untested with macro-tcl fix | Default: ship cascade pending re-test. |
| nvdla | Auto-place fallback | Cascade triggered PDN failure (E74 placement pathology). |

## Component placers (called by Options A/B; not standalone submissions)

- [`cd_lns_sa_cascade/placer.py`](cd_lns_sa_cascade/placer.py) — wall-safe E84 cascade (E25 → E41 → cascading saddle, deadline-bound). Component of both Options A and B.
- [`cd_lns_sa/placer.py`](cd_lns_sa/placer.py) — E25 SDF basin lane (CD + LNS + SA-v2 polish on per-axis breakpoints). Component of cascade.

## Archived

[`_archive/`](_archive/) holds falsified, superseded, or below-floor
placers, retained for the writeup. **Per-bench wins are documented in
the relevant experiment manifests under `experiments/`** (E91, E96, E97,
E100, E107); only the two Options above ship.

- **Prior champions (superseded by Option A/B):** `cd_lns_gridbin/`
  (E12, ADR-007), `cd_adaptive/` (E9), `cd_only/`, `cd_lns_sa_hybrid/`
  (E48, ADR-011), `cd_lns_sa_hessian/` (E74, ADR-012).
- **Sub-floor sweep variants from the E91-E107 wave** (none promoted as
  Tier-1; all below verified floors). Notable per-bench results
  retained in archive:
  - `cd_lns_sa_cascade_dual_levy/` — NG45 0.67939 (best NG45 aggregate
    we've seen, May 14); IBM 1.07305 (below Option A's 1.07820 IBM).
  - `cd_lns_sa_cascade_levy_periphery/` — NEW BEST single-bench
    ariane133 0.65212 (−1.8 % vs E74 0.6641); per-bench, not aggregate.
  - `cd_lns_sa_cascade_dp_multi_lane/` — E96 multi-config DP K=4;
    per-bench wins on ibm10 (−2.0 %) and ibm14 (−7.7 %) but aggregate
    below floor.
  - `cd_lns_sa_cascade_multidir/` — PATH C C3 multi-directional saddle;
    local ibm01 −0.614 % lift, cloud aggregate marginal.
  - `cd_lns_sa_cascade_xplace_levy/` — E102 GPU Lévy LBFGS scaffold,
    blocked on GPU quota; not measured end-to-end.
  - Other probes: `cd_lns_sa_cascade_levy/`, `levy_canvasnorm/`,
    `levy_curvadapt/`, `levy_k5/`, `levy_multiiter/`, `levy_wide/`,
    `dual_levy_k5/`, `portfolio_levy/`, `portfolio_levy_k3/`,
    `dp_levy/` — all probes below floor.
- **Killed sweep variants:** `cd_lns_sa_cascade_tournament/`,
  `cd_lns_sa_cascade_tournament_v2/` (CHAMPION_FOUND 2026-05-14:
  tournament v1 lost to DP-lane on hard benches; v2 killed when user
  pivoted to Tier-2). `cd_lns_sa_cascade_levy_s100/s200/s300/`
  (seed-sweep validation probes, 0 external refs).
- **Falsified mechanisms** (kept as signposts): `_archive/falsified/`
  including the DP-lane autopsy that was later overturned by E91.
- **Baselines:** `will_seed/` (Will's pre-fork submission, 1.5338).

## Not part of the submission

- [`examples/`](examples/) — reference placers (greedy, random) shipped with the harness.
