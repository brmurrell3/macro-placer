# Submissions

## Tier-1 entry candidates (verified, submission-day pick open as of 2026-05-16)

Two placers ship cleanly under the 60-min/bench cap and both have been
verified on the 17-IBM `--all` gate + 4-NG45 `--ng45` gate with zero
overlaps. Final choice is a submission-day decision; see
[`../docs/handoffs/2026-05-14_champion_found_dp_lane.md`](../docs/handoffs/2026-05-14_champion_found_dp_lane.md)
for the comparison data and the no-swap context.

| Option | Placer | IBM `--all` | NG45 `--ng45` | External deps | Notes |
|---|---|---:|---:|---|---|
| **A** | [`cd_lns_sa_cascade/placer_adaptive.py`](cd_lns_sa_cascade/placer_adaptive.py) (`CDLNSSACascadeAdaptivePlacer`) | **1.07820** | **0.68102** | none | PATH A post-A1 wall-safe cascade. Simpler; safer fallback. |
| **B** | [`cd_lns_sa_cascade_dp_lane/placer.py`](cd_lns_sa_cascade_dp_lane/placer.py) (`CDLNSSACascadeDPLanePlacer`) | **1.06650** | **0.68086** | DREAMPlace (optional; falls back to A if `DREAMPLACE_ROOT` unset) | Adds DREAMPlace as a 3rd init lane; plateau picks best of {E25, E41, DP-polished}. |

Composite avg across 21 benches: Option B **0.993** vs Option A 0.998
(−0.5 % composite). Both beat the leaderboard reference 1.1172 by
≥3.5 %; both beat RePlAce 1.4578 by ≥26 %.

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

## Recent probes (active sweep variants, post-E84 wave)

Variants generated during E91-E107 exploration. None promoted; some
still have external code references from `experiments/` and were
retained pending the writeup. To re-activate any of these for the
submission, re-run `--all` for the verified score and compare to
Options A/B above.

| Variant | What it is | Status |
|---|---|---|
| `cd_lns_sa_cascade_levy/` | Lévy heavy-tail ε saddle escape (E97) | Shipped mechanism — incorporated into `cd_lns_sa_cascade_dual_levy/` |
| `cd_lns_sa_cascade_dual_levy/` | Dual Lévy mechanism (E97 + E100 portfolio) | IBM 1.07305 / NG45 0.67939 — NG45 best (May 14); not single-bench-best overall |
| `cd_lns_sa_cascade_dual_levy_k5/` | Dual Lévy with k=5 eigvecs (E100 variant) | Probe |
| `cd_lns_sa_cascade_portfolio_levy/` | Weight-portfolio saddle (E100) | Probe |
| `cd_lns_sa_cascade_portfolio_levy_k3/` | Portfolio variant k=3 | Probe |
| `cd_lns_sa_cascade_levy_curvadapt/` | Lévy curvature-adaptive (E101) | Probe |
| `cd_lns_sa_cascade_levy_periphery/` | Periphery bias init (E107) | NEW BEST ariane133 0.65212 (−1.8 % vs E74) — partial validation |
| `cd_lns_sa_cascade_levy_canvasnorm/` | Canvas-normalized Lévy | Probe |
| `cd_lns_sa_cascade_levy_k5/` | Lévy k=5 eigvecs | Probe |
| `cd_lns_sa_cascade_levy_multiiter/` | Multi-iteration Lévy | Probe |
| `cd_lns_sa_cascade_levy_wide/` | Wide Lévy step | Probe |
| `cd_lns_sa_cascade_multidir/` | PATH C C3 multi-directional saddle | Validated locally (ibm01 −0.614 % lift); cloud aggregate ~0.05-0.15 % marginal |
| `cd_lns_sa_cascade_dp_levy/` | DP + Lévy combined | Probe |
| `cd_lns_sa_cascade_dp_multi_lane/` | Multi-DP K=4 hybrid (E96) | Validated WIN on ibm10 (−2.0 %) and ibm14 (−7.7 %) but not promoted as Tier-1 |
| `cd_lns_sa_cascade_xplace_levy/` | GPU Lévy LBFGS via Xplace (E102) | Blocked on GPU quota; scaffolded |

## Archived

- [`_archive/`](_archive/README.md) — falsified or superseded placers, retained for the writeup:
  - Prior champions: `cd_lns_gridbin/` (E12, ADR-007), `cd_adaptive/` (E9), `cd_only/`, `cd_lns_sa_hybrid/` (E48, ADR-011), `cd_lns_sa_hessian/` (E74, ADR-012).
  - Falsified mechanisms: contents of `_archive/falsified/`.
  - Killed sweep variants: `cd_lns_sa_cascade_tournament/`, `cd_lns_sa_cascade_tournament_v2/` (CHAMPION_FOUND 2026-05-14 verdict: v1 worse than DP-lane; v2 killed when user pivoted to Tier-2), `cd_lns_sa_cascade_levy_s100/s200/s300/` (seed-sweep validation probes, 0 external refs).
  - `will_seed/`: Will's pre-fork submission, kept for baseline comparison (1.5338).

## Not part of the submission

- [`examples/`](examples/) — reference placers (greedy, random) shipped with the harness.
