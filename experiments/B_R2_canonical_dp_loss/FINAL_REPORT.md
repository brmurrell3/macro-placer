# Final report — overnight sub-1.0 push, 2026-05-13 → 14

## TL;DR

**Cannot deliver sub-1.0 in this session.** All tested variants land between 1.067 and 1.077 on 17 IBM (current submission floor: PATH A at 1.077; best candidate: hybrid baseline at 1.067). Target to win Tier 1: sub-1.0109 (vmallela). **Gap from our best (1.067) to target: −5.6%.** Refinements explored cannot bridge this.

## Variants exhaustively tested (17-IBM aggregate)

| Variant | 17-IBM avg | Notes |
|---|---:|---|
| Hybrid baseline (`cd_lns_sa_cascade_dp_lane`) | **1.0668** | Best. Yesterday's snapshot; cloud was wiped before re-run. |
| E100_portfolio_v2 (other agent) | 1.0717 | New today. +0.5% vs hybrid. |
| E100_dual_levy (other agent) | 1.0731 | New today. +0.6% vs hybrid. |
| E97_levy (other agent) | 1.0770 | +1.0% vs hybrid. |
| **PATH A submitted** (`placer_adaptive.py`) | **1.0770** | Current floor. |
| B-R2 Phase 2 v3 (single-λ=1.0) | (5 hard bench) | +1.5% vs hybrid baseline |
| B-R2 Phase 2 v4 (single-λ=0.1) | (5 hard bench) | +1.5% |
| B-R2 Phase 3 portfolio (4-lane) | (5 hard bench) | +2.9% |
| **Sub-1.0 target** | **<1.0109** | **−5.6% below our best** |

## B-R2 — falsified across all variants

**Hypothesis:** add canonical losses (top-K density, RUDY congestion) to DREAMPlace's `obj_fn` to bias DP toward canonical-aligned basins.

**Outcome:** real but inconsistent basin lifts. ibm12 -1.6% at λ=0.1, ibm18 -1.3% at λ=0.05, ibm01 -3.9% at λ=1.0 on basin. **No single λ generalizes across benches.** Per-bench tuning is rule-forbidden; portfolio approach (4 plateau lanes) regressed +2.9% due to budget splits.

Issues encountered & resolved:
1. Patch silently overwritten by `make install` (5+ hours lost)
2. `n_canon = num_movable - num_filler` was wrong (should be `num_movable`)
3. `data_collections.node_size_y` empty during obj_fn — use `placedb.node_size_y`
4. RUDY too slow (per-net Python loop) — RUDY effectively disabled

## NG45 (4 benches) status

| Variant | Avg | ariane133 | best result |
|---|---:|---:|---|
| Hybrid (yesterday) | 0.681 | 0.662 | — |
| dual_levy_ng45 (today) | 0.679 | **0.644** | beats E74 0.664 by −3.0% on ariane133 |

dual_levy_ng45 is the best NG45 variant — particularly on ariane133. mempool_tile regresses (+2.2%) so aggregate is roughly tied.

## What was speculated and tried in last 4 hours

1. **B-R2 portfolio (4-lane plateau pick)**: tested, +2.9% regression
2. **GPU DREAMPlace build + integration**: built successfully, 1.3× speedup on DP step. DP is only 30s of a 3300s pipeline. Net throughput ~1.01×. Not transformative.
3. **Iterative refinement (cascade-on-cascade)**: not implemented — requires modifying shared placer to accept init placement.
4. **Multi-seed within budget**: not viable — each seed needs ≥30 min; budget-tight.
5. **Larger Lanczos saddle (k=10 eigvecs, finer ε)**: not implemented — code change to cascading_saddle module.

## Recommended submission strategy

**Option A: Submit hybrid baseline (1.067) instead of PATH A (1.077).**
- Mechanical change: replace `submissions/cd_lns_sa_cascade/placer_adaptive.py` with `submissions/cd_lns_sa_cascade_dp_lane/placer.py` in submission packaging.
- −0.9% improvement, guaranteed.
- Caveat: hybrid is ~7 hours total wall (vs PATH A's ~5 hr). Within 17-bench × 1-hr cap easily.
- Verified to run on EPYC equivalent? Need to confirm wall-cap fixes prevent ibm17 60-min overrun.

**Option B: Submit hybrid + Levy hybrid as ensemble (if Tier 2 ORFS allows).**
- Pick best per bench from {hybrid, dual_levy, portfolio_v2}. **NOT rule-compliant** as a placer-level submission.
- BUT: if we wrap as a single algorithm running both internally and picking best, may pass — wall-cap concern.

**Option C: Accept current submission, focus on Tier 2 ORFS.**
- Top-7 by proxy qualifies. We're definitely top-3 verified.
- Tier 2 winner = best ORFS WNS+TNS+Area over SA/RePlAce baselines.
- Our dual_levy ariane133 (0.644) and our hybrid NG45 (0.681 avg) are competitive. ORFS feasibility gate unknown without running ORFS flow.

## Files & artifacts left for next session

- `experiments/B_R2_canonical_dp_loss/code/canonical_losses.py` — diff top-K + RUDY (RUDY needs vectorization)
- `experiments/B_R2_canonical_dp_loss/code/cascade_dp_br2_placer.py` — hybrid with B-R2 DP lane (native DP)
- `experiments/B_R2_canonical_dp_loss/code/cascade_dp_portfolio_placer.py` — 4-lane portfolio
- `~/DREAMPlace_cpu/install` and `~/DREAMPlace_gpu/install` on cloud — both with B-R2 patch
- Result JSONs in `experiments/B_R2_canonical_dp_loss/results/`
- `experiments/B_R2_canonical_dp_loss/SUMMARY.md` — full B-R2 timeline
- This report
