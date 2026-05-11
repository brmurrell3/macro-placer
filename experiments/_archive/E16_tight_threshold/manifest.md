---
id: E16
name: tight_threshold
status: marginal
parent: E9
created: 2026-04-27
decided: 2026-04-28
champion_at_time: 1.1055
outcome: 1.1025
champion_delta: -0.0030
graduated_to: null
superseded_by: null
---

# E16: tight_threshold

## Hypothesis
CDAdaptive's analysis showed every hard benchmark exited with last-3-sweep
deltas in the 0.001–0.005 range — i.e. just below the 0.005 plateau
threshold but each sweep still carrying ≈ 0.001 of proxy. Tightening the
threshold to `0.001` and lifting the cap to `7 200 s` should capture those
tail sweeps. Single global hyperparameter swap, applied to every benchmark.

## Method
Reuse the CDAdaptive control loop unchanged. Override only the global
defaults: `plateau_threshold = 0.001` (was `0.005`) and
`hard_cap_s = 7 200` (was `3 600`). No per-benchmark tuning.

## Kill gate
Avg `--all` proxy must drop by ≥ 0.005 vs CDAdaptive (1.1055 → ≤ 1.1005)
to count as a non-marginal win and graduate. Anything smaller is
formally marginal; champion is not promoted.

## Generalization check
Run on the 4 public NG45 designs (ariane133/136, mempool_tile, nvdla) —
proxy on those should also drop or hold steady, not regress. (Pending;
NG45 access not available locally as of decision date.)

## Outcome (filled when decided)
**Marginal — kept as a one-datapoint sensitivity result; champion not
promoted.**

| Metric | CDAdaptive (champion) | E16 | Δ |
|---|---:|---:|---:|
| Avg proxy `--all` | 1.1055 | **1.1025** | **−0.0030 (−0.27 %)** |
| Total wall | 4.85 hr | 7.1 hr | +46 % wall for −0.27 % proxy |
| Overlaps | 0 / 17 | 0 / 17 | clean |

14 wins, 1 tie (ibm15), 1 small regression (ibm13 +0.0013). Strict kill
gate said ≥ 0.005 avg drop required; we got 0.003. Decision: keep E16 as
a sensitivity datapoint and an existence proof that the tail of CD's
descent is not flat; do not promote because the proxy/wall trade is poor
and the gate was authored knowing this kind of result was likely.

ADR-004 (in progress) captures the reasoning behind staying at the
conservative-defaults champion.

## Pointers
- Code: `code/cd_adaptive_e16.py`.
- Results: `results/CDAdaptiveE16Placer_20260428_050235.json`.
- Discussion: `writeup/evidence.md` §7.6; `findings.md` "Verified score
  lineage" row 2; `experiments_overnight.md` E16 block; ADR-004 (in
  progress, sibling agent).
- Parent: `experiments/E9_plateau_detection/manifest.md`.
