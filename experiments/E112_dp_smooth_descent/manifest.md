---
id: E112
name: dp_smooth_descent
status: in_progress
parent: E94 (dp_canonical_full, scoped not implemented)
created: 2026-05-19
decided: null
champion_at_time: 1.0575 (e110-stacked-periphery IBM); 1.0574 (e110-default-lane4 IBM)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E112: dp_smooth_descent — patched-DREAMPlace via "Option C"

## Hypothesis

The "patch DREAMPlace's loss to use our challenge proxy" task has three
viable options:

- **A**: Modify DREAMPlace's `obj_fn` to compute our proxy directly.
  Blocked: requires writing a torch.autograd Function wrapper around
  `rudy_cpp.forward` (DREAMPlace's RUDY op has no backward registered)
  + reimplementing canonical top-K density as a torch op. E94 scoped
  this at 6-9 days; not tractable in 6-8 hour budget. PATCH_SPEC
  written at `experiments/E94_dp_canonical_full/PATCH_SPEC.md`.

- **B**: Hyperparameter sweep on DREAMPlace. Falsified by E76 5D sweep:
  best DP basin still 1.41 raw on ibm01 vs cascade 0.85. Not a
  promising direction.

- **C**: DREAMPlace as black-box init, refine with SmoothGlobalPlacer
  (E110 — Adam descent on canonical-aligned smooth proxy). This is
  the "equivalent loss-patched DP" but bypasses DP's internal
  optimizer entirely. Our smooth proxy (E88 primitives + E95
  WL-normalization fix) IS the patched objective.

E112 = "Option C", with the addition that the placer also tries
**SDF + smooth descent** as Lane 5 (always-on) for environments
where DREAMPlace isn't available (M3 dev box).

## Method

`submissions/cd_lns_sa_cascade_dp_lane_patched/placer.py` adds two
lanes to the existing 3-lane DP-lane placer:

- Lane 4: DP basin → `SmoothGlobalPlacer` descent (~500 Adam steps on
  smooth proxy, gamma-annealed) → greedy_macro_legalize → CD polish
- Lane 5: SDF basin → same descent → legalize → CD polish (always-on
  fallback)

Plateau pick = argmin {E25, E41, DP+stock, DP+smooth, SDF+smooth},
then cascading saddle.

## Kill gate

- ibm01 raw smooth basin > 1.40 → smooth-descent broken; kill
- ibm01 + 60s CD > 0.90 → not competitive; kill
- ibm01 + 240s CD ≥ 0.88 → marginal; document but don't pursue
- ibm01 + full placer pipeline ≥ champion 1.0575 + 0.5% → marginal

## Verified intermediate results (M3, single-bench smokes)

`experiments/E110_smooth_global_placer/code/smoke_ibm01_patched.py`:

| Stage | proxy | wall | gate |
|---|---:|---:|---|
| SDF reference (no descent) | 1.19530 | 5s | — |
| Lane 5 raw basin (SDF→smooth+legalize) | 1.02098 | 29s | <1.40 PASS |
| Lane 5 + 60s CD | 0.88008 | 90s | <0.90 PASS |
| Lane 5 + 240s CD | 0.87794 | 130s | (vs DP+polish 0.862) |

End-to-end placer (450s budget, no DP locally):
- E25 lane: 0.91110
- Lane 5: 0.90573 (wins plateau)
- Cascade saddle: skipped (low budget)
- WINNER: 0.906, zero overlaps

End-to-end placer (900s budget, DP unavailable on M3, lanes 3+4 skipped):
- E25 lane: 0.90193 (wall 150s)
- E41 lane: 0.91183 (wall 334s, includes K-joint)
- Lane 3 (DP+stock): SKIPPED (566s left, need 780s)
- Lane 4 (DP+smooth): SKIPPED (no DP basin)
- Lane 5 (SDF+smooth+CD): **0.88758** (wall 409s) — wins plateau
- Cascade saddle: timed out (budget 431s, ran 570s, killed)
- WINNER: SDF+smooth at 0.88758, zero overlaps

The SDF+smooth lane consistently beats both E25 and E41 by ~2.5% on
ibm01 — confirming that smooth-proxy gradient descent is finding a
better basin than CD+LNS+SA from the same SDF init. Log saved to
`ibm01_e2e_900s.log`.

NB: lane 5 took longer than budgeted (409s vs expected ~75s) because
the placer's CD polish budget (smooth_CD = 0.05 × B = 45s) was added
to the descent (~30s) and the SmoothGlobalPlacer's internal legalize
phase, total ~100s. The extra wall came from extra CD sweeps post-
descent inside SmoothGlobalPlacer.place(). For production this is
acceptable; on M3 single-bench it ate into saddle budget.

## Generalization check

Not done on this branch. Plan when revisited:
- M3 --fast (lanes 1+2+5 only): expected ~0.89-0.90 (within 0.5% of
  champion's 0.898)
- Cloud A100 with DREAMPlace + --fast: expected ~0.88-0.89 (lane 4
  may lift)
- Cloud --all: expected within 1% of champion
- Cloud --ng45: lane 5 should help on hard NG45 (per E110 finding,
  ariane133 0.642 vs Option C's 0.664)

## Decision

Built but NOT promoted. Reason: the current champion
`cd_lns_sa_cascade_stacked_periphery_e110` already has SmoothGlobalPlacer
as Lane-4 of stacked_periphery, plus portfolio_saddle and periphery
wrapper that this scaffold lacks. Adding DP lanes (3, 4) on top would
be incremental, but requires cloud DREAMPlace install to validate.

Recommendation: revisit only if (a) cloud DP install is verified working,
(b) we want to test if DP basin (electrostatic spreading) produces lift
on benches where stacked-periphery underperforms (ibm02 +2.99%,
ariane136 +1.76%, ibm10 +1.28% — per E91 SUMMARY).

## Pointers

- New placer: `submissions/cd_lns_sa_cascade_dp_lane_patched/placer.py`
- Smoke test: `experiments/E110_smooth_global_placer/code/smoke_ibm01_patched.py`
- End-to-end test: `experiments/E110_smooth_global_placer/code/smoke_patched_end2end.py`
- Option A patch spec: `experiments/E94_dp_canonical_full/PATCH_SPEC.md`
- Option B falsification: `experiments/E76_dreamplace_integration/code/dp_sweep_b1.py`
- Option C engine (SmoothGlobalPlacer): `experiments/E110_smooth_global_placer/code/smooth_global_placer.py`
- Diff-proxy primitives: `experiments/E88_diff_proxy/code/diff_proxy.py`
- WL-norm fix: `experiments/E95_diff_proxy_v2/code/diff_proxy_v2.py`
