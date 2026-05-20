# cd_lns_sa_cascade_dp_lane_patched

**Status:** scaffold / research, 2026-05-19.

Variant of `cd_lns_sa_cascade_dp_lane` that adds two extra basin lanes:

| Lane | Source | Polish | Notes |
|------|--------|--------|-------|
| 1    | SDF (E25)         | CD + LNS + SA        | always |
| 2    | DPO (E41)         | CD + LNS + SA + K-joint | always |
| 3    | DREAMPlace        | CD + LNS + SA (stock) | if DP available |
| **4**| **DREAMPlace**    | **SmoothGlobalPlacer descent + CD** | **NEW; if DP available** |
| **5**| **SDF**           | **SmoothGlobalPlacer descent + CD** | **NEW; always-on** |

Plateau pick = argmin over all valid lanes, then cascading saddle.

## Motivation (Option C from the patched-DREAMPlace investigation)

DREAMPlace's native loss is `HPWL + density_weight * eDensity`. That is
structurally misaligned with our challenge proxy
`HPWL + 0.5*density + 0.5*congestion`. Three options to bridge:

- **A**: Patch DREAMPlace's `obj_fn` to use our proxy directly.
  Requires writing a differentiable TILOS-RUDY congestion term in C++
  (DREAMPlace's `rudy.Rudy` op has no autograd backward). Engineering
  cost 5-10 days. Scoped as E92/E93/E94 but not implemented.
- **B**: Tune DREAMPlace hyperparameters until basin matches.
  Falsified by `experiments/E76` 5D sweep — best basin still 1.41
  raw on ibm01 vs cascade 0.85.
- **C (this placer)**: Run DREAMPlace as a black-box init, then refine
  with our smooth proxy + Adam via `SmoothGlobalPlacer` (E110).

Option C is the most tractable. Even if DREAMPlace isn't available
locally (M3 only ships x86_64 .so), lane 5 (SDF → SmoothGlobalPlacer)
gives the same gradient-based basin from a different init. The two
lanes together (4 + 5) explore "smooth-proxy descent" basins from
both DP and SDF starts.

## Smoke test (ibm01, 2026-05-19, M3)

`experiments/E110_smooth_global_placer/code/smoke_ibm01_patched.py`
- Reference: SDF + project_overlaps        → 1.19530
- Lane-5-equivalent raw basin              → 1.02098  (PASS gate <1.40)
- Lane-5-equivalent + 60s CD               → 0.88008  (PASS gate <0.90)
- Lane-5-equivalent + 240s CD              → 0.87794  (vs DP+polish 0.862)

End-to-end ibm01 (450s budget, no DP available locally):
- E25 lane: 0.911
- Lane 5 (SDF+smooth+CD): **0.906** — wins plateau
- WINNER: 0.906, zero overlaps

## Fallback behavior

- `DREAMPLACE_ROOT` not set or install missing → lanes 3+4 skipped,
  lanes 1, 2, 5 still run. This is the M3 dev path.
- Lane 5 (SDF + smooth descent) ALWAYS runs; provides the
  gradient-basin contribution even without DREAMPlace.

## Relation to Option C (current champion as of 2026-05-18)

The current champion `cd_lns_sa_cascade_stacked_periphery_e110` uses
SmoothGlobalPlacer as Lane-4. This new placer adds DREAMPlace as a
SIXTH source (via lanes 3 + 4 when DP is available). The expected
relationship:

| Bench class | This placer vs e110-stacked-periphery |
|---|---|
| Without DREAMPlace (M3, no DP install) | Same Lane-5 contribution; slightly fewer downstream stages → ~same |
| With DREAMPlace (cloud A100) | Strictly more options → ≥ e110-stacked-periphery |

NOT promoted to champion. Verify on cloud (EPYC + DREAMPlace) before
considering for submission.
