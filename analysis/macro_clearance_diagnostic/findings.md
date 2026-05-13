# Macro clearance diagnostic — NG45 results

Generated 2026-05-13 on M3 Max with `cascade_adaptive` placements
(short-budget 600 s runs, representative of the submission entry).

## Aggregate

| Design | n_hard | n_pairs | proxy | viol pairs | viol % | avg push (μm) | max push (μm) | push / canvas |
|--------|-------:|--------:|------:|-----------:|-------:|--------------:|--------------:|--------------:|
| ariane133 | 133 | 8 778 | 0.6818 | 207 | 2.36 % | 4.51 | 6.00 | 0.22 % |
| ariane136 | 136 | 9 180 | 0.6700 | 210 | 2.29 % | 4.18 | 6.00 | 0.20 % |
| mempool_tile | 20 | 190 | 0.7374 | 13 | 6.84 % | 4.88 | 6.00 | 0.39 % |
| nvdla | 128 | 8 128 | 0.6795 | 136 | 1.67 % | 4.21 | 6.00 | 0.14 % |

Every design: **zero hard-macro overlaps**, max push **exactly 6.00 μm**
(= (12 - 0)/2, i.e., the worst case is two touching macros each shifted
by 6 μm).

## Interpretation

- **Push magnitude is bounded and small.** Max 6 μm, avg 4–5 μm. Canvas
  displacement fractions are all under 0.4 %.
- **mempool_tile has the highest violation rate** (6.84 %) because it
  has only 20 macros on an 885 × 885 μm canvas — small designs tend to
  pack more tightly when cost-pressured by CD.
- **nvdla has the lowest** (1.67 %) — largest canvas (2128 μm side),
  more room for the placer to spread.

## Tier 2 risk assessment

Without local OpenROAD we can't directly measure WNS/TNS impact. The
geometric perturbation from the ORFS push is small (< 0.4 % of canvas
diagonal across the board), which usually correlates with bounded WL /
density / congestion change at the routed level. Specifically:

- **Most likely outcome:** ORFS push moves 1.7–6.8 % of macros by ≤ 6 μm.
  Routing recovers, WNS/TNS stays within baseline range. **Pass Tier 2
  feasibility on all 4 public designs.**
- **Tail risk:** on a hard timing path, a pushed macro lands in a more
  congested region → routing detour → WNS regression beyond baseline.
  No data to estimate probability locally.

## Recommendation

**Submit as-is.** The push amounts are too small to warrant a
clearance-enforcement pass in the placer — adding one would risk
degrading the proxy (1.0771 IBM / 0.6870 NG45) to maybe fix a Tier 2
issue that probably doesn't exist. The 1-day setup cost of local
OpenROAD doesn't beat the expected information gain.

If we wanted to be paranoid:

1. Run cascade with a *post-placement clearance enforcement pass* that
   iteratively pushes any pair < 12 μm to exactly 12 μm and re-validates
   proxy + overlaps. Check that proxy doesn't degrade > 0.5 %.
2. If proxy doesn't degrade, ship that version — full Tier 2 control.
3. If it does, accept the current state.

This is ~1 day of engineering. Worth it only if some other signal
suggests the push will hurt timing — which the current data does not.

## How to reproduce

```bash
uv run python analysis/macro_clearance_diagnostic/diagnostic.py \
    --bench ariane133            # runs cascade_adaptive, then diagnoses
uv run python analysis/macro_clearance_diagnostic/diagnostic.py \
    --bench ariane133 --placement <path.pt>   # use cached
```
