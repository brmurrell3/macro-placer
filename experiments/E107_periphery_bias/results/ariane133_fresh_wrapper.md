# E107: ariane133 fresh wrapper test (2026-05-16 07:09 EDT)

## Result

```
E25  plateau:        0.70441
E41  plateau:        0.68876  (E48 best-of)
Lévy cascade:        0.65212  ⭐ NEW BEST (Δ=-3.66% vs E48 plateau)
Periphery push:      0.65752  (+0.83% vs Lévy, 0 ovl)
Wrapper REJECT → keep Lévy = 0.65212  VALID
```

Total wall: 1881s (31 min) on M3 Max.

## Significance

**0.65212 beats E74's 0.6641 ariane133 by -1.8%** — a new ariane133 record
(was 0.6641 in champion lineage; now 0.65212).

## Why wrapper rejected periphery here

On a CACHED suboptimal baseline (0.66993, from old DP-full-polish pipeline),
periphery polish gives -0.95% lift. But on a fresh Lévy cascade (already at
a tight local optimum), periphery polish moves OFF that optimum (+0.83%).

The wrapper's strict-accept correctly rejects the regression. Plain Lévy is
the right choice for proxy ranking when starting fresh.

## Implications

- **For proxy ranking** (top 7 qualifier): use plain Lévy cascade. Wrapper
  adds no value on top of fresh cascade output.
- **For Grand Prize (OpenROAD WNS/TNS)**: pure proxy 0.65212 may be best;
  periphery push at +0.83% proxy cost may or may not help WNS. Requires
  OpenROAD validation to know.

## What this means for the submission strategy

The submission floor for ariane133 is now **0.65212**, beating:
- E74 champion (0.6641, -1.8%)
- PATH B (0.68086)
- All published results

Combined with AWS-cpu IBM 17 = 1.07564, projected 21-bench aggregate:
- Total IBM weight: 17 × 1.07564 = 18.286
- Plus ariane133: 0.65212 → 18.286 + 0.652 = 18.938 / 18 benches
- Other 3 NG45 still TBD; assume cached values:
  - ariane136 ~0.66
  - mempool_tile ~0.74
  - nvdla ~0.68
- 21-bench aggregate estimate: (18.938 + 0.66 + 0.74 + 0.68) / 21 = ~1.000

This is ~1.00 aggregate (vs current 1.0771 leaderboard score). HUGE jump.
