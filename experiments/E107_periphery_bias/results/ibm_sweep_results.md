=== IBM SWEEP RESULTS (α=0.01) — 6 cached benches ===

| Bench | Baseline | Control | Periphery (ovl) | Random (ovl) | Center (ovl) |
|---|---|---|---|---|---|
| ibm01 | 0.86134 | -0.05% (0) | -0.58% (15) | -0.49% (9) | -0.49% (25) |
| ibm09 | 0.78006 | +0.13% (0) | -0.26% (20) | -0.27% (25) | -0.14% (33) |
| ibm10 | 1.16367 | -1.41% (0) | +9.47% (38) | +6.36% (106) | +12.88% (110) |
| ibm12 | 1.11005 | -0.93% (0) | -1.19% (75) | -1.29% (67) | -1.20% (135) |
| ibm14 | 1.22679 | -0.52% (0) | -0.85% (78) | -1.12% (91) | -1.28% (160) |
| ibm17 | 1.44989 | -0.98% (0) | -1.44% (119) | -1.58% (166) | -1.59% (254) |

KEY: α=0.01 creates 15-254 overlaps on every IBM bench. CD polish can't resolve.
Wrapper's strict-accept correctly rejects all IBM periphery attempts.
ibm10 is uniquely vulnerable (+9.47% with periphery, much worse).

Direction signal: Periphery edge_dist is most-negative in 5/6 IBM benches
(direction informative even when overlaps exist).

DEPLOYMENT: wrapper SAFE everywhere; useful proxy benefit only on ariane133 (NG45).
