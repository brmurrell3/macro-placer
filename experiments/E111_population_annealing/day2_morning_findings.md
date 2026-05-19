# v2 Day 2 Morning Findings — extended SA polish falsified

## Result: v2-ext = -0.07% on --fast (1 win, 3 losses)

| Bench | Base (Option C) | v2-ext (300s polish carved) | Δ % |
|---|---|---|---|
| ibm01 | 0.92686 | 0.88846 | -4.14% ✓ |
| ibm04 | 1.14136 | 1.15270 | +0.99% ✗ |
| ibm09 | 0.95240 | 0.97055 | +1.91% ✗ |
| ibm13 | 1.19866 | 1.20482 | +0.51% ✗ |
| **avg** | **1.05482** | **1.05413** | **-0.07%** |

All runs at 1500s budget per bench, parallel execution, fresh containers.

## Root cause: cascade-budget cuts cost more than polish recovers

The composer carved 300s out of `budget_seconds` for the post-cascade SA polish.
Empirically, on 3/4 benches the cascade's PRE-POLISH output regressed by 3.0–3.3%
compared to base's full-budget cascade, but the SA polish only recovered
0.4–2.3%. Net: -1 to -2% per bench.

| Bench | base cascade output | v2-ext cascade (cut budget) | v2-ext post-polish | regress | recover |
|---|---|---|---|---|---|
| ibm04 | 1.14136 | 1.17879 | 1.15270 | +3.3% | -2.21% |
| ibm09 | 0.95240 | 0.98491 | 0.97055 | +3.4% | -1.46% |
| ibm13 | 1.19866 | 1.23264 | 1.20482 | +2.8% | -2.26% |

ibm01 is the anomaly: v2-ext cascade reached 0.89086 (better than ~0.97
estimated for base cascade). This is likely a periphery-rejection artifact —
base's periphery push was accepted (raising 0.96 → 0.92686 reduction), v2-ext's
was rejected with 8 overlaps so it kept the deeper cascade output 0.89086.
Periphery acceptance correlates with stochastic overlap-generation behavior.

## Why "use leftover time for polish" doesn't fix this

Alternative: don't carve upfront, just use any leftover budget after cascade
finishes. Empirically, Option C exhausts (or overruns) its budget:

| Bench | budget | wall | over/under |
|---|---|---|---|
| ibm01 | 1500s | 1682s | +12% over |
| ibm04 | 1500s | 1489s | -0.7% under |
| ibm09 | 1500s | 1416s | -5.6% under |
| ibm13 | 1500s | 1457s | -2.9% under |

Only ibm09/ibm13 leave ~50-90s spare. At polish rates (0.4-2.3% lift in 300s),
50-90s would lift 0.07-0.7% — within variance.

## Decision: ship Option C unchanged

`submissions/cd_lns_sa_cascade_stacked_periphery/placer.py` at 1.0575 IBM avg /
0.6893 NG45. Already -1.8% from current rank 9 (1.0771). No new code or risk.

Falsified hypotheses (kept for the record):
- H1 LP-dual destroy ranking: signal within LNS variance floor
- H2 PA: 3.7% loss to SA-v2 at same budget
- H2 multi-start: ~0% inter-chain diversity benefit
- H2 extended-SA polish: -0.07% on --fast, regresses on 3/4 benches

## Path forward options

1. **Ship Option C, use day-2-evening for cross-platform validation + writeup.**
   Safe, guaranteed -1.8% improvement.

2. **One more breakthrough attempt: K-joint moves in polish.**
   E39 implements K-macro joint moves. SA inside the cascade uses per-axis
   moves. Replacing with K-joint in a post-cascade polish phase MIGHT escape
   single-macro local optima. Empirical risk: same budget-carving regression
   pattern likely repeats. EV estimate: ~30% chance of 0.5% lift,
   70% chance of further regression.

3. **More invasive: modify Option C's internal SA budget allocation.**
   SA inside cascade gets ~3.6% of total budget (~120s at 3300s). Increase to
   ~10% (330s) by stealing from less-impactful phases (E41 lane, portfolio
   saddle). Same total compute, redistribute. EV: marginal lift (~0.2%) but
   touches load-bearing code.

Going with option 1 unless redirected.
