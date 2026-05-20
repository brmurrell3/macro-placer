# V3Minimal breakthrough — 2026-05-19 21:55 EDT

## TL;DR

**Built `E111MinimalPlacer` — V3 (per-net trace congestion) + 4-min CD polish.
--all avg 1.030 (vs Option C 1.0575 = −2.58% lift), --ng45 0.6798 (−1.4%).
13/3/1 win/tie/loss vs Option C. Total wall ~1 hr vs Option C's 14 hr CPU.**

Launcher updated to V3Minimal. This is the new champion.

## How we got here today

1. **Profile (morning)**: Confirmed 94% of cascade time is Python CD polish.
2. **Subagent Path A (afternoon)**: Built per-net trace congestion (E111)
   that matches canonical ±15-25% (vs old bbox-uniform ±200-260% off).
   ibm17 raw 1.30 vs E110 1.76. ibm17+CD60s 1.246 vs Lane-4 default 1.324.
3. **V3Minimal placer (evening)**: Drop cascade entirely. V3 descent +
   CD polish only, 4-min budget per bench.
   --all 1.030, NG45 0.680.

## V3Minimal --all per-bench breakdown (vs Option C)

```
ibm01: 0.85375  Δ -0.09%  WIN
ibm02: 1.00746  Δ -1.16%  WIN
ibm03: 0.92236  Δ -1.01%  WIN
ibm04: 0.94964  Δ -1.09%  WIN
ibm06: 1.09082  Δ -2.11%  WIN
ibm07: 1.05452  Δ +0.20%  TIE
ibm08: 1.06837  Δ -2.05%  WIN
ibm09: 0.79013  Δ -4.28%  WIN
ibm10: 1.02292  Δ +2.19%  LOSS
ibm11: 0.82669  Δ -4.20%  WIN
ibm12: 1.19464  Δ -0.02%  TIE
ibm13: 0.89110  Δ -4.89%  WIN
ibm14: 1.14114  Δ -4.25%  WIN
ibm15: 1.08568  Δ -5.59%  WIN  ← big
ibm16: 1.09170  Δ -3.19%  WIN
ibm17: 1.26991  Δ -4.06%  WIN  ← hard bench!
ibm18: 1.25316  Δ -6.61%  WIN  ← hardest bench, biggest lift!

avg:   1.03024  Δ -2.58%  (13W / 3T / 1L)
```

## V3Minimal NG45

| Bench | V3Min | Option C | Δ |
|---|---:|---:|---:|
| ariane133 | 0.6731 | 0.6641 | +1.36% |
| ariane136 | 0.6700 | 0.6518 | +2.79% |
| mempool_tile | 0.7089 | 0.7376 | −3.89% |
| nvdla | 0.6671 | 0.6716 | −0.67% |
| **avg** | **0.6798** | **0.6893** | **−1.38%** |

## What this means for the leaderboard

- Public leaderboard: Carrotato 0.967, vmallela 1.011, our submitted 1.077
- M3 number 1.030 → EPYC ~1.05 (with 2.4% variance)
- Should land us around rank 5-7 (between Cezar 1.037 and Place,Route,Roll 1.059)

## Currently iterating in parallel

1. **V3Min-480s (8-min budget)**: more CD polish time. ETA ~22:30.
2. **V3Min-multiseed (3 seeds)**: variance reduction. ETA ~22:55.
3. **Path B v2 (subagent)**: Xplace-style optimizer on V3 proxy.

## Next steps

- Tomorrow AM: EPYC cross-validation on best variant
- Tomorrow PM: final tweaks
- Wed: submit by 11:59 PT
