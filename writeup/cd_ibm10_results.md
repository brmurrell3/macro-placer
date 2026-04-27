# E2: CD-Only Diagnostic on ibm10

Run timestamp: 2026-04-27 01:20:16
Wall budget: 2400 s   (CD-only; SDF init excluded)
Total sweeps: 13
Total accepted moves: 15000
Total per-axis probes: 71336
Golden-section fallbacks: 0
Final overlap count: 0

## Comparison Table

| Method | proxy | WL | Density | Congestion |
|---|---|---|---|---|
| SDF init (E2 start) | 1.4112 | 0.0687 | 0.7614 | 1.9235 |
| RePlAce baseline (avg, 17 IBM) | 1.4578 | – | – | – |
| DPO best_of_v2 (champion on ibm10) | 1.254 | 0.080 | 0.269 | 0.905 |
| **E2 final (CD-only, 2400s from SDF)** | **1.0632** | **0.0790** | **0.5700** | **1.3984** |
| Leaderboard target (avg, all 17) | 1.117 | – | – | – |

Improvement vs SDF init: **24.7%**

## Decision rule triggered

**E2 final ≤ 1.20:** CD-only is the answer. Recommend graduating to all 17 benchmarks and writing a CD-only placer.

## Trajectory (first/last sweeps)

| sweep | elapsed (s) | proxy | wl | density | congestion | accepted | probes | gs |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 1.4112 | 0.0687 | 0.7614 | 1.9235 | 0 | 0 | 0 |
| 1 | 206.6 | 1.1777 | 0.0886 | 0.6275 | 1.5505 | 4370 | 5536 | 0 |
| 2 | 407.5 | 1.1161 | 0.0835 | 0.5987 | 1.4663 | 2741 | 5536 | 0 |
| 3 | 601.0 | 1.0960 | 0.0821 | 0.5862 | 1.4415 | 1949 | 5536 | 0 |
| 4 | 786.9 | 1.0835 | 0.0815 | 0.5785 | 1.4255 | 1337 | 5536 | 0 |
| 5 | 977.4 | 1.0770 | 0.0804 | 0.5764 | 1.4170 | 1046 | 5536 | 0 |
| 6 | 1159.8 | 1.0726 | 0.0800 | 0.5760 | 1.4094 | 817 | 5536 | 0 |
| 7 | 1337.3 | 1.0694 | 0.0801 | 0.5729 | 1.4059 | 680 | 5536 | 0 |
| 8 | 1518.8 | 1.0664 | 0.0797 | 0.5711 | 1.4022 | 604 | 5536 | 0 |
| 9 | 1699.9 | 1.0649 | 0.0794 | 0.5701 | 1.4007 | 457 | 5536 | 0 |
