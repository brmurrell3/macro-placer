# E27 — basin persistence summary

THE GATE: do post-E25 plateaus all sit in one big basin (kill post-E25 line) or several deep basins (justify E28+ multi-day builds)?

Method: K diverse-init CD trajectories per benchmark; L2 single-linkage clustering of final placements at threshold = 5 % of canvas diagonal; verdict from cluster sizes + means vs E25 per-bench floor.

**Overall verdict across 4 benchmarks:** single_basin × 0, multi_basin × 0, ambiguous × 4.

**Verdict ambiguous:** rerun with longer CD budget or more diverse inits before deciding.


## ibm11

- E25 floor: **0.9136**
- Trajectories: 4
- Canvas diagonal: 72.9, clustering threshold: 3.6 (= 5 % of diag)
- **Verdict: `ambiguous`**

| cluster | size | mean proxy | std | min | max | vs E25 floor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.9295 | 0.0015 | 0.9283 | 0.9316 | +0.0159 |
| 0 | 1 | 1.3658 | 0.0000 | 1.3658 | 1.3658 | +0.4522 |

  - cluster 1: sdf/1=0.9316, sdf/2=0.9283, sdf/42=0.9288
  - cluster 0: greedy/0=1.3658

## ibm13

- E25 floor: **0.9766**
- Trajectories: 4
- Canvas diagonal: 79.2, clustering threshold: 4.0 (= 5 % of diag)
- **Verdict: `ambiguous`**

| cluster | size | mean proxy | std | min | max | vs E25 floor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1.0077 | 0.0026 | 1.0058 | 1.0114 | +0.0311 |
| 0 | 1 | 1.3918 | 0.0000 | 1.3918 | 1.3918 | +0.4152 |

  - cluster 1: sdf/1=1.0114, sdf/2=1.0058, sdf/42=1.0059
  - cluster 0: greedy/0=1.3918

## ibm14

- E25 floor: **1.2205**
- Trajectories: 5
- Canvas diagonal: 84.7, clustering threshold: 4.2 (= 5 % of diag)
- **Verdict: `ambiguous`**

| cluster | size | mean proxy | std | min | max | vs E25 floor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1.2616 | 0.0034 | 1.2585 | 1.2663 | +0.0411 |
| 0 | 1 | 1.9389 | 0.0000 | 1.9389 | 1.9389 | +0.7184 |
| 2 | 1 | 1.7996 | 0.0000 | 1.7996 | 1.7996 | +0.5791 |

  - cluster 1: sdf/1=1.2663, sdf/2=1.2585, sdf/42=1.2600
  - cluster 0: greedy/0=1.9389
  - cluster 2: uniform/42=1.7996

## ibm15

- E25 floor: **1.1797**
- Trajectories: 4
- Canvas diagonal: 95.6, clustering threshold: 4.8 (= 5 % of diag)
- **Verdict: `ambiguous`**

| cluster | size | mean proxy | std | min | max | vs E25 floor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1.2255 | 0.0037 | 1.2215 | 1.2304 | +0.0458 |
| 0 | 1 | 1.4996 | 0.0000 | 1.4996 | 1.4996 | +0.3199 |

  - cluster 1: sdf/1=1.2304, sdf/2=1.2215, sdf/42=1.2245
  - cluster 0: greedy/0=1.4996