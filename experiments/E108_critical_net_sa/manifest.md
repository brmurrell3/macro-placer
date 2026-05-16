---
id: E108
name: critical_net_sa
status: in_progress
parent: E25
created: 2026-05-16
decided: null
champion_at_time: 1.06650
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E108: critical_net_sa

## Hypothesis
HPWL is dominated by high-degree nets. The current SA polish (`run_sa_polish_v2`) samples macros uniformly over `hard_movable`, treating a macro on a 200-pin clock net the same as a macro on a 3-pin signal net. Weighting the SA macro-selection by `Σ_j∈nets(i) net_degree[j]²` concentrates move proposals on macros with the most leverage on HPWL. The choice is bench-agnostic (same heuristic everywhere) but adapts to each bench's net topology automatically — not per-benchmark tuning.

## Method
Modify `run_sa_polish_v2` to accept an optional `macro_weights` parameter. Pre-compute `w[i] = Σ_j∈nets(i) deg(j)²` from `benchmark.net_nodes` once. Sample `idx` via `rng.choice(hard_movable, p=w_normalized)` instead of uniform `rng.integers(0, len(hard_movable))`. Everything else identical. Apply to E25 SA polish first (highest impact since it owns most of the SA wall time in cascade_dp_lane).

## Kill gate
- --fast aggregate worse by >0.5% vs E25 baseline.
- OR: ibm10 / ibm14 individually worse by >1.5% (the high-net-count IBM designs where this should help most).

## Generalization check
If --fast passes, run --all. If --all avg < 1.085 (within 0.4% of E48 1.08151), run --ng45 and check ariane133 ≥ 0.66 (not catastrophic regression).

## Outcome (filled when decided)
[Empty]

## Pointers
- Code: `code/cd_lns_sa_critnet.py`, `code/critical_net_sa.py`
- Results: `results/`
- Discussion: TBD
