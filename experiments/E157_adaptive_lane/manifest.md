---
id: E157
name: adaptive_lane
status: in_progress
parent: E138
created: 2026-05-21
decided: null
champion_at_time: 0.98379
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E157: adaptive_lane

## Hypothesis

Different placer variants win different IBM benchmarks (v2-extCD wins
10/17, E138 wins 7/17). If a structural attribute predicts which lane
wins, an adaptive selector that probes the benchmark at runtime and
dispatches to the right pipeline beats either single placer alone.

## Method

For each of the 17 IBM benches, compute structural attrs via
`macro_place/loader.py`: num_macros, num_hard, num_soft, num_nets,
num_ports, grid_rows, grid_cols, canvas_area, hard_area, hard_density,
total_density, avg_net_size. Search single-threshold and
two-attribute rules for the rule that minimizes 17-bench avg using
existing per-bench EPYC --all data for v2-extCD vs E138.

If a strong rule exists, build `code/placer.py` which:
1. Loads the benchmark
2. Computes the attribute
3. Dispatches to `submissions/thinkorplace-v2/placer.py` or
   `experiments/E138_bounded_saddle/code/placer.py` accordingly

This is structural-attribute selection, NOT per-bench-name tuning.

## Kill gate

If the discovered rule's offline-projected avg is within run-to-run
noise floor (~0.5%) of v2-extCD or E138 baseline AND
leave-one-out validation rejects the rule on >25% of held-out benches,
declare falsified.

## Generalization check

LOO single-attr CV on the 17 benches. M3 smoke on one v2-winning bench
(ibm04) to verify lane selection picks v2.

## Outcome

**Single-attr signal found: `hard_density` (sum of hard macro areas /
canvas area).**

Best rule: `hard_density < 0.50 -> v2-extCD lane, else E138 lane`.

Offline projection on 17-bench data:
- v2-extCD-only       avg = 0.98388
- E138-only           avg = 0.98379
- Adaptive (HD<0.50)  avg = 0.98143  (-0.25% vs v2, -0.24% vs E138)
- Per-bench oracle    avg = 0.98065  (theoretical ceiling)

Rule recovers 79% of the oracle gap (0.98388 -> 0.98143 vs ceiling 0.98065).
14/17 correct on full-data fit, 13/17 correct on LOO held-out
(misses: ibm03 +0.0051, ibm07 +0.0067, ibm15 +0.0015, ibm18 +0.0051 —
all sub-0.7% diffs, within run-to-run noise).

Two-attribute search returns `(grid_cols < 55) AND (hard_density < 0.50)`
at 16/17 fit / projected avg 0.98104 (-0.29%) but the extra clause
catches only ibm15 — likely overfit (margin 0.14%, well below noise).

**Conclusion**: signal exists but lift is BELOW M3 run-to-run noise floor
(0.36% per memory). 8 of 17 benches have margins ≤ 0.5% (below noise);
only 4 benches have margins > 1%. The rule's defensive failure mode
(picks v2 when uncertain) bounds downside.

EPYC smoke on ibm04 (clear v2 win): rule correctly dispatches to v2 lane.

**Recommendation**: marginal lift, do NOT promote as primary. Useful
as a backup/diagnostic. Keep adaptive logic shelved; ship v2-extCD or
E138 directly. Specifically, given the deadline pressure and noise
floor, the v2-vs-E138 choice should be made by whichever has the BETTER
EPYC variance characteristics on the judge machine, not by per-bench
heuristic that gains only 0.22% LOO.

## Pointers

- Code: `code/placer.py` (selector); `code/probe_attrs.py` (attr probe);
  `code/analyze.py`, `code/analyze2.py`, `code/loo.py` (offline analysis).
- Attrs: `attrs.json` (full per-bench attribute dump).
- Source pipelines:
  - `submissions/thinkorplace-v2/placer.py` (v2-extCD lane)
  - `experiments/E138_bounded_saddle/code/placer.py` (E138 lane)
- Discussion: this manifest's Outcome section.
