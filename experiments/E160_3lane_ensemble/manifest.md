---
id: E160
name: 3lane_ensemble
status: deferred
parent: E155
created: 2026-05-21
decided: null
champion_at_time: 0.98387      # v2-extCD EPYC --all combined
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E160: 3-lane parallel ensemble with diverse search topologies
# DEFERRED 2026-05-21 — E155 2-lane --all is the actual shipping path.

## Hypothesis

E155 demonstrated the parallel-ensemble idea works (Lane B reaches
saddle-lifted basin in parallel with Lane A) but had two issues:

1. Master budget (1700s) was tight for ibm17 — Lane A's CD polish was
   prematurely terminated, defeating the ensemble.
2. Only 2 structurally distinct lanes (v2-extCD vs E138). With per-bench
   variance dominating, adding a 3rd distinct lane should improve
   plateau-pick coverage of the basin manifold.

Hypothesis: a 3-lane parallel ensemble with (v2-extCD + E138 + Lane C
with aggressive overlap_lambda=50 schedule) gives -0.5% to -1% lift
over v2-extCD's 0.98387 EPYC by capturing structurally distinct basins.
Offline analysis of v2/E138 split already projects -0.33% lift; a third
lane should add diversity for a more aggressive lift.

## Method

In-process master spawns 3 subprocesses, one per lane:

- **Lane A** (v2-extCD): V4+Gauss descent + greedy_legalize +
  project_overlaps + CD polish 700s. rng_seed=42.
- **Lane B** (E138): V4+Gauss + legalize + CD polish 700s + bounded
  saddle escape 120s + CD2 polish 240s. rng_seed=142.
- **Lane C** (NEW — aggressive overlap): V4+Gauss with
  overlap_lambda_end=50, overlap_ramp_pct=0.6 (faster ramp), 400 steps
  + legalize + CD polish 700s. rng_seed=242.

Master budget 2700s (45 min) to safely accommodate the longest lane.
On 8-vCPU EPYC, 3 subprocesses get ~2.5 vCPU each (CPU-bound CD polish
will mostly NOT contend because Python is GIL-bound to 1 thread per
process). GPU shared but mostly idle during CD polish.

Each lane's subprocess re-loads benchmark/plc cleanly, pickles result.
Master joins all 3, picks canonical-best across {A, B, C}.

## Kill gate

EPYC ibm17 smoke:
- Final proxy < 1.180 → continue to --all.
- 1.180 ≤ proxy < 1.183 → marginal (within noise of v2-extCD 1.18269).
  Recommend ship only if --all margin clean.
- proxy ≥ 1.183 → kill, ship v2-extCD.

Also kill if any lane fails to produce zero-overlap result (no working
fallback in C1).

## Generalization check

If smoke passes, run `--all --json --hypothesis E158_full` on EPYC
2-way parallel (each evaluate worker gets 4 vCPU, each lane subprocess
within evaluate gets ~1.3 vCPU). Promote only if --all avg < v2-extCD
0.98387 by >0.3% (real-lift threshold per M3 variance memo).

## Outcome (filled when decided)

[Pending smoke result.]

## Pointers
- Code: `code/placer.py` (master + spawn orchestration)
- Code: `code/e158_lane_worker.py` (subprocess entry)
- Parents:
  - v2-extCD (`submissions/thinkorplace-v2/placer.py`)
  - E138 (`experiments/E138_bounded_saddle/code/placer.py`)
  - E155 (`experiments/E155_parallel_ensemble/code/placer.py`) — 2-lane variant.
- Reuses bounded_saddle.py from E138 unchanged.
