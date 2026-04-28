---
id: E17
name: multiseed_jitter
status: in_progress
parent: E9
created: 2026-04-28
decided: null
champion_at_time: 1.0990
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E17: multiseed_jitter

## Hypothesis
SDF init is fully deterministic — different RNG seeds produce byte-identical
placements because nothing in the optimization loop consumes randomness.
Adding Gaussian jitter to the init position before SDF gradient descent
samples *different basins* of the placement landscape. Best-of-K should
beat single-seed CDAdaptive on benchmarks where the deterministic basin
isn't globally optimal.

## Method
Wrap CDAdaptive's pipeline (SDF init → project_overlaps → IncrementalProxyEvaluator
→ run_cd_adaptive) in a multi-seed loop. For each benchmark, run K=3 times:
one deterministic baseline (`seed=42, jitter=0.0`, byte-identical to E9
champion) plus K-1 jittered runs (`seed=43,44`, `jitter=0.05 × canvas_diag`).
Return the placement with the lowest canonical compute_proxy_cost.

Default behavior of E9 / E12 champions is unchanged. The jitter parameter
on `SDFPlacer.__init__` defaults to 0.0; only this experiment passes >0.

Including the deterministic baseline in the pool guarantees best-of-K ≥
single-seed champion, eliminating regression risk; the jittered seeds are
pure exploration with a probabilistic upside.

## Kill gate
On --fast (4 benches), best-of-3 average must beat single-seed champion
(1.0990) by ≥ 0.5% AND show at least 2 of 4 jittered-seed wins. If
the deterministic baseline wins on every bench (jitter contributes nothing),
status: falsified.

Earlier --fast probe (run on prior session, K=3 wrapped around CDAdaptive E9
not E12): avg 0.9295 vs E9-mac 0.9442 = -1.56%, BUT -0.40% of that was
Mac→Windows float-difference luck on the deterministic, so true jitter
lift was -1.10% driven mostly by ibm04 seed=43 finding a -3.84% basin.
ibm09 and ibm13 jitter contributed nothing.

## Generalization check
Run --all to confirm fast-set lift wasn't ibm04-specific. NG45 hidden test
is the cross-domain check; multi-seed jitter is benchmark-agnostic so
should transfer.

## Outcome (filled when decided)
[pending]

## Pointers
- Code: `code/cd_adaptive_multiseed.py`
- SDFPlacer change (default-preserving): `macro_place/sdf_init.py`
  (added `init_jitter=0.0` kwarg)
- Prior session probe: see `results/experiment_log.jsonl` for entries
  tagged `multiseed_K3_*`.
