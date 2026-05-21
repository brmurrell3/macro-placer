---
id: E142
name: kpin_restart
status: falsified
parent: thinkorplace-v2
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.987   # combined 21-bench avg of stacked_periphery (Option C)
outcome: 0.9212   # ibm04 smoke; tied with v2 baseline 0.9210 within noise
champion_delta: null
graduated_to: null
superseded_by: null
---

# E142: kpin_restart

## Hypothesis
After v2's V4+Gaussian descent → CD polish has settled the placement into a
local minimum, a *small* perturbation that re-randomizes K=15 hard movable
macros (NOT a spatially clustered LNS destroy) preserves the global basin
found by descent while opening up alternative local arrangements. Re-
legalizing and re-polishing should let CD find a strictly lower plateau on
at least some fraction of iterations.

Distinct from E130 cascade_multirestart (which redoes the whole descent for
each seed); distinct from group-LNS (which removes spatially-coherent
clusters). K-pin random restart is the smallest perturbation that still
crosses a basin boundary in the CD-move graph.

## Method
Pipeline = thinkorplace-v2 with a K-pin restart loop inserted between CD1
and CD2:

  1. V4+Gaussian descent → greedy legalize → project_overlaps. (same as v2)
  2. CD polish (~300 s) to reach the local minimum.
  3. **K-pin restart loop** (3 iterations × 150 s each):
     - Pick K=15 random hard movable macros (uniform without replacement).
     - Reset each to a uniformly random valid position in
       `(half_w, canvas_w - half_w) × (half_h, canvas_h - half_h)`.
     - `greedy_macro_legalize` → `project_overlaps` to recover legality.
     - CD polish 150 s.
     - Accept iff `compute_proxy_cost` strictly improves over current best.
  4. Final CD polish (~200 s).

Total wall budget 1500 s/bench.

## Kill gate
- Smoke (`ibm04` only): if final proxy >= v2 baseline + 1 % AND zero
  iterations accept → falsify (perturbation too large, never recovers).
- `--all` (17 IBM): if avg proxy >= 0.985 (worse than v2 on M3) → falsify.
- If any iteration ACCEPT signal appears on smoke (>= 1 of 3 accepted by
  canonical) → graduate to `--all`.

## Generalization check
Must not regress NG45 (4 designs) by more than +0.5 % vs v2.
Submission-day fallback: if K-pin lifts but NG45 regresses → ship v2.

## Outcome (filled when decided)
**Falsified on ibm04 smoke (2026-05-21).** Final proxy 0.92120 on M3, vs v2
baseline ~0.9210 (most recent: 0.92104 / 0.92099 / 0.92749 — run-to-run M3
variance ~0.36%). E142 lands within 0.02% of v2 = indistinguishable from
the noise floor; well inside the kill gate.

Per-iter signal (all REJECTED by canonical):
- iter 1 (K=15): pre_polish 0.97664 → polished 0.93656; reject vs 0.92485
- iter 2 (K=15): legalize+project left 1 ovl → SKIPPED
- iter 3 (K=15): pre_polish 0.96365 → polished 0.92944; reject vs 0.92485

Wall: descent 17s + CD1 113s + K-pin 158s + CD2 62s = 360s total. CD1 did
the heavy lifting (1.08959 → 0.92485 = −0.165). K-pin spent 158s and
returned 0.0 lift.

Failure mode: 150s polish recovers most of the perturbation cost
(K=15 perturb adds ~0.05 proxy; 150s polish removes most but not all),
landing within 1.3% of the CD1 baseline but never beating it. The
"alternative local arrangement" thesis is false on this bench: the
basin around CD1's minimum is *not* multimodal at K=15 granularity, OR
the 150s polish is too short to fully recover. Either way K-pin is a
net loss vs spending those 158s on more CD polish.

Compare to siblings on ibm04 smoke (same M3, same date):
- E141 group_lns: 0.91551 (lighter weight, +0.7% lift over v2)
- E147 lp_refine: 0.91927
- E139 sa_swap: 0.92042
- **E142 kpin_restart: 0.92120 (no lift)**
- v2 baseline: ~0.92100

No reason to run --all — kill gate triggered.

## Pointers
- Code: `code/placer.py`
- Smoke log: `/tmp/e142_smoke.log`
- Smoke result: `results/smoke_ibm04.json` (parsed from harness)
