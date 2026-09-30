---
id: E162
name: v4_multiseed_basin
status: in_progress
parent: E127
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E162: v4_multiseed_basin — V4-Gaussian best-of-N seeds for hard benches

## Hypothesis

thinkorplace-v2's V4-Gaussian Adam descent uses a single seed (seed=42)
per attempt; retries only fire on overlap failure (seed + 100, seed + 200).
On hard benches (ibm12, ibm14, ibm17, ibm18) Adam may converge to a
suboptimal basin that's fixed by the initial randomness. Running V4 with
N=3 different seeds and picking the best canonical-proxy basin (before
CD polish) is a basin-attractor diversity test.

If multi-seed lifts ≥ 0.3% on at least one hard bench, basin sensitivity
is real and best-of-N is justified. If lifts are < 0.1% everywhere, the
V4 basin is seed-insensitive and the production single-seed approach is
optimal.

## Method

Three seeds: {42, 123, 999}. For each:
- Run V4-Gaussian descent with that seed, same hyperparameters as production.
- Legalize with greedy + project fallback (skip seed if legalize fails).

Pick the lane with lowest canonical proxy → CD polish.

Wall budget allocation:
- 3 × V4 descent (~150s each on hard bench) = 450s
- CD polish: remaining budget (target 600s+)
- Total budget: 1500s (matches production)

Compared to thinkorplace-v2's ~150s descent + 900s CD = 1050s used, this
shifts the budget toward basin search at the cost of CD polish wall.

## Kill gate

ibm17 smoke: best-of-3 final proxy ≥ thinkorplace-v2 baseline + 0.005
(no improvement within noise) → kill. ibm17 chosen as it's the densest/
hardest bench where seed sensitivity most likely.

If best seed = production seed (42) on every test bench, basin is
seed-insensitive → kill.

## Generalization check

`--fast` (4 benches): aggregate proxy ≤ thinkorplace-v2 baseline minus
0.2% AND no regression > 0.5%. If passes, run `--all` on hard subset
(ibm10, ibm12, ibm14, ibm15, ibm17, ibm18).

## Outcome (filled when decided)

**Smoke ibm17 PARTIAL** (2026-05-21):
- Per-seed pre-CD basin: seed=42:1.28618, seed=123:1.27019, seed=999:1.28781
- **Basin spread 1.24% between best (123) and worst (999)** — basin attractor IS seed-sensitive on ibm17
- PICK: seed=123
- CD polish 600s budget, post-CD final = **1.17154** (ovl=0, total wall=1085s)
- vs thinkorplace-v2 single-seed=42 baseline on ibm17: NOT directly verified;
  needs head-to-head run. If v2-baseline ibm17 ≈ 1.20 (handoff estimate),
  multi-seed gives ~-2.5% on this bench.

**Conclusion**: V4 basin attractor varies with seed on hard benches; pre-CD
basin diversity is real (1.24% on ibm17). Whether this translates to a
post-CD lift requires comparison vs single-seed baseline on the same bench.
NEXT STEP: run thinkorplace-v2 unchanged on ibm17 to lock baseline.

Kill gate (final >= v2_baseline + 0.005) cannot be evaluated without
verified single-seed baseline; status remains in_progress pending that.


## Pointers

- Code: `code/placer.py`
- Parent: thinkorplace-v2 (`submissions/thinkorplace-v2/placer.py`)
- Related: E135 (two-lane SDF/DPO, blocked); E161 (E135 unblock attempt)
