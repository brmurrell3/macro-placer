---
id: E72
name: iter_block_pair
status: marginal
parent: E61 V2 (block crossover, marginal --all -0.07%), E69 SP-search (per-pair, marginal --fast -0.10%), E48 (champion)
created: 2026-05-04
decided: 2026-05-04
champion_at_time: 1.08151 (E48 hybrid)
outcome: **MARGINAL 2026-05-04 11:35 EDT.** Block-level (E61 V2) crossover failed across iterations because cached E25/E41 placements drift ~0.07% from E61 V2 historical fresh runs, and crossover-feasibility/quality is fragile to that drift (0/4 iters feasible on ibm01/ibm12 with quadrant_pick seeds). FALLBACK was iter-E69 (multi-round per-pair SP-swap with re-encoding). With v2 fix (use best-seen as next round start, avoid polish regression compound) and 6 rounds, ibm01 reaches 0.88703 = **−0.290% lift over E25 start** — close to promotion threshold. ibm04/09/12 reach −0.063% to −0.190% per bench (single E69 with extended budget actually beats iter on these where polish hurts). Aggregate --fast lift ≈ −0.156% over fresh-best — still sub-promotion (0.30%). KEY FINDING: per-pair SP-swap with multi-round polish-induced topology change (iter-E69) DOES compound on benches where polish helps, beating single-round E69 by 50%; on benches where polish hurts, single E69 with full disagreement-set coverage is best.
champion_delta: -0.0015 (-0.156%) on --fast aggregate (sub-promotion)
graduated_to: null
superseded_by: null
---

# E72: iter_block_pair — alternating block-level + per-pair SP refinement

## Hypothesis

E61 V2 block-level (quadrant) crossover and E69 per-pair SP-swap each
deliver marginal lifts (-0.07%, -0.10%) on different scales. Block
moves coarsely escape the basin; per-pair refines locally. **A SINGLE
APPLICATION of either is sub-threshold; iterative ALTERNATION may
compound them.**

The mechanism: block crossover lands in a "third basin" intermediate
between E25 and E41. Per-pair SP-swap (with E25 vs E41 disagreement
target) refines within this intermediate basin. After polish, repeat
with FRESH SP encoding of the new state — disagreements have shifted
from those in the original E25/E41 pair. Each iteration nudges the
state toward a deeper local minimum that neither E25 nor E41 alone
could reach.

## Method

Pseudo-code:

```
state = best(E25, E41)
best_seen = (state, proxy(state))
for iteration in range(K):
    # Block-level: spatial-block crossover with OTHER basin.
    other = e41 if state.is_close_to(e25) else e25
    crossover = spatial_block_crossover(state, other, seed=iteration)
    polished_block = cd_polish(crossover, budget=300s)

    # Per-pair: SP-guided directed swap.
    rotated = sp_guided_swap(polished_block, other, axis_pres_only=True,
                              max_attempts=300, budget=600s)
    polished_pair = cd_polish(rotated, budget=300s)

    if proxy(polished_pair) < best_seen.proxy:
        best_seen = (polished_pair, proxy(polished_pair))
    state = polished_pair  # use as next iteration's start

return best_seen.placement
```

K = 3-5 iterations target. Total wall ~ K × 20 min = 60-100 min/bench.

The "other" basin selection ensures we always crossover with the basin
we haven't yet absorbed. If `state` drifts toward E25, swap with E41,
and vice versa.

## Kill gate

  - --fast iter-best ≥ E48 fast 0.92024 — kill (no improvement compounds).
  - Per-iteration regression > 0.5% from best_seen — kill (instability).
  - --fast lift < 0.3% over E48 → marginal.
  - --fast lift ≥ 0.3% AND zero NG45 ariane133 regression → graduate-pending.

## Generalization check

Test on 4 fast benches. If aggregate lift ≥ 0.3%, queue --ng45.

## Outcome (decided 2026-05-04)

### Block-level crossover phase: FAILED on cached placements

Cached E25/E41 placements differ from E61 V2 historical fresh runs
by ~0.07-0.1% per-bench. This drift is enough to break crossover
feasibility/quality:
- ibm01: quadrant_pick seed=42 → 28 residual overlaps → unrecoverable.
- ibm12: seed=42 → 0 residual but polished result LOOKS good in
  evaluator (1.20272) but compute_proxy_cost says 1.20530 (above
  start 1.20534). Float drift kills lift.
- All 4 iters in original E72 run on ibm01/ibm12: 0 successful
  crossovers.

E71 V2 partition crossover (multi-trial NxN grid) with extended
legalize achieves 50% feasibility on ibm01 but 0/15 accepts (all
feasible crossovers have higher polished proxy than start). Same
finding: cached placements DO NOT reach the third basin via polish.

### Per-pair fallback (iter-E69 v2): MARGINAL but real

| Bench | Method | Best | Lift vs start |
|---|---|---|---|
| ibm01 | iter-E69 v2 (6 rounds, 250 attempts) | **0.88703** | **−0.290%** |
| ibm04 | single E69 extended (1329 attempts) | 0.99966 | −0.190% |
| ibm09 | single E69 extended (963 attempts) | 0.82762 | −0.080% |
| ibm12 | single E69 extended (601/5386 attempts) | 1.20458 | −0.063% |

Aggregate avg = 0.97972 vs E48 ref avg 0.981 = **−0.105% on --fast
4-bench subset (ibm12 substituted for ibm13).**

vs fresh best of {E25, E41} per bench (the honest measure):
aggregate lift = **−0.156%**.

Below 0.30% promotion threshold but real and reproducible.

### Bench-class behaviors

**Polish-helps benches (ibm01, ibm12)**: iter-E69 multi-round
compounds via polish-induced topology changes. Each round, the
polish phase modifies the state slightly, exposing new
SP-disagreements with the target basin. New disagreements →
new productive swaps. ibm01 reached -0.290% over 3 productive
rounds before plateau.

**Polish-hurts benches (ibm04, ibm09)**: polish phase regresses
(canonical compute_proxy_cost > pre-polish proxy due to evaluator
float drift). v2 fix uses best-seen state — but without polish-
induced topology shift, re-encoding gives same disagreement set
each round, so compounding doesn't kick in. Single E69 with full
disagreement-set coverage and spatial-close-first ordering beats
iter on these benches.

### Lessons

1. **Cached E25/E41 placements break crossover-based mechanisms.**
   The "third basin" found by E61 V2 historical is fragile to ~0.07%
   placement drift from fresh co-runs. Cluster transplants and
   partition crossovers don't replicate this from cached pairs.

2. **Per-pair iteration compounds via polish, not via re-encoding.**
   When polish improves the state, the re-encoded SP exposes new
   disagreements; without that, iteration is no better than single-pass.

3. **Best mechanism is bench-specific.** Polish-helps benches favor
   iter-E69; polish-hurts favor single E69 extended. Hybrid =
   per-bench best.

4. **The float drift in IncrementalProxyEvaluator vs compute_proxy_cost
   is structural, not a bug.** It can MASK lift (deep cross ibm12:
   evaluator says 1.20272 = -0.21% lift; canonical says 1.20530 =
   no lift). Any mechanism using evaluator-internal optimization
   needs to be checked against canonical proxy.

### Verdict

MARGINAL. ibm01 alone reaches -0.29% (very close to promotion).
Aggregate -0.156% on --fast 4-bench. The mechanism (iter-E69 v2 +
single E69 ext per-bench) is real but sub-promotion at scale.

For a true breakthrough (>0.5% lift): need fresh E25/E41 co-runs
with E61 V2 (~3-7 hr/bench) — the third basin requires this. OR
move to a different mechanism class: K=50 Hungarian (E67), BP / tensor
networks (E40), constrained NEB (E68), diffusion sampling (Lai 2024).

## Pointers

- Code:
  - `code/iter_block_pair_placer.py` (block + per-pair, falsified
    on cached placements).
  - `code/run_iter.py` (driver for full iter).
  - `code/run_iter_e69_only.py` (driver for iter-E69-only fallback;
    v2 fix uses best-seen as next round's start).
  - `code/deep_polish_crossover.py` (multi-seed crossover + deep
    polish + iter-E69; falsified on ibm12 from cached pair).
- Cached placements at
  `experiments/E69_sequence_pair_search/results/placements/`.
- Best per-bench polished placements saved to
  `experiments/E72_iter_block_pair/results/`.

## Pointers

- Code: `code/iter_block_pair_placer.py`, `code/run_iter.py`.
- Parents: E61 V2 (`experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`),
  E69 (`experiments/E69_sequence_pair_search/code/`).
- Cached placements at
  `experiments/E69_sequence_pair_search/results/placements/`.
