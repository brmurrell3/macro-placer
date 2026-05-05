---
id: E73
name: axis_rotation_transplant
status: falsified
parent: E69 (SP search marginal), E48 (champion)
created: 2026-05-04
decided: 2026-05-04
champion_at_time: 1.08151 (E48 hybrid, ADR-011)
outcome: **FALSIFIED 2026-05-04 06:55 EDT on ibm01 smoke (200 attempts).** 0 accepts; 183 feasibility failures (91.5%); 17 proxy regressions. Per-pair transplant (move 2 macros to other basin's positions) is too disruptive — the target positions are occupied by OTHER macros in the current basin, and project_overlaps' 50-iter cap can't resolve the cascading displacement. Even when legalized (~9%), the disrupted local layout has worse proxy. **Right transplant unit is cluster-level (5-30 macros), not pair (E71).** The 85-93% axis-rotating disagreement majority remains a real address space, but per-pair operations can't reach it.
champion_delta: 0 (no improvement on smoke)
graduated_to: null
superseded_by: E71 (cluster-level transplant — same hypothesis at right granularity)
---

# E73: axis_rotation_transplant — coordinated 2-macro repositioning

(Renumbered from E70 — E70 was claimed by parallel agent's
`E70_kjoint_hungarian_integrated`, created ~30 min earlier 2026-05-04.)

## Hypothesis

E69 revealed that 85-93% of basin disagreements are AXIS-ROTATING:
E25 places (i, j) horizontally-adjacent and E41 places them
vertically-adjacent (or vice versa). E69's per-pair POSITION SWAP
cannot change adjacency type, so it left this 5–10× larger
disagreement mass unattacked.

A SWAP exchanges (i.x, i.y) ↔ (j.x, j.y). Geometric relation between
i and j is unchanged. To rotate axis, we need a coordinated
2-macro REPOSITIONING — move both macros to NEW coordinates.

The cleanest target is the OTHER basin's positions for (i, j). E25
and E41 placements are both feasible and globally low-proxy. If
E41's position for (i, j) achieves the rotated relation we want,
"transplanting" them there should produce a placement that's still
mostly E25 but locally matches E41's topology for that pair.

## Method

For each axis-rotating disagreement pair (i, j) — i.e. pair where
relation in SP_25 ∈ {LEFT, RIGHT} and relation in SP_41 ∈ {BELOW, ABOVE}
or vice versa:

  1. candidate = current.clone()
  2. candidate[i] = e41_placement[i]   (transplant from other basin)
  3. candidate[j] = e41_placement[j]
  4. candidate, n_iters = project_overlaps(candidate, benchmark)
     (resolve overlaps with neighbors via push-apart)
  5. if zero overlap AND proxy < current proxy: accept (mutate current)
  6. else: revert

Process pairs in spatial-close-first order (small displacement →
fewer overlaps to repair → higher feasibility rate).

After all transplant attempts, optional short CD polish.

This is a fundamentally different move type than E69's swap:
- swap preserves total area used in (i, j) pair; transplant relocates them.
- swap can only rotate within {LEFT↔RIGHT, BELOW↔ABOVE}; transplant
  can achieve full {LEFT↔BELOW↔RIGHT↔ABOVE} rotations.
- swap has lower disruption to neighbors; transplant disrupts more
  but reaches deeper into topology space.

## Kill gate

  - Zero accepted transplants on ibm01 → mechanism doesn't exist; kill.
  - --fast > E48 fast 0.92024 + 0.5% → kill (regression).
  - --fast lift < 0.3% over E48 → marginal (matches E69).
  - --fast lift ≥ 0.3% AND zero NG45 ariane133 regression → graduate-pending.

## Generalization check

Test on 4 fast benches (ibm01, ibm04, ibm09, ibm12). If aggregate lift
≥ 0.3%, queue --ng45 ariane133.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/axis_rotation_placer.py`, `code/run_axis_rotation.py`.
- Parents: E69 SP-search (`experiments/E69_sequence_pair_search/`).
- Cached placements at
  `experiments/E69_sequence_pair_search/results/placements/`.
- Discussion: extends E69's topology-distance probe by addressing the
  85-93% axis-rotating disagreement mass that per-pair swap missed.
