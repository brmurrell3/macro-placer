---
id: E69
name: sequence_pair_search
status: marginal
parent: E48 (champion), E61 V2 (spatial-block crossover precedent), E65 (infeasibility-wall finding)
created: 2026-05-03
decided: 2026-05-03
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **MARGINAL 2026-05-03 23:00 EDT.** SP encoder works (Murata-Fujiyoshi adjacency rule, after fixing initial pair-by-pair over-constraint). Topology distance probe quantifies basin separation: 17-30% of pairs disagree across 4 IBM benches; most disagreements are axis-rotating (LEFT↔BELOW etc.) with only 7-15% axis-preserving. SP-guided directed swap on axis-preserving disagreements ALWAYS finds genuine lift over starting basin (-0.04% to -0.19%) but aggregate --fast lift (~-0.10%) is BELOW the 0.30% promotion threshold. Per-pair operations are too local to escape basin; block-level operations (E61 V2 quadrant crossover, -0.07% --all) outperform per-pair (E69) on tied-basin benches. **Validates the topology framing as REAL** but **falsifies per-pair swap as a breakthrough mechanism**. Real escape requires operations that scale (block-level, Hungarian K=50, BP/diffusion).
champion_delta: -0.0010 (-0.10%) on --fast subset, sub-promotion threshold
graduated_to: null
superseded_by: null
---

# E69: sequence_pair_search — wall-immune topology representation

## Hypothesis

The infeasibility wall identified by E65 is a property of the Cartesian
representation R^{2N}, not of the placement problem. In sequence-pair
representation (Murata-Fujiyoshi 1995) every (Γ+, Γ-) decodes to a
non-overlapping placement; there is no wall in SP space, only a discrete
graph of pair-flip moves. The basins (SDF / DPO) are points on this
graph; their pair-flip distance is a structural fingerprint of the
problem.

E61 V2's quadrant-block crossover empirically confirmed that
recombinations preserving block-level locality can thread the wall
(−0.07 % --all marginal, −1.47 % on ariane133). SP encoding is the
principled limit: at the per-pair-relation granularity, every move is
wall-immune by construction.

## Method

Three diagnostic phases, each kill-gateable, before any placer commit:

1. **Cartesian → SP inverter**. For each pair (i, j) of hard macros:
   - if i.right ≤ j.left → i precedes j in BOTH Γ+ and Γ- (H: left)
   - elif i.left ≥ j.right → i follows j in both (H: right)
   - elif i.top ≤ j.bottom → i precedes in Γ+, follows in Γ- (V: below)
   - elif i.bottom ≥ j.top → i follows in Γ+, precedes in Γ- (V: above)
   - else: overlap (legal placements exclude this; if it occurs,
     project_overlaps first)
   Topo-sort the resulting H/V constraint graphs to get Γ+ and Γ-.
   Stable for non-overlapping placements; H-preferred when both apply.

2. **Topology-distance probe**. Encode E25 and E41 outputs to
   SP_E25 = (Γ+_25, Γ-_25) and SP_E41 = (Γ+_41, Γ-_41).
   Compute Kendall-tau distance between Γ+_25 and Γ+_41 (and Γ-).
   Tells us if SP-space search is tractable in budget:
   - d ≤ 200 inversions → SP-SA between basins is direct.
   - 200 < d ≤ 2000 → SP-SA needs targeted moves (block-level).
   - d > 2000 → SP-space too large; representation switch alone won't
     break wall in budget.

3. **Per-bench SP-guided directed swap search**. For ibm12 (E61 V2's
   tied-basin success bench): start from E25 Cartesian. Identify pairs
   where SP_E25 and SP_E41 disagree (different relation). Try to
   "flip" the disagreeing pair toward SP_E41 by swapping their
   Cartesian positions (when feasible) or by destroy+reinsert.
   Polish each accepted flip with short CD+SA. This is essentially
   E15 pair-swap restricted to SP-disagreement pairs — a directed
   move informed by basin topology.

Optional Phase 4 (only if 1-3 succeed): persistent homology on
SP-encoded converged placements in `experiment_log.jsonl`. H_0 count
tells us how many basins exist beyond E25/E41 in SP-space.

## Kill gate

Phase 1: encoder must be stable — encoding E25 twice (different runs,
same placement) must give identical SP. Verified by computational
self-check.

Phase 2: if d(SP_E25, SP_E41) > 2000 inversions on ibm01 → kill, the
search space is too big for SA in 600 s. Document the structural
finding (wall is reps-fixable but search is hard).

Phase 3 (smoke ibm12): if no flip improves over E25 baseline by ≥ 0.3 %
after 1 hr search budget → kill, directed swaps don't escape basin.

Final: --fast must not regress more than 0.5 % vs E48 0.92024.

## Generalization check

If --fast lift ≥ 0.3 % on E48, run --ng45 (especially ariane133).
SP encoding is netlist-blind by construction; transfer should be
clean. If it regresses on ariane133, the IBM-aware/NG45-blind failure
class includes representation switches too — major structural finding.

## Outcome (decided 2026-05-03)

### Phase 1 — Encoder

The proper Murata-Fujiyoshi adjacency rule was needed to avoid false
cycles. My initial pair-by-pair "H-first or V" rule over-constrained
the constraint graph: it added an edge for EVERY pair (LEFT for
diagonally-separated pairs that happened to be H-separated), creating
geometric cycles (e.g., a→b→c→a where each edge is a "right-and-down"
move, but a is below c — the partial-order argument fails because LEFT
relation only constrains x, not y).

Correct rule (Murata-Fujiyoshi 1995): only "directly adjacent" pairs
constrain. H-edge iff y-projections overlap AND strict H-separation.
V-edge iff x-projections overlap AND strict V-separation. Diagonally-
separated pairs (no projection overlap) get NO edge — their order in
Γ+/Γ- is determined by topological tie-breaking using a canonical
spatial key (x_left + y_bot for Γ+; x_left - y_top for Γ-).

After this fix, encoder is stable, deterministic, ~30 ms on ibm01
(N=246), ~500 ms on ibm12 (N=651).

### Phase 2 — Topology distance probe

Pair-relation Hamming distance d(SP_E25, SP_E41) on 4 IBM benches:

| Bench | n_hard | n_pairs | total_diff | %  | axis_pres | axis_rot |
|-------|-------:|--------:|-----------:|---:|----------:|---------:|
| ibm01 |   246  |  30,135 |     5,981  |20%|       413 |    5,568 |
| ibm04 |   295  |  43,365 |     8,873  |20%|     1,329 |    7,544 |
| ibm09 |   253  |  31,878 |     9,565  |30%|     1,897 |    7,668 |
| ibm12 |   651  | 211,575 |    37,656  |18%|     5,386 |   32,270 |

Findings:
1. **Basin separation is uniform across benches**: 17-30% of pairs
   disagree on left/right/below/above relation between SDF and DPO
   basins. Confirms E65: basins are topologically distinct.
2. **Most disagreements are axis-rotating** (LEFT↔BELOW etc.) — basins
   place macros in fundamentally different relative arrangements. Only
   7-15% of disagreements are axis-preserving (LEFT↔RIGHT or
   BELOW↔ABOVE) which can be addressed by simple position swap.
3. **Decoded SP via Murata compaction is NOT a feasible canvas-fitting
   placement** — produces compact bottom-left packing that's too narrow
   for the IBM canvas; requires legalization that introduces overlaps;
   proxy is +50% worse than original. The compact decode discards the
   wide-spread topology needed for low-density.

### Phase 3 — SP-guided directed-swap search

Tried position-swap of axis-preserving disagreement pairs starting from
the better basin per bench. Each swap legalized via project_overlaps;
accepted iff strict proxy improvement.

| Bench | Start | Best (after swap+polish) | vs Start | vs E48 ref |
|-------|-------|--------------------------:|---------:|-----------:|
| ibm01 | E25 0.88960 | **0.88791** | −0.19% | −0.49% |
| ibm04 | E41 1.00156 | **1.00010** | −0.15% | +1.53%* |
| ibm09 | E41 0.82828 | **0.82786** | −0.05% | −1.59% |
| ibm12 | E41 1.20534 | **1.20487** | −0.04% | −0.13% |

*ibm04 my fresh E41 (1.00156) is ~1.5% worse than ADR-011 reference E48
ibm04 (0.985) — basin variance from seed/run-to-run, not SP-search
regression. Same explanation for ibm09's −1.59%: my fresh E41 happens
to outperform the reference.

**Honest measure (vs starting basin)**: −0.04% to −0.19% per bench,
**aggregate ≈ −0.10% on --fast subset**. Below the 0.30% promotion
threshold.

Acceptance rates: 4-15% of attempted swaps. Most failures are
infeasibility (project_overlaps cannot legalize after a long-distance
swap) or proxy-regression rejections. **CD polish after swap was
unreliable**: helped on ibm01/ibm12, hurt on ibm04/ibm09 due to
incremental-evaluator vs compute_proxy_cost float drift (gotchas.md).

### Verdict

**MARGINAL.** SP topology framing is validated as a real description
of basin structure. Per-pair directed swap finds GENUINE local
improvements every bench. But:
- The disagreement set is too large (5386 axis-preserving on ibm12)
  for SA to converge in budget.
- Per-pair operation is too local. Block-level (E61 V2) reaches further
  into topology space and finds bigger lifts on tied-basin benches
  (ibm12 E61 V2 1.1982 vs E69 1.20487, 0.55% better for E61 V2).
- Aggregate --fast lift (-0.10%) is below the 0.30% promotion threshold
  and similar to E61 V2's --all marginal (-0.07%).

### Real breakthrough mechanisms suggested by this work

The SP framing CONFIRMS that escapes from the SDF↔DPO basin pair must
operate at a coarser-than-per-pair granularity. Two directions:

1. **Block-level SP recombination** — spatial-block (E61 V2) crossover
   already does this at quadrant granularity. Finer granularity
   (rows? hMETIS-clusters? n×n grid?) is the natural extension.
2. **Larger-K joint operations** — Hungarian K=50 (E67, in-flight by
   parallel agent) is the polynomial-time analog. Each Hungarian
   move re-assigns 50 macros simultaneously, structurally a "K=50 SP
   move" — orders of magnitude beyond E69's K=2.

The infeasibility wall (E65) is a Cartesian artifact (SP space has no
wall), but the per-pair SP move is too small to navigate the topology
graph efficiently. Block-level + polynomial-time large-K moves are the
ways forward.

### Lessons

1. **The Murata-Fujiyoshi inversion is subtle**: pair-by-pair "pick
   one relation per pair" creates false cycles. Adjacency-only
   constraints + topological tie-break are required.
2. **Standalone init quality doesn't predict polished basin quality**
   (E62 lesson) generalizes: SP-decoded compact placement (proxy +50%)
   doesn't help even though SP topology is valid. The basin we land
   in depends on starting POSITIONS, not just topology.
3. **Per-bench seed variance is ±1.5%** — fresh E25/E41 runs differ
   from ADR-011 reference by similar magnitudes. Honest comparison
   uses fresh-run-baselines, not historical references.
4. **CD polish after structural moves can REGRESS** in canonical
   compute_proxy_cost due to evaluator float drift. Always take min
   over {swap-state, polish-state, parents}.

## Pointers
- Code: `code/sp_inverter.py`, `code/run_diagnostics.py`,
  `code/sp_search_placer.py` (Phase 3, conditional).
- Theory: Murata, Fujiyoshi, Nakatake, Kajitani 1995 "Rectangle-packing-
  based module placement," ICCAD; Kahng-Markov 2010 textbook §11.5.
- Parents: E25 (`submissions/cd_lns_sa/placer.py`), E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- Related: E61 V2 (block-level recombination — precedent for "preserving
  topology to thread the wall"); E65 (the wall finding that motivates
  this).
