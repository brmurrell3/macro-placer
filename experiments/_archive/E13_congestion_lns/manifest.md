---
id: E13
name: congestion_lns
status: marginal
parent: E12
created: 2026-04-28
decided: 2026-04-29
champion_at_time: 1.0990
outcome: 0.9384 (--fast); -0.45% lift over E16 baseline 0.9426 — below the 0.5% generalization threshold; mechanism works on ibm01 but doesn't generalize
champion_delta: null
graduated_to: null
superseded_by: null
---

# E13: congestion_lns

## Hypothesis
E12's grid-bin LNS overlay reaches avg 1.0990 on `--all` by destroying the
K worst-cost macros (`_cost_aware_destroy` move-to-center proxy delta) and
reinserting each at its proxy-minimizing legal grid-bin center. The destroy
step is **cost-ranked but spatially-blind**: the K picked macros may sit on
opposite sides of the canvas. Per E8's diagnostic, congestion is 74 % of the
proxy improvement headroom, and congestion bottlenecks are spatially clustered
(a few grid cells dominate the abu-5% sum). If we destroy K macros that
**share a hot-congestion region**, reinsertion releases the joint constraint —
a coordinated move that destroying isolated cost-ranked individuals can't.
This targets the spatial locality CD's per-axis fixed point and E12's
destroy-by-cost both miss.

## Method
Replace E12's `_cost_aware_destroy` with `_congestion_region_destroy`:

1. Read the current per-cell congestion field from the incremental
   evaluator — same field abu-summed in `_congestion_cost`:
   `V_total + H_total = smooth(V_net/grid_v_routes) + V_macro/grid_v_routes
   + smooth(H_net/grid_h_routes) + H_macro/grid_h_routes`. Reshape to
   `[grid_row, grid_col]`.
2. Threshold = `median(field) + 1*std(field)`. Cells above are "hot."
3. For each hard movable macro, find its containing grid cell from
   `(x/grid_w, y/grid_h)`. Score by descending containing-cell congestion
   (ties: macro index for determinism). Pick top K.
4. **Fallback** (flat-congestion benchmarks): if fewer than K macros sit in
   hot cells, the descending sort naturally pads with the next-most-congested
   cells until K is reached — no candidate-list refill needed.

Reinsert step is **identical** to E12 (`_gridbin_reinsert`): for each
destroyed macro, search every legal grid-bin center, commit the
proxy-minimizing one.

K formula: `max(1, min(destroy_cap=30, int(0.05 * |H|)))` — same as E12.
The "median + 1σ" threshold is global (applied to each benchmark's own
distribution); not per-benchmark tuning. CD phase, legality checks
(`_is_legal_2d`), fixed-macro preservation, and overlap validation are
unchanged from E12. Hyperparameter defaults: `cd_hard_cap_s=3000`,
`lns_budget_s=600`, `lns_destroy_frac=0.05`, `lns_destroy_cap=30`.

`cost_aware` and `random` destroy strategies kept as ablations (selectable
via `lns_destroy_strategy`).

## Kill gate
Per roadmap E13 (adapted to current baselines):
- If `--fast` avg ≥ E16 baseline 0.9425 (i.e., congestion-region destroy
  produces no clear improvement over the CD plateau), kill.
- If 0/3 LNS samples on the median benchmark improve baseline by ≥ 0.5 %,
  kill (signals the destroy step finds no joint-constraint releases).

## Generalization check
If `--fast` improves over baseline by ≥ 0.5 %, validate on NG45 ariane133
before queueing `--all`. Hot-congestion destroy is the kind of strategy
that could over-fit to IBM's net-density profile, so NG45 sanity is
required before champion-tier validation.

## Outcome (filled when decided)
**Marginal 2026-04-29 — falsified in spirit.** E13 passes the first
literal kill gate (avg `--fast` 0.9384 < 0.9425) but fails the
generalization-check threshold its own manifest sets for queueing `--all`
(lift 0.45 % < 0.5 %), and fails the second kill gate on three of four
fast benchmarks.

`--fast` numbers (zero overlaps everywhere):

| Benchmark | E13 LNS | CD plateau | E13 LNS lift | E16 baseline | E12-random | E13 vs E12-random |
|---|---:|---:|---:|---:|---:|---:|
| ibm01 | 0.9049 | 0.9129 | **0.94 %** | 0.9135 | 0.9073 | −0.27 % |
| ibm04 | 1.0150 | 1.0147 | 0.08 % | 1.0179 | 1.0150 | tied |
| ibm09 | 0.8573 | 0.8568 | 0.40 % | 0.8605 | 0.8541 | +0.37 % |
| ibm13 | 0.9763 | 0.9714 | 0.15 % | 0.9785 | 0.9724 | +0.40 % |
| **AVG** | **0.9384** | — | — | 0.9426 | 0.9372 | **+0.13 %** |

**The mechanism works on ibm01 but doesn't generalize.** ibm01's LNS
phase produced a 0.94 % lift over CD plateau (sample 1 alone delivered
−0.76 % — a real joint-constraint release). The other three benchmarks
showed LNS lifts of 0.08 %–0.40 %, which is the same scale as E12's
random-destroy ablation produces — i.e., congestion-region destroy
provides no additional joint-constraint discovery beyond random destroy
on these three.

**Vs E12 random-destroy ablation (the closer comparison than E16
baseline).** E13 is **+0.13 % WORSE on average** than E12 random destroy
on `--fast` — within noise but not a directional improvement. Per-bench:
E13 wins ibm01 by 0.27 %, ties ibm04, loses ibm09 by 0.37 % and ibm13 by
0.40 %. Mixed wins/losses, consistent with the destroy strategies being
interchangeable on most fast benchmarks.

**The kill gates and the manifest's own gen-check.**
- First kill gate (avg `--fast` ≥ 0.9425): NOT hit — passed by 0.9384.
- Second kill gate (0/3 LNS samples on median bench improve ≥ 0.5 %):
  hit on ibm04 (largest single-sample lift 0.06 %), ibm09 (0.36 %), and
  ibm13 (0.12 %); ibm01 alone meets it (sample 1 at 0.76 %). Three of
  four benchmarks hit the second kill gate.
- Generalization check (manifest's own clause): "If `--fast` improves
  over baseline by ≥ 0.5 %, validate on NG45 ariane133 before queueing
  `--all`." Lift is 0.45 % — below threshold. **Do not queue `--all`
  without NG45 validation first**, and the budget squeeze for this
  overnight push doesn't accommodate the NG45+`--all` pair.

**Decision: marginal, do not queue `--all`.** Surface to human for
verdict on whether to (a) accept marginal and move on, (b) queue NG45
sanity then `--all` despite the gen-check threshold, or (c) refine
destroy heuristic (e.g., destroy nearest-neighbors WITHIN hot cells, not
just the residents of hot cells).

**Source JSON:** `results/CDLNSCongestionPlacer_20260429_011439.json`.
Run log: `experiments/E13_congestion_lns/run.log`. Logged with
hypothesis tag `e13_congestion_lns_fast` in
`results/experiment_log.jsonl`.

## Pointers
- Code: `code/cd_lns_congestion.py`.
- Parent: E12 grid-bin LNS (`submissions/cd_lns_gridbin/placer.py`,
  champion at avg 1.0990).
- Diagnostic justification: `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md`
  (E8 — congestion is 74 % of proxy headroom).
- Discussion: `docs/roadmap.md` E13 entry.
