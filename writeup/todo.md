# Writeup TODO

The DPO-era pre-writeup TODO is retired — every item from that list
(ablations, multi-seed, RUDY analysis, polyhedra traversal, novelty
literature check, version cleanup) is captured in `experiment_notes.md`.
What follows is the remaining work to ship the writeup.

---

## Prose drafts (one file per outline section)

Outline source of truth: `outline.md`. Each section below should become
its own draft file.

- [ ] `sec1_introduction.md` — problem, RePlAce baseline, our 1.1055
  result + two-pivot trajectory
- [ ] `sec2_polyhedra.md` — decomposition theorem, LP within each
  polyhedron, dual sensitivity. Source: `theory.md` §1-2, `problem.md`
- [ ] `sec3_navigation.md` — SDF init, assignment extraction, HiGHS LP
  + dual extraction, GridSurrogate, ClusterScreener, navigation loop.
  Source: `contributions.md` §1, `eval_pipeline_design.md`
- [ ] `sec4_barrier.md` — overnight 22-experiment sweep, Miftari
  rho=-0.001, swap+LP. Source: `historical_results.md`,
  `lp_hpwl_diagnostic.md` (in `docs/`)
- [ ] `sec5_dpo.md` — DPO architecture, smooth components, penalty
  continuation. Source: `dpo.md` §3-5
- [ ] `sec6_dpo_theory.md` — penalty as barrier crossing, complexification,
  low-seed-variance evidence. Source: `dpo.md` §6, `theory.md` §2a,
  `contributions.md` §4 + §6
- [ ] `sec7_rudy_limit.md` — congestion weight sweep killed, cell-by-cell
  RUDY/real, 10.9% hotspot overlap. Source: `experiment_notes.md` §10-11,
  `rudy_analysis.py`
- [ ] `sec8_cd.md` — incremental evaluator (E1), breakpoint enumeration,
  CDOnly→CDAdaptive (E9). Source: `cd_ibm10_results.md`,
  `closing_the_gap.md`, `experiment_notes.md` §15-17,
  `contributions.md` §8-9 + §12
- [ ] `sec9_results.md` — full per-bench tables for CDAdaptive,
  champion lineage, dead-ends summary. Source: `historical_results.md`,
  `docs/results.md`, `docs/experiment_index.md`
- [ ] `sec10_discussion.md` — non-decomposability, "bypass don't fix",
  infrastructure unlocks algorithms (twice — E1 and E9). Source:
  `contributions.md` §5 + §10 + §11
- [ ] `references.md` — ~25-30 cites, see `outline.md` references list

---

## Data captures (re-runnable)

- [ ] `data/rudy_ibm01.txt` — capture stdout from `rudy_analysis.py`
- [ ] `data/e9_per_bench.json` — extract from `results/experiment_log.jsonl`
- [ ] `data/dpo_ablation.json` — extract from `experiment_log.jsonl`

---

## Figures

All listed in `outline.md`. Generate from data once captured.

- [ ] LP-HPWL vs proxy scatter (rho=-0.001) — outline §4
- [ ] Congestion vs proxy scatter (rho=0.825) — outline §4
- [ ] Overnight sweep bar chart (22 experiments, all flat) — outline §4
- [ ] RUDY vs real congestion heatmap (ibm01) — outline §7
- [ ] Per-cell ratio distribution (RUDY/real) — outline §7
- [ ] Champion lineage bar chart (1.50→1.49→1.38→1.12→1.10) — outline §9
- [ ] CD convergence curve (sweep deltas vs wall time) — outline §9
- [ ] CDAdaptive per-benchmark wall time chart — outline §9

---

## Final pass

- [ ] Length/density check vs 12-14 page target
- [ ] Math notation consistency
- [ ] References ordered + DOIs
- [ ] Title + abstract draft
- [ ] Final read-through with someone unfamiliar with the project
