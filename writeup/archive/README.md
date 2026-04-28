# Archive

Material kept for the writeup but not part of the live submission.
Everything here is referenced by name in `docs/experiment_index.md` as
falsified, superseded, or stale planning.

Nothing here is imported by live code. The champion
(`submissions/cd/cd_adaptive_placer.py`) and the live ablation
`submissions/cd/cd_only_placer.py` continue to depend only on
`scripts/cd_ibm10_diagnostic.py`, `submissions/cd/sdf_init.py`,
and the `macro_place/` package.

## Contents

### `planning/`

Stale writeup-planning docs that pre-date the CD pivot and now
contradict the current narrative.

- `writeup_plan.md` — original plan; presents DPO at 1.4246 as the
  resolution. The current narrative (CDAdaptive at 1.1055) is in
  `writeup/outline.md`, `writeup/contributions.md`, and
  `writeup/framing.md`.
- `dpo.md`, `dpo_next.md` — DPO architecture notes and next-steps;
  superseded by the CD line.
- `next_experiments.md` — DPO-era experiment prioritization; self-flagged
  "SUPERSEDED" at the top of the file.
- `eval_pipeline_design.md` — earlier surrogate-evaluation pipeline
  design; superseded by the E1 incremental evaluator.
- `overnight_driver.md` — autonomous experiment-driver prompt for
  /loop runs; not paper material.

### `submissions/`

Superseded placer code. Kept as evidence for the writeup's
falsification narrative.

- `dpo/` — full DPO ablation set (best avg 1.3834). Superseded by CD on
  the incremental evaluator. Multi-seed verification (E5) showed basin
  lock; congestion-only refinement (E10) and diverse priors (E11) both
  flat on `--all`. See `docs/experiment_index.md` rows for E5/E10/E11.
- `cd_lns_placer.py`, `lns.py` — E3 single-macro LNS. Falsified
  2026-04-27: full-canvas and 5×5 local-window variants both flat on
  ibm17 vs CDOnly. See `docs/experiment_index.md` row E3.

### `runs/`

Per-run JSON outputs from the evaluation harness (51 files). The
aggregated single source of truth is `results/experiment_log.jsonl`;
these per-run files are kept for the rare case the writeup needs the
exact runtime / per-benchmark breakdown of a specific run.

### `run_logs/`

Raw stdout from E3 LNS and E9 Adaptive `--all` / `--fast` runs
(5 `*.log` files). Contains per-sweep deltas and plateau-exit decisions
for the leaderboard-beating E9 run.

### `scripts/`

Untracked-on-main probe scripts that informed the LNS/init-diversity
experiments but never produced productionizable code.

- `lns_escape_probe.py` — multi-seed test of whether CD plateaus are
  escapable via a random destroy set.
- `multi_init_cd_probe.py` — single-run worker for SDF-seed basin-
  diversity probing.

### Pre-existing items (left in place)

`ablation_*.txt`, `cd_ibm10_diagnostic.json`, `miftari_generalize.txt`,
`overnight_run.log`, `polyhedra_traversal.txt`, `profiling/` —
materials archived during earlier cleanups.
