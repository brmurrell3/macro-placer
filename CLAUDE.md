# CLAUDE.md — Thinkorplace

## Current state

Post-competition. The Partcl/HRT Macro Placement Challenge 2026 is closed; the
organizers' final leaderboard ranks team "thinkorplace" ninth (verified mean
proxy 0.9974). Technical report v1.0 (30 September 2026, source in
`writeup/latex/`) is the current account of the work. The May 2026 agent
instructions are archived in `docs/competition/CLAUDE.competition-era.md` and
no longer apply. Do not run solver experiments unless asked.

## Repository map

- `thinkorplace/` solver package (corrected v1 and v2 runtime)
- `submissions/` competition entry points and archived placers; root `placer.py` launches v2
- `macro_place/` challenge infrastructure: loader, proxy cost, `evaluate`, incremental evaluator
- `writeup/` report source (`latex/`), frozen evidence (`data/publication/`), study configs (`study/`), run records (`runs/`), explorer (`interactive/`), May data log and archived code (`archive/`)
- `experiments/` per-experiment manifests, including falsified ones; index in `docs/experiment_index.md`
- `docs/` May 2026 development records; `docs/competition/` holds competition-era files
- `scripts/` study runner, publication renderers and checks; `test/` focused tests

## Commands

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py -b ibm01   # or --all, --ng45
.venv/bin/python scripts/run_publication_study.py --config writeup/study/<config>.json --output writeup/runs/<id>
make -C writeup/latex                                             # PDF in writeup/latex/build/main.pdf
.venv/bin/python scripts/check_publication_paper.py               # release check; --source-only while editing
.venv/bin/python -m pytest -q test/test_thinkorplace_package.py test/test_publication_runner.py \
  test/test_submission_safety.py test/test_postpilot_repairs.py test/test_sdf_topology.py
```

## Evidence rules

Full rules: `writeup/latex/EDITORIAL_STANDARDS.md`.

- Label each result once: Official (organizers' final leaderboard), Historical (May 2026 records; unmatched hardware, budget and concurrency, mostly no saved coordinates), Measurement (post-competition diagnostics), Reproduction (post-competition run of the ranked configuration). Do not mix classes in one comparison.
- Every number traces to a frozen record in `writeup/data/publication/` or `writeup/runs/`. Regenerate `generated/` tables with their render scripts; do not hand-edit them.
- Keep failed, timed-out, partial and checkpoint-recovered runs visible. Never substitute a later repair for a primary observation or average over incomplete coverage.
- 0.9974 is attributed, by inference (judged commit not recorded), to the 20 May single pipeline; by the same inference the three-lane system was never scored, so never attribute 0.9974 to it.
- The Reproduction label has no reported result yet: the first run of the ranked configuration (`writeup/runs/ranked-reproduction-20260930/`) was stopped by a checkpoint defect. Do not present its partial designs as a result.
- Proxy scores say nothing about routed timing, power or area.
- Tables: caption above the tabular, `\footnotesize` minimum, no `\resizebox`, at most six decimals. No TODO or TBD in release sources.
- Never write: 0.98387, 0.97745, 0.99115 or 0.991 as results; "ceiling"; "true local minimum"; "saddle" as established fact (say "negative-curvature direction of a smooth surrogate"); "peer-reviewed", "refereed" or "external referee"; percentile claims; interim ranks; "I built"; sponsor logos; the author's email address.

## Do not touch

- Do not edit the runtime closure without a new runtime freeze (`scripts/freeze_publication_runtime.py`). It covers `thinkorplace/`, `submissions/`, `macro_place/`, `external/`, `benchmarks/`, `test/`, `pyproject.toml`, `uv.lock`, `requirements.txt`, `scripts/run_publication_study.py`, `scripts/publication_worker.py`, and every file listed in `writeup/data/publication/corrected_runtime_snapshot.json`.
- `docs/reviews/` is local and untracked; do not commit it or link to it from tracked files. The same holds for `Claude outputs/` and `writeup/scratch/`.
