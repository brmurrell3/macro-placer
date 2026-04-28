# `writeup/` directory map

Five markdown files. Each has a single, distinct role.

| File | Role |
|------|------|
| **`paper.md`** | **Canonical draft.** Section scaffolding + locked-in numbers + inline `TODO(...)` markers. Source-of-truth for the paper. |
| **`evidence.md`** | **Experimental archive.** Per-bench tables, ablations, dead ends, diagnostic experiments. Every datapoint cited by `paper.md` lives here. |
| `contributions.md` | Claim / novelty-boundary / evidence registry. Drafting aid — the "have I defended every claim?" checklist. Fold into `paper.md` before final submission. |
| `theory.md` | Supplementary mathematical material (polyhedral decomposition, complexification, etc.). Includes the explicit list of theory connections *excluded* from the paper body. |
| `README.md` | This file. |

Plus subdirectories:

| Path | Role |
|------|------|
| `data/` | Frozen artifacts referenced by `paper.md` figures (e.g. `rudy_ibm01.txt`). Currently empty. |
| `scripts/` | Figure-generation and analysis scripts (`rudy_analysis.py`, `polyhedra_traversal.py`, `profile_placer.py`). |
| `archive/` | Historic logs, removed code (polyhedra modules, DPO submissions), and prior planning docs. Reference only — do not edit. |

## Where to make changes

- **New numbers / experiments / results** → `evidence.md`.
- **New prose** → `paper.md`.
- **Figure data** → `data/` as a frozen artifact, then cite from `paper.md`.

## Source-of-truth rules

- Numbers in `paper.md` and `evidence.md` must trace to one of:
  - `results/experiment_log.jsonl` (the append-only run log), or
  - `results/<placer>_<timestamp>.json` (a frozen run JSON), or
  - a captured artifact in `writeup/data/`.
- The current champion is `submissions/cd_lns_gridbin/placer.py` at
  **1.0990** avg `--all`
  (`results/CDLNSGridBinPlacer_20260428_155739.json`).
  Prior champion: `submissions/cd_adaptive/placer.py` at 1.1055
  (`results/CDAdaptivePlacer_20260427_132214.json`), superseded
  2026-04-28 (ADR-007).
- If a number cannot be traced, mark it with `TODO(verify)` in `paper.md`.

## Single TODO list

The "Master TODO list" at the bottom of `paper.md` is the single rolled-up
list of every missing data / figure / prose item. Inline `TODO(...)`
markers in each section of `paper.md` are the authoritative source for
that list.
