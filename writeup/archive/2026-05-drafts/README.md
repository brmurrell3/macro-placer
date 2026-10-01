> **Superseded May 2026 draft.** Numbers and framings here (for example 0.98387 'shipped', 'ceiling') are corrected in technical report v1.0.

# `writeup/` directory map (May 2026)

This was the map of `writeup/` while these drafts were active. The drafts now live in this folder; the current map is [`../../README.md`](../../README.md).

Six markdown files. Each has a single, distinct role.

| File | Role | Status |
|------|------|--------|
| **`STORY.md`** | **Narrative arc.** Three-act + epilogue + Act-5 structure, section mapping to standard CAD-paper format, priority-ordered TODO list for publication readiness. Start here when planning what to write next. | current 2026-05-22 |
| **`paper.md`** | **Canonical draft.** Section scaffolding + locked-in numbers + inline `TODO(...)` markers. §§8.9–8.11 (Hessian saddle escape, central innovation), §8.12 (DREAMPlace lane), and §8.13 (shipped V4+Gaussian + 3-lane ensemble) have publication-ready prose; earlier sections still have `TODO(prose)` placeholders for the prose pass. | partly drafted; §8.13 added 2026-05-22 |
| **`evidence.md`** | **Experimental archive.** Per-bench tables, ablations, dead ends, diagnostic experiments. Every datapoint cited by `paper.md` lives here. | needs E74/E84/A4 + E117/v2 per-bench tables |
| `contributions.md` | Claim / novelty / evidence registry, numbered 1–23. Drafting aid — the "have I defended every claim?" checklist. Will fold into `paper.md` before final submission. | §§21–23 added for E74/E84/A1; v2 claims TODO |
| `theory.md` | Supplementary literature material (polyhedral decomposition, mountain-pass / NEB theory, complexification, etc.) + the explicit list of literature connections *excluded* from the paper body. | sufficient for §10 Related Work source material |
| `README.md` | This file. | current 2026-05-22 |

Plus subdirectories:

| Path | Role |
|------|------|
| `data/` | Frozen artifacts referenced by `paper.md` figures. Currently sparse. |
| `scripts/` | Figure-generation and analysis scripts. |
| `archive/` | Historic logs, removed code, prior planning docs. Reference only. |

## Where to make changes

- **Re-plan the narrative** → `STORY.md`.
- **New prose for an existing section** → `paper.md`.
- **A new numbered contribution** → add to `contributions.md`, cross-reference from `paper.md`.
- **New numbers / experiments** → `evidence.md` (with provenance to `results/experiment_log.jsonl` or a frozen run JSON).
- **Figure data** → `data/` as a frozen artifact, then cite from `paper.md`.

## Current champion (the number the paper has to defend)

**`submissions/thinkorplace-v2/placer.py`** — V4+Gaussian descent +
3-lane parallel polish ensemble (the shipped submission, 2026-05-21).

| Metric | Value | Source |
|--------|------:|--------|
| IBM avg `--all` (EPYC 9655P, 2-way `--jobs`, `budget_seconds 1500`) | **0.98387** | EPYC 2-way extended-CD sweep, 2026-05-21 |
| IBM avg `--all` (M3, 3-lane default budget) | 0.99115 | local verification 2026-05-21 |
| V4+Gaussian standalone (M3, no ensemble) | 0.98199 | E117 run 2026-05-20 |
| vs RePlAce 1.4578 | **−32.5 %** | published baseline |
| vs leaderboard Carrotato 0.967 (#1) | +1.74 % gap | verified leaderboard |
| vs leaderboard Shoom 0.978 (#2) | +0.60 % gap | verified leaderboard |
| Overlaps | 0 / 17 IBM | canonical eval |

The cascade-saddle pipeline that the paper's central novelty defends
(§§8.9–8.12) is no longer the shipped submission but remains the
paper's load-bearing mechanism contribution. §8.13 documents the
shipped V4+Gaussian / 3-lane-ensemble system as Era 4.

| Cascade pipeline (paper §§8.9–8.12) | Value | Source |
|--------|------:|--------|
| IBM `--all` Option B (cascade + DREAMPlace 3rd lane) | 1.06650 | `results/CDLNSSACascadeDPLanePlacer_*.json` |
| NG45 `--ng45` Option B | 0.68086 | same |
| IBM uncapped ceiling (M3, E84 cascade saddle) | 1.0612 | `experiments/E84_cascading_saddle/results/cascade_*.pt` |

## Champion lineage (paper-relevant)

```
RePlAce baseline (1.4578)
  → DPO best-of (1.3834, E11)
  → CD-only (1.1193, E2)
  → CD-adaptive (1.1055, E9)
  → CD + grid-bin LNS (1.0990, E12, ADR-007 champion 2026-04-28)
  → + SA-v2 + DPO init + K-joint (1.0848, E41 composed)
  → per-bench best-of-{E25, E41} hybrid (1.08151, E48, ADR-011 2026-05-02)
  → Hessian saddle escape on E48 plateau (1.0666, E74, ADR-012 2026-05-05)
  → cascading saddle escape (1.0612 uncapped ceiling, E84)
  → cascade + A1 speedup (1.0782 cloud post-A1, 2026-05-13)
  → cascade + DREAMPlace third init lane (1.0665, Option B, 2026-05-16)
  → V4+Gaussian standalone (0.98199 M3, E117, 2026-05-20)
  → V4+Gaussian + 3-lane polish ensemble (0.98387 EPYC, 2026-05-21) ← shipped
```

## Source-of-truth rules

- Numbers in `paper.md` and `evidence.md` must trace to one of:
  - `results/experiment_log.jsonl` (the append-only run log), or
  - `results/<placer>_<timestamp>.json` (a frozen run JSON), or
  - a captured artifact in `writeup/data/`.
- If a number cannot be traced, mark it `TODO(verify)` in `paper.md`.

## Open prose / data work (highest-leverage first)

Pulled from `STORY.md` §"What's missing for a publication-ready writeup":

1. **§10 Related Work** — does not currently exist. Needs NEB / dimer /
   gentlest-ascent literature (Henkelman & Jónsson 2000, 1999;
   E & Zhou 2011), macro-placement state of the art (RePlAce,
   DREAMPlace, OpenROAD, analytical placers, Xplace, ePlace electric-
   field-based density), and transition-state methods applied to
   other combinatorial problems.
2. **Prose for §§4–8 of `paper.md`** — early sections still have
   `TODO(prose)` blocks. §§8.9–8.11 (Hessian saddle escape, the
   central innovation), §8.12 (DREAMPlace lane), and §8.13
   (V4+Gaussian / 3-lane ensemble, Era 4 shipped) already have
   publication-ready prose.
3. **§2 Background** — proxy formal definition, PlacementCost
   semantics. Currently distributed across §§1, 4, 7.
4. **Per-benchmark champion table** for §9 (E84 / Option B / v2
   per-bench numbers alongside RePlAce baseline).
5. **Figures.** λ-spectrum bar chart for ibm01 plateau (motivates the
   high-index-saddle claim); cascade lift-per-iteration plot;
   per-bench E48 / E74 / Option A / Option B / v2 comparison;
   per-bench V3 vs V4+Gaussian (E117); per-bench v2-default vs
   v2-extCD; v2 lane-pick (A/B/C) winner distribution; macro-clearance
   histogram from `analysis/macro_clearance_diagnostic/`.
6. **Frozen data CSVs.** Add `e117_gaussian_lift.csv`,
   `v2_extcd_per_bench.csv`, `v2_lane_pick_distribution.csv` to
   `writeup/data/` from the 2026-05-20/21 EPYC sweep result JSONs.
7. **Abstract prose** — current text now hits Eras 1–4 in narrative
   order; cold-read pass remaining.
