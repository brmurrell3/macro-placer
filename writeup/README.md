# `writeup/` directory map

Six markdown files. Each has a single, distinct role.

| File | Role | Status |
|------|------|--------|
| **`STORY.md`** | **Narrative arc.** Three-act structure, section mapping to standard CAD-paper format, priority-ordered TODO list for publication readiness. Start here when planning what to write next. | current |
| **`paper.md`** | **Canonical draft.** Section scaffolding + locked-in numbers + inline `TODO(...)` markers. §§8.9–8.11 (the central innovation) have publication-ready prose; earlier sections still have `TODO(prose)` placeholders for the prose pass. | partly drafted |
| **`evidence.md`** | **Experimental archive.** Per-bench tables, ablations, dead ends, diagnostic experiments. Every datapoint cited by `paper.md` lives here. | needs E74/E84/A4 per-bench tables |
| `contributions.md` | Claim / novelty / evidence registry, numbered 1–23. Drafting aid — the "have I defended every claim?" checklist. Will fold into `paper.md` before final submission. | §§21–23 added for E74/E84/A1 |
| `theory.md` | Supplementary literature material (polyhedral decomposition, mountain-pass / NEB theory, complexification, etc.) + the explicit list of literature connections *excluded* from the paper body. | sufficient for §10 Related Work source material |
| `README.md` | This file. | current |

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

**`submissions/cd_lns_sa_cascade/placer_adaptive.py`** —
`CDLNSSACascadeAdaptivePlacer`.

| Metric | Value | Source |
|--------|------:|--------|
| IBM avg `--all` (post-A1, cloud EPYC, 60-min cap) | **1.0771** | `results/CDLNSSACascadeAdaptivePlacer_20260513_073143.json` (A4-v2) |
| NG45 avg `--ng45` (post-A1) | **0.6870** | `results/CDLNSSACascadeAdaptivePlacer_20260513_091359.json` |
| IBM uncapped ceiling (M3) | 1.0612 | cached cascade `experiments/E84_cascading_saddle/results/cascade_*.pt` |
| vs RePlAce 1.4578 | −26.1 % | published baseline |
| Overlaps | 0 / 17 IBM + 4 NG45 | canonical eval |

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
  → cascade + A1 speedup (1.0771 cloud post-A1, 2026-05-13) ← current submission
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
   DREAMPlace, OpenROAD, analytical placers), and transition-state
   methods applied to other combinatorial problems.
2. **Prose for §§4–8 of `paper.md`** — early sections still have
   `TODO(prose)` blocks. §§8.9–8.11 (the central innovation) already
   have publication-ready prose as of 2026-05-13.
3. **§2 Background** — proxy formal definition, PlacementCost
   semantics. Currently distributed across §§1, 4, 7.
4. **Per-benchmark champion table** for §9 (E84 / A4-v2 per-bench
   numbers alongside RePlAce baseline).
5. **Figures.** λ-spectrum bar chart for ibm01 plateau (motivates the
   high-index-saddle claim); cascade lift-per-iteration plot;
   per-bench E48 / E74 / A4-v2 comparison; macro-clearance histogram
   from `analysis/macro_clearance_diagnostic/`.
6. **Abstract prose** — current text is ~250-word TODO bullets;
   needs prose hitting the three-act arc.
