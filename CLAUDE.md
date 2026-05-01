# CLAUDE.md — Macro Placement Challenge 2026

## What this project is

Partcl/HRT Macro Placement Challenge. Place 200-537 rectangular macros on a 2D
chip canvas to minimize proxy cost (wirelength + density + congestion) with zero
overlaps. Prize: $29K+. Deadline: May 21, 2026.

**Current champion:** CDLNSGridBin (E12), avg proxy 1.0990 on --all (17 IBM
benchmarks). Beats public leaderboard 1.1172 by -1.63%, beats RePlAce 1.4578
by -24.6%, zero overlaps. Entry: `submissions/cd_lns_gridbin/placer.py`.
Prior champion: CDAdaptive (E9) at 1.1055.

**STRONGEST VERIFIED CANDIDATE (awaiting human decision):** CDLNSSAHybrid
(E48) at **1.08151** on --all (**−1.59% vs E12 1.0990**, **−3.21% vs
leaderboard 1.1172**, −0.30% vs E41 candidate, −0.76% vs E18 candidate).
**Per-bench best-of-{E25, E41} pipeline.** E25 wins 5/17 (ibm01, 06,
07, 17, 18 — basins where SDF outperforms DPO); E41 wins 12/17 (rest).
Zero overlaps everywhere. Wall 7 hr wall-clock under --jobs 4 parallel
(under 17-hr cap). Code at `experiments/E48_hybrid_e25_e41/code/cd_lns_sa_hybrid.py`;
ADR-011 needed (in progress).

**Prior candidate (superseded by E48 if ADR-011 accepted):** CDLNSSADPOKJoint
(E41) at 1.0848 (−1.29% vs E12). 14 wins / 3 losses vs E25; biggest
hard-plateau wins (ibm11 −4.06%, ibm14 −1.73%, ibm15 −1.24%). NG45
0.69022. Code at `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`;
ADR-010 *Proposed* (mark *Superseded* on ADR-011 acceptance).

**Older candidate:** CDLNSSADPOInit (E18) at 1.08979 (−0.84% vs E12, 4/4
NG45 wins). ADR-009 *Proposed* (mark *Superseded* on ADR-011).

**Oldest candidate:** CDLNSSA (E25) at 1.0954 (−0.33% vs E12). Code at
`submissions/cd_lns_sa/placer.py`. ADR-008 *Proposed* (mark *Superseded*
on ADR-011).

Reference baselines: RePlAce 1.4578, Will's pre-fork seed 1.5338, leaderboard
target 1.1172.

## Your role

You autonomously implement, evaluate, iterate, and kill hypothesis variants.
The human's role is (1) sourcing hypotheses and (2) judging graduated results.
**Never ask "should I continue?" for routine decisions.** The thresholds answer that.

## Commands

```bash
# Fast iteration (4 benchmarks, ~10s, structured output)
uv run evaluate <placer.py> --fast --json --hypothesis <name>

# Full validation (17 benchmarks, ~40s)
uv run evaluate <placer.py> --all --json --hypothesis <name>

# NG45 commercial designs (top-7 submissions only)
uv run evaluate <placer.py> --ng45 --json

# Single benchmark for debugging
uv run evaluate <placer.py> -b ibm01
```

## Feedback loop

1. **Set up the experiment.** Copy `experiments/_template/` to
   `experiments/E<NN>_<short_name>/`. Fill in the manifest's `Hypothesis`,
   `Method`, `Kill gate`, and `Generalization check`. Put placer code in
   `code/`. Find the next free `E<NN>` with `ls experiments/ | grep '^E'`.

2. **Run.** Always pass `--hypothesis E<NN>` so the row in
   `results/experiment_log.jsonl` is attributable:
   ```
   uv run evaluate experiments/E<NN>_*/code/<placer>.py --fast --json --hypothesis E<NN>
   ```

3. **Decide from the JSON.**
   - **fast_gate fail** → tweak parameters, try next variant.
   - **fast_gate pass** → run `--all --json`.
   - **avg < 1.12** → likely noise; verify carefully.
   - **avg < 1.099 (at-or-below champion)** → STOP. Surface to human (current champion E12 is 1.0990; E25 candidate at 1.0954 is awaiting promotion decision).
   - **avg < 1.05** → STOP immediately. New champion territory.

4. **If killed:** Set the manifest's `status: falsified`, fill `decided`
   and `outcome`, write the killing data into `Outcome`. Falsified
   experiments stay forever — they're the signpost that stops the next
   person re-running the dead end.

5. **If graduated:** Surface to human first. Promotion = move code to
   `submissions/<name>/placer.py`, set `status: graduated, graduated_to: ...`,
   write an ADR in `docs/decisions/NNN_short_name.md` if the decision is
   load-bearing.

If best score hasn't improved by >2 % after N variants → kill the hypothesis.

## On a result, update these

| When | Update |
|------|--------|
| Any terminal decision | Manifest's `status` / `decided` / `outcome` / `champion_delta` |
| Falsified | Manifest's `Outcome` section with the killing data |
| Graduated | Move code to `submissions/<name>/placer.py`; set `graduated_to:` |
| New champion | CLAUDE.md champion line; `docs/results.md`; `submissions/README.md`; `writeup/evidence.md` §1 |
| Structural decision | New ADR in `docs/decisions/NNN_short_name.md` (immutable once accepted) |
| Paper-relevant finding | `writeup/evidence.md` (the experimental archive) |
| Always | `docs/experiment_index.md` (status table) |

**Manifest vs ADR.** A manifest records *what you tried and how it went*
(per-experiment, many per generation). An ADR records *a decision about
how the project works going forward* (rare, structural, immutable). Most
experiments only need a manifest. ADRs are reserved for things like
"we abandon LP-HPWL ranking" or "we promote E12 as champion."

## Key files

### Where to put new work

| Path | Role |
|------|------|
| `experiments/_template/` | Boilerplate for a new experiment manifest |
| `experiments/E<NN>_*/` | Per-experiment manifest + `code/` + `notes.md` |
| `analysis/<name>/` | Diagnostic probes (one-shot, not score-improvement experiments) |
| `submissions/<name>/placer.py` | Graduated placer (post-promotion only) |

### Read at session start

| Path | Role |
|------|------|
| `results/experiment_log.jsonl` | Append-only log of all runs |
| `docs/roadmap.md` | Phased action plan, champion lineage, risk register |
| `docs/results.md` | Current champion per-benchmark tables |
| `docs/approach.md` | Current architecture + prior approaches |
| `docs/experiment_index.md` | Catalog of every experiment (live + falsified) |
| `experiments/*/manifest.md` | Per-experiment hypothesis, status, kill gate, outcome |

### Decisions, gotchas, formal spec

| Path | Role |
|------|------|
| `docs/decisions/` | ADRs — structural decisions, immutable once accepted |
| `docs/gotchas.md` | Codebase footguns (net_nodes empty bug, single-step revert, etc.) |
| `docs/problem.md` | Formal mathematical problem statement |
| `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` | E8 — 6 % WL / 20 % density / 74 % congestion |

### Library and submissions

| Path | Role |
|------|------|
| `macro_place/evaluate.py` | Evaluation harness (don't modify unless infra work) |
| `macro_place/objective.py` | Proxy cost computation |
| `macro_place/incremental_evaluator.py` | E1 — 4657× speedup; load-bearing for CD |
| `macro_place/benchmark.py` | Benchmark dataclass (PyTorch tensors) |
| `macro_place/sdf_init.py` | SDF initialization (used by every champion) |
| `submissions/cd_lns_gridbin/placer.py` | **CHAMPION** — E12 CD + grid-bin LNS, 1.0990 |
| `submissions/cd_lns_sa/placer.py` | Champion candidate — E25 CD + LNS + SA-v2, 1.0954 (not yet promoted) |
| `submissions/cd_adaptive/placer.py` | Prior champion — E9 CDAdaptive, 1.1055 |
| `submissions/cd_only/placer.py` | Prior-prior — CDOnly fixed-budget, 1.1193 |
| `submissions/examples/` | Reference placers (greedy, random) |

### Writeup (innovation prize)

| Path | Role |
|------|------|
| `writeup/paper.md` | Canonical draft with TODO markers |
| `writeup/evidence.md` | Experimental archive — every paper.md datapoint lives here |
| `writeup/contributions.md` | Claim / novelty / evidence registry |
| `writeup/theory.md` | Supplementary mathematical material |

## At session start

1. Read `results/experiment_log.jsonl` to know what's been tried
2. Read `docs/roadmap.md` for current phase and next actions
3. Read `docs/results.md` for latest results
4. Read `docs/approach.md` for current strategy
5. Find live experiments: `grep -l "status: in_progress" experiments/*/manifest.md`
6. Ask what to work on, or continue the most promising alive hypothesis

## Writing a placer

A placer is a Python file with a class that has a `place(self, benchmark) -> Tensor` method.
The returned tensor is `[num_macros, 2]` with (x, y) center coordinates.
Fixed macros (`benchmark.macro_fixed`) must not be moved.
Hard macros must not overlap. Soft macros can overlap.

See `submissions/examples/greedy_row_placer.py` for reference.
See `SETUP.md` for the full API (Benchmark fields, compute_proxy_cost, validation).

## Hard constraints

- **Zero overlaps.** Any overlap on any benchmark = disqualified.
- **Fixed macros stay fixed.** Don't move them.
- **Canvas bounds.** All macros must be fully within the canvas.

## Hardware

M3 Max (36GB unified, MPS/PyTorch). Fast subset: ~10s. Full --all: ~40s.
Champion (`submissions/cd_lns_gridbin/placer.py`) takes ~7.85 hr on --all (CD plateau + LNS overlay).
Champion candidate (`submissions/cd_lns_sa/placer.py`, E25, not promoted) takes ~10.3 hr on --all.
Cloud not needed for development. The bottleneck is thinking, not compute.
