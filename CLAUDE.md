# CLAUDE.md — Macro Placement Challenge 2026

## What this project is

Partcl/HRT Macro Placement Challenge. Place 200-537 rectangular macros on a 2D
chip canvas to minimize proxy cost (wirelength + density + congestion) with zero
overlaps. Prize: $29K+. Deadline: May 21, 2026.

**Current champion (PROMOTED 2026-05-02 via ADR-011):** CDLNSSAHybrid (E48),
avg proxy **1.08151** on --all (17 IBM benchmarks). Beats public leaderboard
1.1172 by **−3.21%**, beats RePlAce 1.4578 by −25.8%, zero overlaps everywhere.
NG45 commercial transfer **0.6922** (−1.66 % vs E12, tied with E18 0.69193).
Entry: `submissions/cd_lns_sa_hybrid/placer.py`. Per-bench best-of-{E25, E41}
pipeline (E25 wins 5/17 — ibm01, 06, 07, 17, 18 — basins where SDF beats DPO;
E41 wins 12/17 — rest). Wall ~7 hr `--jobs 4` parallel (under 17-hr cap).
ADR-011 *Accepted*.

**Champion lineage:**
- E12 CDLNSGridBin (1.0990) — prior champion (ADR-007); CD plateau + grid-bin LNS overlay.
- E48 CDLNSSAHybrid (1.08151) — current champion (ADR-011); per-bench best-of-{E25, E41}.

**Component placers (called by the E48 hybrid; not separate champions):**
- E25 CDLNSSA (`submissions/cd_lns_sa/placer.py`) — SDF init pipeline used by E48 lane 1.
- E41 CDLNSSADPOKJoint (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`) — DPO init + K-joint pipeline used by E48 lane 2.

**Overnight 2026-05-01 → 02 — three follow-ups, none lifted past E48:**
- E53 GPU DPO basin polish — falsified (0/350 GPU restarts accepted at full budgets).
- E53m multi-seed hybrid (E25 + E41 s42 + E41 s1) — marginal (--all tied at 1.08128, --ng45 +0.23 %).
- E54 congestion-targeted destroy — falsified (--ng45 0.7022, ariane133 +5.14 %).

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
   - **avg < 1.082 (at-or-below champion)** → STOP. Surface to human (current champion E48 hybrid is 1.08151).
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
| `submissions/cd_lns_sa_hybrid/placer.py` | **CHAMPION** — E48 hybrid best-of-{E25, E41}, 1.08151 (ADR-011) |
| `submissions/cd_lns_gridbin/placer.py` | Prior champion — E12 CD + grid-bin LNS, 1.0990 (ADR-007) |
| `submissions/cd_lns_sa/placer.py` | Component of E48 hybrid (lane 1, SDF basin) — E25 CDLNSSA, 1.0954 standalone |
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
Champion (`submissions/cd_lns_sa_hybrid/placer.py`, E48) takes ~7 hr on --all under `--jobs 4` parallel (88 452 s aggregate CPU-time across workers); each bench runs E25 then E41 sequentially within a worker. Prior champion E12 (`submissions/cd_lns_gridbin/placer.py`) was ~7.85 hr.
Cloud not needed for development. The bottleneck is thinking, not compute.
