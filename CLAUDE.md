# CLAUDE.md — Macro Placement Challenge 2026

## What this project is

Partcl/HRT Macro Placement Challenge. Place 200-537 rectangular macros on a 2D
chip canvas to minimize proxy cost (wirelength + density + congestion) with zero
overlaps. Prize: $29K+. Deadline: May 21, 2026.

**Current champion:** CDAdaptive (E9), avg proxy 1.1055 on --all (17 IBM
benchmarks). Beats public leaderboard 1.1172 by -1.05%, beats RePlAce 1.4578
by -24.2%, zero overlaps. Entry: `submissions/cd/cd_adaptive_placer.py`.

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

1. Implement a placer variant in `submissions/`
2. Run `--fast --json` → read the JSON output
   - **fast_gate fail** → tweak parameters, try next variant
   - **fast_gate pass** → run `--all --json`
   - **avg < 1.12 (above champion)** → likely noise; verify carefully
   - **avg < 1.10 (graduate)** → STOP. Surface to human for review.
   - **avg < 1.05 (new champion)** → STOP immediately. This beats E9 Adaptive.
3. If best score hasn't improved >2% after N variants → kill hypothesis
4. Update `docs/results.md` and `docs/experiment_index.md` after every significant result

## Key files

| File | Purpose |
|------|---------|
| `results/experiment_log.jsonl` | Append-only log of all runs (read at session start) |
| `docs/roadmap.md` | Phased action plan, champion lineage, risk register |
| `docs/results.md` | Current champion (CDAdaptive) per-benchmark tables |
| `docs/approach.md` | Current CD-on-incremental-evaluator architecture + prior approaches |
| `docs/problem.md` | Formal mathematical problem statement |
| `docs/experiment_index.md` | Rigorous catalog of every experiment (live + falsified) |
| `docs/lp_hpwl_diagnostic.md` | E8 — proxy decomposition (6% WL / 20% density / 74% congestion) |
| `macro_place/evaluate.py` | Evaluation harness (don't modify unless infra work) |
| `macro_place/objective.py` | Proxy cost computation |
| `macro_place/incremental_evaluator.py` | E1 — 4657× speedup; load-bearing for CD |
| `macro_place/benchmark.py` | Benchmark dataclass (PyTorch tensors) |
| `submissions/cd/cd_adaptive_placer.py` | **CHAMPION** — full-proxy CD + plateau detection |
| `submissions/cd/cd_only_placer.py` | Prior champion (CDOnly fixed-budget) |
| `submissions/cd/sdf_init.py` | SDF initialization (used by champion) |
| `submissions/examples/` | Reference placers (greedy, random) |
| `writeup/` | Innovation-prize writeup; killed-hypothesis history (DPO, polyhedra, theory, leaderboard recipe) |

## At session start

1. Read `results/experiment_log.jsonl` to know what's been tried
2. Read `docs/roadmap.md` for current phase and next actions
3. Read `docs/results.md` for latest results
4. Read `docs/approach.md` for current strategy
5. Ask what to work on, or continue the most promising alive hypothesis

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
Champion (`cd_adaptive_placer.py`) takes ~5 hr on --all (per-benchmark plateau).
Cloud not needed for development. The bottleneck is thinking, not compute.
