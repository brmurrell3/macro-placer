# CLAUDE.md — Macro Placement Challenge 2026

## What this project is

Partcl/HRT Macro Placement Challenge. Place 200-537 rectangular macros on a 2D
chip canvas to minimize proxy cost (wirelength + density + congestion) with zero
overlaps. Prize: $29K+. Deadline: May 21, 2026.

Target: beat RePlAce baseline (1.4578 avg proxy cost on 17 IBM benchmarks).
Current best submission: Will's SA seed (1.5338).

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

1. Pick a hypothesis from `hypotheses.yaml` (alive ones only)
2. Implement a placer variant in `submissions/`
3. Run `--fast --json` → read the JSON output
4. Check against thresholds in `hypotheses.yaml`:
   - **fast_gate fail** → tweak parameters, try next variant
   - **fast_gate pass** → run `--all --json`
   - **avg < 1.50 (graduate)** → STOP. Surface to human for review.
   - **avg < 1.46 (champion)** → STOP immediately. This beats RePlAce.
5. If best score hasn't improved >2% after N variants → kill hypothesis
6. Update `docs/hypothesis-status.md` after every significant result

## Key files

| File | Purpose |
|------|---------|
| `hypotheses.yaml` | Per-hypothesis kill/graduate thresholds |
| `results/experiment_log.jsonl` | Append-only log of all runs (read at session start) |
| `docs/hypothesis-status.md` | Human-readable status dashboard |
| `docs/novel-solutions.md` | The 6 hypotheses with theory and kill criteria |
| `macro_place/evaluate.py` | Evaluation harness (don't modify unless infra work) |
| `macro_place/objective.py` | Proxy cost computation |
| `macro_place/benchmark.py` | Benchmark dataclass (PyTorch tensors) |
| `submissions/examples/` | Reference placers (greedy, random) |

## At session start

1. Read `results/experiment_log.jsonl` to know what's been tried
2. Read `docs/hypothesis-status.md` for current state of each hypothesis
3. Check `hypotheses.yaml` for thresholds
4. Ask what to work on, or continue the most promising alive hypothesis

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
Cloud not needed for development. The bottleneck is thinking, not compute.
