# analysis/ — diagnostic probes

This directory holds *diagnostic probes*: one-off analyses that answer "why
does the system behave this way?" questions about the placement pipeline.
They produce understanding, not score gains.

## Distinction from experiments/

| | `experiments/` | `analysis/` |
|---|---|---|
| Goal | improve avg proxy | explain a phenomenon |
| Output | a score, logged in `results/experiment_log.jsonl` | a finding (correlation, ratio, falsification) |
| Re-run cadence | until graduated or killed | typically one-shot |
| Lifetime | superseded by next champion | durable; cited from writeups and ADRs |

A probe gets a directory under `analysis/` when it answers a structural
question whose conclusion changes how we plan future experiments.

## Naming convention

Each probe lives in `<short_name>/` containing:

- `README.md` — the question, the run command, the headline finding, whether
  it should be re-run, and where the frozen output lives.
- The probe's source (`probe.py`, `<name>.py`, or a markdown writeup).
- Optionally, frozen captured output (e.g. `data/rudy_ibm01.txt`).

Probes are typically static — re-run only when a structural assumption
changes (RUDY model edited, LP cost function modified, CD move type changed).

## Index

| Probe | Question | Headline finding |
|---|---|---|
| [`rudy_fidelity/`](rudy_fidelity/) | Does RUDY's congestion grid match real ICCAD-evaluator congestion cell-by-cell? | No — top-5% hotspot Jaccard 0.057 (near-random); 77% of real top-5% cells not in RUDY's top-10%; gap is structural (L-routing, macro blockage, smoothing), not a uniform scale. |
| [`lp_hpwl_diagnostic/`](lp_hpwl_diagnostic/) | Does LP-HPWL predict refined proxy cost? | No — ρ = −0.001 across 24 feasible topologies on ibm01. Proxy is congestion-dominated (74%) and HPWL is blind to congestion; this killed every LP-only architecture. |
| [`multi_init_probe/`](multi_init_probe/) | Does SDF-jitter multi-init find different basins for CD? | No — 0/8 jittered inits on ibm09 beat the unjittered control. SDF→CD is contractive; perturbing strictly hurts. |
| [`lns_escape_probe/`](lns_escape_probe/) | Can subset-CD destroy-and-reinsert escape converged CD plateaus? | No — 0/24 samples on ibm09+ibm12 improved baseline by ≥0.1%. Escape requires a *different move type*, which E12 grid-bin LNS provides. |
