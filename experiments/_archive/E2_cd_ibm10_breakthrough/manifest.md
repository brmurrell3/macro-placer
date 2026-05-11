---
id: E2
name: cd_ibm10_breakthrough
status: graduated
parent: E1
created: 2026-04-27
decided: 2026-04-27
champion_at_time: 1.3834
outcome: 1.0632
champion_delta: -0.3202
graduated_to: submissions/cd_only/placer.py
superseded_by: E9
---

# E2: cd_ibm10_breakthrough

## Hypothesis
The E8 LP-HPWL diagnostic showed proxy is 6 % WL / 20 % density / 74 %
congestion, so HPWL-only CD caps at ~5 % gain; full-proxy CD on the
incremental evaluator should be the right architecture. Validate single-bench
on ibm10 (DPO's basin-locked benchmark with the largest gap to leaderboard)
before committing to a full `--all` run.

## Method
Run full-proxy coordinate descent from the SDF init, using
`macro_place/incremental_evaluator.py` for per-move proxy queries. Per-axis
breakpoint search with golden-section fallback; hard-overlap rejection. Wall
budget 2 400 s on ibm10 (CD-only; SDF init excluded from the budget).

## Kill gate
Single-bench validation with ≤ 1.20 to graduate to `--all`. Anything larger
indicates CD is no better than DPO and the architecture should not be
scaled up.

## Generalization check
After ibm10 single-bench passes, scale up to a full `--all` 17-bench run
(this became CDOnly).

## Outcome (filled when decided)
**Graduated. Sparked the CDOnly champion (and ultimately E9).**

| Method | proxy | WL | Density | Congestion |
|---|---:|---:|---:|---:|
| SDF init (E2 start) | 1.4112 | 0.0687 | 0.7614 | 1.9235 |
| RePlAce baseline (avg, 17 IBM) | 1.4578 | – | – | – |
| DPO best_of_v2 (champion on ibm10) | 1.254 | 0.080 | 0.269 | 0.905 |
| **E2 final (CD-only, 2 400 s from SDF)** | **1.0632** | **0.0790** | **0.5700** | **1.3984** |
| Leaderboard target (avg, all 17) | 1.117 | – | – | – |

13 sweeps, 15 000 accepted moves, 71 336 per-axis probes, 0 golden-section
fallbacks, 0 overlaps. Improvement vs SDF init: **24.7 %**. WL goes UP
13–15 % while density drops 21–32 % and congestion drops 23–40 % — exactly
the trade-off DPO can't make because RUDY misidentifies which cells are
congested.

Promoted to `submissions/cd_only/placer.py` as a fixed-budget `--all`
placer, then superseded by E9 plateau-adaptive CD on 2026-04-27.

## Pointers
- Code: graduated to `submissions/cd_only/placer.py`. The diagnostic
  driver lives at `scripts/cd_ibm10_diagnostic.py` (kept for reference, not
  moved).
- Discussion: `writeup/evidence.md` §7.2; `docs/experiment_index.md` (E2 row);
  `writeup/cd_ibm10_results.md` (per-sweep trajectory).
- Parent: `experiments/E1_incremental_evaluator/manifest.md`.
- Successor: `experiments/E9_plateau_detection/manifest.md`.
