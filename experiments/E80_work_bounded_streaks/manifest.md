---
id: E80
name: work_bounded_streaks
status: in_progress
parent: E79
created: 2026-05-05
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E80: work_bounded_streaks

## Hypothesis

E79 fits the 60-min cap on 3/4 `--fast` benches but fails on ibm13 (78 min)
because E25 and E41 lanes consume their FULL phase budgets when no
plateau is reached.  Adding **plateau/saturation streak termination**
should let phases exit early when stuck, reclaiming ~10–15 min on the
hard benches without sacrificing proxy on the benches that already
plateau.

Specifically:
- LNS: terminate on **5 consecutive non-improving samples** (vs current 1)
  — the current 1-streak is too eager; 5 gives more confidence the
  plateau is real.
- SA-v2: terminate on **1000 moves with no global-best improvement**
  — currently SA-v2 has wall-only termination, so it spends its full
  600s on benches where best was found at t=0.
- K-joint: **dropped entirely** (Mitigation #4) — Hessian saddle escape
  finds deeper minima than K-joint K=3, so the K-joint phase is
  redundant when Hessian comes after.  Saves ~10 min/bench.

## Method

- Streaked LNS/SA functions in `code/streaked_phases.py`.
- Worker (`code/_worker.py`) monkey-patches `run_lns_gridbin` and
  `run_sa_polish_v2` on the source modules in the subprocess BEFORE
  importing E25/E41, so champion source files are not modified.
- Worker also replaces `run_kjoint_lns` with a no-op (Mitigation #4).
- Main placer (`code/cd_lns_sa_hessian_streaked.py`) is otherwise
  identical to E79: parallel E25⊥E41 + compressed Hessian saddle escape.
- ε grid expanded to (0.1, 0.3, 1.0, 3.0) — picks up the small-ε
  finding from E77 ibm12 where ε=0.1 was the only productive setting.

## Kill gate

- Wall > 60 min on any --fast benchmark → falsified.
- Proxy regression > 1 % on any --fast benchmark vs E79 → falsified.
- Any overlap on any benchmark → falsified.

## Generalization check

- `--all` proxy ≤ 1.0800 (within 1.3 % of E74 1.0666).
- `--all` wall ≤ 60 min/bench on largest IBM (ibm17, ibm18).
- `--ng45` proxy ≤ 0.700.

## Outcome (filled when decided)

[TBD]

## Pointers

- Code:
  - `code/streaked_phases.py` (streaked LNS + SA-v2)
  - `code/_worker.py` (monkey-patches before subprocess E25/E41)
  - `code/cd_lns_sa_hessian_streaked.py` (main placer)
- Reference: `submissions/cd_lns_sa_hessian/placer.py:_saddle_escape` reused.
- Results: `results/`.
