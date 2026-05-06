---
id: E83v2
name: clock_aware_tight
status: in_progress
parent: E83
created: 2026-05-06
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E83v2: clock_aware_tight

## Hypothesis

E83 v1's `--all` results showed 5/17 benches finished within 1 min of the
60-min cap on Windows.  EPYC's 1.2-1.5× slowdown would push those over →
disqualification.

E83 v2 tightens phase budgets to give a real safety margin while keeping
the same overall algorithm structure.

## Method

Identical algorithm to E83 v1; only constants change.  Same workers, same
clock-aware Hessian degradation logic.

| Knob | v1 | v2 | Effect |
|---|---|---|---|
| CD cap | 1500 s | **1200 s** | -5 min/lane |
| LNS budget | 360 s | **240 s** | -2 min/lane |
| SA budget | 360 s | **240 s** | -2 min/lane |
| K-joint | 0 (disabled) | 0 (disabled) | - |
| **Phase 1+2 max wall (parallel max)** | **37 min** | **28 min** | **-9 min** |
| Hessian full polish | 180 s | **120 s** | per-trial cheaper |
| Hessian full ε values | {0.3, 1.0, 3.0} | **{0.3, 1.0}** | 6 → 4 polishes |
| **Hessian full max wall** | **18 min** | **8 min** | **-10 min** |
| Hessian full threshold | 15 min remaining | 12 min | wider |
| **Total max wall (M3 equiv)** | **55 min** | **36 min** | **-19 min margin** |
| **Projected EPYC max wall (×1.5)** | **82 min ❌** | **54 min ✓** | **fits cap** |

## Kill gate

* `--all` proxy > 1.10 → falsified (worse than weak baselines).
* `--all` proxy regression > 2 % vs E83 v1 (1.0859) → falsified.
* Wall > 50 min on any --fast benchmark on Windows → falsified.

## Generalization check

* `--all` proxy ≤ 1.10 (still beats public leaderboard 1.1172).
* All walls < 50 min on Windows → safe EPYC margin.
* `--ng45` proxy ≤ 0.700 (matches E48 region).

## Outcome (filled when decided)

[TBD — `--all` running.]

## Pointers

* Code: `code/cd_lns_sa_hessian_clock_v2.py` (subclasses E83 v1 placer,
  overrides class constants only).
* Workers shared with E83 v1 (`experiments/E83_clock_aware/code/_worker.py`).
* Results: `results/`.
