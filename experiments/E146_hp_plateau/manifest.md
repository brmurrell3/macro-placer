---
id: E146
name: hp_plateau
status: in_progress
parent: E129
created: 2026-05-21
decided: null
champion_at_time: 0.987       # cd_lns_sa_cascade_stacked_periphery combined 21-bench
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E146: hp_plateau

## Hypothesis

Hyperparameter plateau-pick. Multi-restart (E129) explores SEED variance —
each FP32-nondeterministic seed lands in a slightly different basin. But
the descent itself (lr_frac, gamma_end_frac, overlap_lambda_end) controls
which basin family the descent is BIASED toward. Three differently-tuned
descents will explore HP space, not seed space, and may catch per-bench
HP optima that no single fixed config hits.

Same total wall as a 3-restart pipeline (3 descents), but broader basin
coverage. Mechanism: each config converges to slightly different basin;
best-of-3 catches the per-bench-optimal HP.

## Method

Three configs, run in series, each pipelined: V4+Gauss descent + legalize
+ light CD polish (200 s, partial). Pick best basin by canonical proxy.
Final CD polish on chosen basin: 600 s. Budget total: 1800 s.

Configs:
- A (current default): lr_frac=0.005, gamma_end_frac=5e-5, overlap_lambda_end=10.0
- B (lower lr, looser overlap): lr_frac=0.003, gamma_end_frac=5e-5, overlap_lambda_end=5.0
- C (higher lr, sharper gamma): lr_frac=0.008, gamma_end_frac=2e-5, overlap_lambda_end=20.0

Reuses `experiments/E127_v4_gaussian/code/smooth_global_placer_v4_gaussian.py`
(DO NOT mutate). Reuses `macro_place/cd_core.py::run_cd_adaptive` as the
CD polish kernel.

## Kill gate

Smoke ibm04 on M3 at 1800 s budget:
  - FAIL if final proxy > thinkorplace-v2 M3 ibm04 reference (will fetch
    from cache) by more than +1 % — HP plateau is net harmful at this budget.
  - FAIL if total wall > 2100 s — budget overrun.
  - PASS if proxy is below the per-config A basin by at least 0.5 % and
    zero overlaps.

## Generalization check

If smoke passes, recommend `--all` on M3 to confirm 17-bench average
lifts vs v2 baseline 0.984 (M3). Promote to submission only if --all
avg < 0.984 AND zero overlaps everywhere.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/placer.py` (E146HPPlateauPlacer).
- Reuses `experiments/E127_v4_gaussian/code/smooth_global_placer_v4_gaussian.py`.
- Results: `results/smoke_ibm04.json`.
