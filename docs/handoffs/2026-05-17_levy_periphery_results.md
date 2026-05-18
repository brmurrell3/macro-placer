# 2026-05-17 — Lévy + periphery overnight results (E107 agent)

Summary of the E107 periphery-bias agent's overnight cross-validation
work. Superseded by the **1.0575 stacked_periphery champion** (see
`2026-05-17_morning_champion.md`); kept here for the per-variant
empirical record.

## TL;DR

Five CPU placer variants run end-to-end on `--all` IBM 17:

| Placer | IBM 17 | NG45 4 | Notes |
|---|---:|---:|---|
| PATH B (`cd_lns_sa_cascade_dp_lane`) | 1.06650 | 0.68086 | GPU-conditional (Lambda) |
| **Stacked+periphery (champion)** | **1.0575** | **0.6893** | Other agent; CPU only |
| Lévy (`cd_lns_sa_cascade_levy`) | 1.07268 | 0.67897* | Best CPU before stacked |
| Portfolio Lévy (`cd_lns_sa_cascade_portfolio_levy`) | 1.07404 | (untested) | 4-weight Hessian |
| Multidir (`cd_lns_sa_cascade_multidir`) | 1.07415 | **0.6779** ⭐ | Best NG45 |
| PATH A (`cd_lns_sa_cascade/placer_adaptive`) | 1.07564 | 0.68102 | Baseline |

*Lévy NG45 verified on M3 (`cd_lns_sa_cascade_levy_periphery` wrapper,
periphery rejected on all 4 → effectively plain Lévy).

## Per-bench wins (cross-variant)

Different placers win on different benches. Within-budget runtime
best-of-N theoretical ceiling (Lévy ∪ Portfolio per-bench): **1.07029**
(only -0.36% above PATH B 1.06650; not deployable as one placer.py
within 60-min wall cap).

Multidir wins NG45 because its multi-direction polish layer compounds
on PATH A's cascade output. Lévy wins IBM 11/17 benches but Portfolio
beats it on ibm01, ibm07, ibm13, **ibm18 (-1.80%)**.

## E107 periphery wrapper findings (12 multi-seed tests)

Cross-bench α=0.01 multi-seed (5 random seeds each):

- Random kicks WIN proxy 10/12 by 1-14σ vs periphery direction.
- Periphery direction reliably pushes TOWARD edge (Δedge negative in
  10/12); Random reliably drifts AWAY from edge (Δedge positive).
- Center direction always WORST (proxy + drifts inward).
- ibm10 uniquely fragile (any push at α=0.005 or 0.01 catastrophic).
- Strategic implication: periphery wrapper is safe (strict-accept
  never regresses) and useful only if OpenROAD WNS/TNS validation
  shows the periphery shift improves Grand Prize objective.

## Best-of-N analysis (cached placements)

Per-bench best across 189 cached `.pt` files: **0.97957 21-bench
aggregate** (1.05031 IBM, 0.67895 NG45). 6 of 17 IBM benches have
deeper minima in `experiments/E91_dp_full_polish/` and
`experiments/E84_cascading_saddle/` than any production placer
currently reaches.

Realizing the 1.05031 IBM ceiling would require a runtime best-of-N
that runs DP-lane (GPU-gated) + cascade + hessian + multi_dp within
the 55-min budget — out of scope before the deadline.

## Code patches shipped

- `submissions/cd_lns_sa_cascade_levy/placer.py`: catch `E25
  RuntimeError` on 1-pixel float32 overlaps, fall back to E41-only.
  Prevents nvdla-style failures.
- `submissions/cd_lns_sa_cascade_portfolio_levy/placer.py`: same fix.

(Note: the canonical fix is in
`submissions/cd_lns_sa/placer.py` with `project_overlaps` retry,
shipped by the champion agent — these two wrappers were patched
defensively before that fix landed.)

## Failed paths

- DP hyperparam dispatch (Path A from `2026-05-15_priorities.md`):
  blocked on GPU; AWS-GPU quota was approved late and used for E101
  (Lévy curvadapt) instead.
- Periphery wrapper on cached placements: works on cached (ariane133
  -0.95%), but on fresh tight Lévy minima the periphery push only
  costs proxy (+0.83%) without OpenROAD validation. Conservative
  accept correctly rejects.

## Commits

```
88f6cc9 Portfolio Lévy --all 1.07404
2b4922b Lévy cascade --all 1.07268 (best CPU)
faa2332 Multidir --ng45 0.6779 (best NG45)
cecd892 Multidir --all 1.07415 (-0.14% vs A, below promo gate)
7653155 Best-of-N ceiling analysis (0.97957)
704f0c0 E107 chain: 12-test multi-seed validation
6c66431 ariane133 fresh wrapper: 0.65212 NEW BEST (pre-champion)
```
