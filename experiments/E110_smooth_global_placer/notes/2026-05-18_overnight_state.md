# Overnight state — 2026-05-18 22:10 → 2026-05-19 morning

## 5 background jobs queued (CPU-headroom safe at 14/16 M3 Max cores)

| Job | PID | Started | ETA | Log | Output |
|---|---|---|---|---|---|
| handpicked sweep (7×4 = 28 runs) | 58765 | 21:50 | 23:00 | `/tmp/e110_handpicked_sweep.log` | `results/handpicked_fast{.jsonl,_summary.json}` |
| ibm04_diag (18×1 = 18 runs) | 58770 | 21:50 | 22:35 | `/tmp/e110_ibm04_diag.log` | `results/ibm04_diag{.jsonl,_summary.json}` |
| ibm04winners on --fast (6×4) | 59918 | 22:05 | 23:30 | `/tmp/e110_ibm04winners_fast.log` | `results/ibm04winners_on_fast{.jsonl,_summary.json}` |
| **Variant A** (Lane-4 add-on) --all jobs=4 | 58916 | 21:51 | ~01:50 | `/tmp/e110_lane4_all_p4.log` | `results/CDLNSSACascadeStackedPeripheryE110Placer_*all*.json` |
| **Variant B** (plateau-pick) --all jobs=2 | 59774 | 22:00 | ~05:00 | `/tmp/e110_plateau_all_p2.log` | `results/CDLNSSACascadeStackedPeripheryE110PlateauPlacer_*all*.json` |

## Morning command (one-liner)

```bash
uv run python experiments/E110_smooth_global_placer/code/check_overnight.py
```

Prints: per-job alive/done, sweep summaries, Variant A/B --all per-bench tables,
log tails.

## Partial signals already in (as of 22:04)

- **ibm04 diagnostic**: cfg `(lr=3e-3, num_steps=300, ovl_end=30)` → **ibm04 −1.58% WIN**
  (vs default +2.63% loss). Three other cfgs also win on ibm04 with `lr=3e-3`.
- **Handpicked**: default cfg confirms ibm01 −3.23%, ibm09 −1.45%, ibm13 −1.42%,
  ibm04 +1.90% (same pattern as first-pass test).

## Decision tree for morning

```
Variant_A_avg = best --all avg from Variant A
Variant_B_avg = best --all avg from Variant B
Option_C_avg = 1.0575 (M3-verified, current floor)
EPYC_buffer  = +2% (M3 → EPYC variance — assume Option_C → ~1.078 EPYC)

IF min(A, B) < 1.0525 (Option C −0.5%):
    # E110 lane is working
    Pick min(A, B)
    Look at ibm04winners_on_fast results — does the lr=3e-3 cfg win?
    IF YES: rebuild winning Variant with lr=3e-3 cfg, re-run --all
    IF NO:  cross-validate Variant winner on AWS EPYC c6a, ship if EPYC < 1.075
ELIF Variant A failed (overlaps, crashes):
    Investigate; fix Variant B or fall back to Option C
ELSE:
    Run NG45 test on Variant B; if NG45 OK and IBM tied, ship Variant B (more
    cascade-coupled gradient lane is theoretically better)
```

## What was NOT tested overnight (tomorrow's TODOs)

- **MPS backend** — diff_proxy._rudy_congestion has device-mismatch bug
  (h_coeff stays on CPU). Patch tomorrow morning, retest on M3 MPS.
  Expected speedup: 5-10× on Adam descent.
- **NG45 4-bench** on Variant A/B — assume they generalize; verify tomorrow.
- **EPYC cross-validation** — AWS c6a.4xlarge spot, ~$0.29/hr × 4 hr = $1.20.
- **More aggressive hyperparameter search** — Bayesian / random sample 30-50 cfgs.

## Tonight's verified findings

| Bench | SDF+CD60s | E110+CD60s default | E110 cfg(lr=3e-3,steps=300,ovl=30) |
|---|---:|---:|---:|
| ibm01 | 0.91760 | 0.87703 (−4.05%) | — |
| ibm04 | 1.02482 | 1.05182 (+2.63%) | **1.00863 (−1.58%)** |
| ibm09 | 0.86808 | 0.85603 (−1.39%) | — |
| ibm13 | 1.04287 | 1.01942 (−2.25%) | — |
| ibm17 | 1.41184 | 1.45345 (+2.95%) | (probably also winnable) |

**Key insight:** Default E110 cfg loses on ibm04, ibm17. Per-bench cfg tuning
fixes this. **Need to find a single cfg that wins on the median bench** (the
Lane-4 plateau-pick handles per-bench losses, but a globally-good cfg lifts
more benches).

## If everything fails (worst case)

Ship Option C (`submissions/cd_lns_sa_cascade_stacked_periphery/placer.py`)
unchanged. M3-verified 1.0575, EPYC-projected 1.07-1.09. Rank 7-10
on the public leaderboard (Carrotato 0.967, vmallela 1.011).
