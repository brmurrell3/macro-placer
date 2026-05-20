# Overnight plan — 2026-05-18 → 19 morning

## Background jobs running

| Job | Log | ETA | Expected output |
|---|---|---|---|
| handpicked sweep (7 cfgs × 4 benches) | `/tmp/e110_handpicked_sweep.log` | ~90 min | `results/handpicked_fast{.jsonl,_summary.json}` |
| ibm04_diag sweep (18 cfgs × 1 bench) | `/tmp/e110_ibm04_diag.log` | ~45 min | `results/ibm04_diag{.jsonl,_summary.json}` |
| Lane-4 placer --all (jobs=4) | `/tmp/e110_lane4_all_p4.log` | ~4 hr | `results/CDLNSSACascadeStackedPeripheryE110Placer_all.json` |

## Champion verdict criteria (Lane-4 --all)

- **WIN (promote)**: Avg --all ≤ 1.0525 M3 (Option C 1.0575 −0.5%) AND zero overlaps everywhere
- **TIE**: 1.053–1.062. Decide tomorrow based on per-bench pattern
- **LOSS**: >1.062 or any overlaps → Lane-4 default config doesn't work; pick best from sweep

## Tomorrow morning actions (in order)

1. **Read all 3 job logs** to check completion + crashes
2. **Read sweep summaries** to find best E110 config
3. **Read Lane-4 --all JSON** for per-bench breakdown
4. If sweep winner ≠ default config → **rebuild Lane-4 placer with winner cfg**, run --all
5. **Run NG45 4-bench** on Lane-4 placer (~1.5 hr)
6. **Cross-validate on EPYC** if time permits

## Key files for tomorrow morning

- `/tmp/e110_handpicked_sweep.log` — sweep progress (tail -50 first)
- `/tmp/e110_ibm04_diag.log` — ibm04 diagnostic (tail -50)
- `/tmp/e110_lane4_all_p4.log` — Lane-4 --all progress (look for ===)
- `experiments/E110_smooth_global_placer/results/handpicked_fast_summary.json` — top configs
- `experiments/E110_smooth_global_placer/results/ibm04_diag_summary.json` — ibm04 best
- `results/CDLNSSACascadeStackedPeripheryE110Placer_all.json` — Lane-4 --all result
- `experiments/E110_smooth_global_placer/code/check_overnight.py` — morning summary script

## If Lane-4 default --all is a clear win

→ Switch `submissions/cd_lns_sa_cascade_stacked_periphery_e110/placer.py` to the
chosen variant; cross-validate on AWS c6a.4xlarge EPYC; package for May 21
submission.

## If Lane-4 default --all is mixed or loses

→ Take best config from sweep; re-run --all overnight tomorrow.
→ Likely 5 hr → done by Wed evening.
→ Wed morning: hardware cross-validation on EPYC.
→ Wed evening: final --all + submission package.

## If everything fails

→ Ship current Option C (1.0575 M3 verified). E110 stays as research record.
