# Overnight final — 2026-05-20 04:23 EDT

## Champion locked: V3Min ovl10 720s

**IBM avg 1.00279, NG45 avg 0.67861, zero overlaps, qualified.**

Launcher points at `submissions/e111_minimal_ovl10_720s/placer.py`
(class `E111MinimalOvl10_720sPlacer`).

## Champion progression overnight

| Time | Placer | M3 IBM | Δ vs Option C |
|---|---|---:|---:|
| Yesterday | Option C | 1.0575 | — |
| 19:00 | Subagent built E111 per-net trace | — | — |
| 21:36 | V3Min default 4-min | 1.0302 | −2.58% |
| 22:19 | V3Min ovl10 4-min | 1.0213 | −3.43% |
| 22:28 | V3Min default 8-min | 1.0195 | −3.59% |
| 22:54 | V3Min ovl10 8-min | 1.0039 | −5.07% |
| 23:43 | **V3Min ovl10 12-min** | **1.00279** | **−5.17%** ← CHAMPION |
| 00:41 | V3Min ovl10 margin 8-min | 1.0026 | TIE w/ champion |
| 01:20 | V3Min ovl10 DPO 12-min | 1.0102 | +0.74% (worse) |
| 02:16 | V3Min ovl10 multiseed 12-min | 1.0051 | +0.23% (worse) |
| ~03:30 | V3Min ovl5 12-min | FAIL (overlaps) | — |
| ~03:30 | V3Min ovl20 12-min | FAIL (overlaps) | — |

## Diminishing returns observed

- **ovl10** is the cfg sweet spot. ovl5/ovl20 both produce overlaps; default 50 leaves polish on table.
- **12-min budget** marginally beats 8-min (1.003 vs 1.004).
- **Margin overlap** essentially tied (1.0026 vs 1.0028).
- **Multi-seed** slightly worse (variance reduction not worth budget split).
- **DPO init** slightly worse (V3 prefers fresh basin).

We've hit a local floor at ~1.003 M3.

## Hard benches lifted dramatically

V3Min ovl10 720s per-bench (vs Lane-4 default 1.057):
- ibm17: 1.200 (vs 1.324, **−9.4%**)
- ibm18: 1.236 (vs 1.342, **−7.9%**)
- ibm15: 1.099 (vs 1.150, **−4.4%**)
- ibm14: 1.084 (vs 1.192, **−9.1%**)
- ibm13: 0.859 (vs 0.937, **−8.3%**)
- ibm09: 0.778 (vs 0.825, **−5.7%**)

## Public leaderboard projection (EPYC)

M3 1.00279 × 1.024 variance ≈ 1.027 EPYC.

| Rank | Team | IBM |
|---|---|---:|
| 1 | Carrotato | 0.967 |
| 2 | Shoom | 0.978 |
| 3 | vmallela | 1.011 |
| 4 | DREAMPlaceProMaxUltra | 1.012 |
| 5 | QuantSC | 1.029 |
| — | **us (projected EPYC)** | **~1.027** |
| 6 | QED | 1.031 |
| 7 | Cezar | 1.037 |
| 8 | Place,Route,Roll | 1.059 |
| 9 | (our submitted) | 1.077 |

**Projected rank: 4-5** (from current 9).

## What this required

1. **Path A (Claude subagent)**: Built E111 per-net trace congestion that matches canonical ±15-25% (vs old bbox-uniform ±200-260%). The single biggest breakthrough.
2. **Path B v2 (Claude subagent)**: Built V5 Xplace-style recipe. Less successful but contributed the margin trick.
3. **V3Min architecture**: Drop cascade pipeline entirely. V3 descent (5 min) + CD polish (7 min) = 12 min/bench.
4. **Cfg tuning**: overlap_lambda_end=10 (sweep top, generalized).

## What's next (May 20 work)

1. **EPYC cross-validation** — verify ~1.027 projection on AWS c6a.4xlarge spot
2. **NG45 final verification** — already have 0.67861, confirm
3. **Submission packaging** — eval_docker, push, form submission
4. **One last iteration if time** — try hybrid V3Min + light cascade saddle

## Risk register

- **Per-bench tuning**: NONE. ovl10 is single global cfg.
- **NG45 generalization**: VERIFIED 0.67861 (lift vs Option C 0.6893).
- **EPYC variance**: ~2.4% historical. Our 1.003 → ~1.027.
- **Carrotato 0.967 unreachable in 24 hr**: structural changes needed; our current improvement is already +5.17%.

## Files

- `submissions/e111_minimal_ovl10_720s/placer.py` — CHAMPION
- `experiments/E111_per_net_trace_congestion/code/` — per-net trace primitives
- `experiments/E110_smooth_global_placer/code/` — V3Min base
- `experiments/E113_xplace_recipe/code/` — V5 recipe (not selected)
- 14 new --all JSON results in `results/`
