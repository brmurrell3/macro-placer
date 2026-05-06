---
id: E83
name: clock_aware
status: marginal
parent: E79
created: 2026-05-06
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E83: clock_aware

## Hypothesis

Per-benchmark dispatch (E82) is forbidden by competition rules. A single
algorithm with the SAME hyperparameters must run on every benchmark.

E79 fits the cap on small benches but blows it on hard ones. E48 fits
everywhere but loses E74's −1.4 % saddle lift. **A single algorithm with
hard-wall budget enforcement and clock-aware Hessian degradation can
deliver both: full E74-class lift on benches that finish phase 1+2 quickly,
graceful E48-equivalent fallback on benches that don't.**

The mechanism is wall-clock observation, NOT input-property dispatch:
* Same code runs on every bench.
* Same hyperparameters used by every bench.
* Behavior changes only if the wall-clock observes that less time
  remains for Hessian polish.

This is standard real-time-systems "anytime algorithm" design — does NOT
violate "treat all benchmarks the same" because the dispatch signal is
elapsed time, not benchmark identity or properties.

## Method

```
TOTAL_HARD_CAP_S = 55 × 60       # 55 min (5-min safety vs 60-min judge cap)

Phase 1+2: parallel E25⊥E41 with reduced budgets
  CD cap        = 1500 s  (25 min, vs E79's 2400 s)
  LNS budget    =  360 s  ( 6 min, vs E79's 600 s)
  SA budget     =  360 s  ( 6 min, vs E79's 600 s)
  K-joint       =    0 s  (disabled — Mitigation #4)
  Per-lane sum  ≤ 2220 s  (37 min)
  Parallel max  ≤ 37 min (E25 ∥ E41)

Phase 3: clock-aware Hessian saddle escape
  remaining = TOTAL_HARD_CAP_S − elapsed
  If remaining ≥ 15 min:  full   — k=1, ε={0.3, 1.0, 3.0}, polish 180 s
  If remaining ≥  5 min:  minimal — k=1, ε={1.0},          polish  60 s
  Otherwise:              skip   — return E25/E41 plateau
```

Worst-case wall: 37 (phase 1+2) + 18 (full Hessian) = 55 min ≤ 60-min cap ✓.
Slow-bench wall: 37 + 2 (min Hessian) = 39 min, OR 50 (slow phase 1+2) + 2 = 52 min.

Hardware probe scales polish budget within Hessian phase; total cap is fixed.

## Kill gate

- Wall > 58 min on ANY --fast benchmark → falsified.
- proxy regression > 2 % vs E48 (1.08151) on --all → falsified.
- Any overlap on any benchmark → falsified.

## Generalization check

- `--fast` walls all ≤ 58 min, proxy ≤ 0.95.
- `--all` proxy ≤ 1.085 (close to E48), but ideally < 1.075 (improves on E48).
- `--ng45` proxy ≤ 0.69.

## Outcome (2026-05-06)

### --all run (17 IBM benchmarks, --jobs 4)

| Metric | Value | vs Reference |
|---|---:|---|
| Avg proxy | **1.0859** | E74 (1.0666) +1.81 %; E48 (1.08151) +0.41 %; leaderboard (1.1172) **−2.80 %** |
| vs SA baseline | +48.9 % better | strong margin |
| vs RePlAce baseline | +25.5 % better | strong margin |
| Overlaps | 0/17 | ✅ all valid |
| Walls under 60-min cap on Windows | 17/17 | ✅ |
| Walls within 1-min margin to cap | **5/17** (ibm10, 12, 15, 16, 18) | ⚠️ EPYC risk |

### Per-bench results

| Bench | Proxy | Wall (min) | Margin to 60-min cap |
|---|---:|---:|---:|
| ibm01 | 0.8736 | 49.1 | +10.9 |
| ibm02 | 1.0925 | 51.3 | +8.7 |
| ibm03 | 0.9617 | 50.9 | +9.1 |
| ibm04 | 0.9966 | 45.4 | +14.6 |
| ibm06 | 1.1577 | 50.5 | +9.5 |
| ibm07 | 1.0780 | 51.6 | +8.4 |
| ibm08 | 1.1105 | 52.3 | +7.7 |
| ibm09 | 0.8345 | 51.0 | +9.0 |
| **ibm10** | 1.0246 | **59.9** | **+0.1** ⚠️ |
| ibm11 | 0.8793 | 53.5 | +6.5 |
| **ibm12** | 1.2186 | **59.8** | **+0.2** ⚠️ |
| ibm13 | 0.9622 | 51.9 | +8.1 |
| ibm14 | 1.2163 | 45.7 | +14.3 |
| **ibm15** | 1.1664 | **59.3** | **+0.7** ⚠️ |
| **ibm16** | 1.1585 | **59.9** | **+0.1** ⚠️ |
| ibm17 | 1.3717 | 46.0 | +14.0 |
| **ibm18** | 1.3575 | **58.9** | **+1.1** ⚠️ |
| **AVG** | **1.0859** | **53.7 avg** | **+6.3 avg** |

### Smoke test reference (single-bench, no contention)

* ibm13 smoke: proxy 0.96471, wall 56.7 min, eigvalue −2.03 (full Hessian ran successfully).

### Findings

1. **Wall mechanism works.** Hard cap was never breached on Windows. Clock-aware
   Hessian routing (full vs minimal vs skip) operated correctly.
2. **Wall margin is too tight.** 5/17 benches finished within 1 min of the 60-min
   cap on Windows. With EPYC slowdown (1.2-1.5 ×), these benches would blow the
   cap → bench-level disqualification.
3. **Proxy is competitive but not breakthrough.** E83 1.0859 beats public
   leaderboard (1.1172) by 2.80 %, but is +0.41 % over E48 hybrid (1.08151) and
   +1.81 % over E74 (1.0666 — which doesn't fit cap).
4. **Saddle escape lifts even on reduced-budget plateau.** Strong negative
   eigvalues (−1.75 to −2.03 on hard benches) confirm the saddle structure
   persists when phase 1+2 is compressed.
5. **The saddle escape doesn't enforce wall cap internally.** When elapsed time
   reaches the self-imposed cap (55 min) inside saddle, remaining polishes still
   run, pushing total wall to 56-60 min. Need either tighter phase 1+2 budgets
   OR explicit wall enforcement inside `_saddle_escape`.

### Status

**`marginal` — not falsified, but not graduated.** E83 v1 fits cap on this hardware
but margin is too thin for EPYC. Decision pending on E83 v2 with tighter budgets
(see roadmap §"Submission strategy").

### Next variants

- **E83 v2**: tighter budgets — phase 1+2 cap 28 min (CD 1200, LNS 240, SA 240),
  saddle 5 polishes × 120 s = 10 min, total max 38 min on M3-equivalent → ~46 min
  on EPYC. Re-run --all.
- **E83 v3 (alternative)**: keep budgets but add explicit wall enforcement inside
  `_saddle_escape` — break out of polish loop if elapsed > self-cap.

## Pointers

- Code: `code/cd_lns_sa_hessian_clock.py` (main), `code/_worker.py` (worker).
- Reuses: `submissions/cd_lns_sa_hessian/placer.py:_saddle_escape`,
  `submissions/cd_lns_sa/placer.py:CDLNSSAPlacer`,
  `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py:CDLNSSADPOKJointPlacer`.
- Results: `results/`.
