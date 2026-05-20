# Morning analysis — 2026-05-19 08:00

## Key findings from overnight

### 1. Default Lane-4 = TIE with Option C

| Variant | --all avg | Δ vs Option C 1.0575 |
|---|---:|---:|
| Option C (yesterday baseline) | 1.0575 | — |
| Variant A (Lane-4 add-on, default cfg) | 1.05737 | −0.013% TIE |
| Variant B (Plateau-pick, default cfg) | 1.06040 | +0.27% loss |

**Conclusion:** Default E110 cfg doesn't beat full cascade. The gradient lane
sweep showed E110+CD wins vs SDF+CD, but cascade already saturates that gain.

### 2. Sweep found 3 cfgs that lift --fast by 3.9-4.4%

| cfg | Δ on --fast | wins |
|---|---:|---|
| `lr=5e-3, steps=500, ovl_end=10` | **−4.39%** | 4/4 |
| `lr=2e-3, steps=1500, ovl_end=50` | −3.98% | 4/4 |
| `no_cong, lr=3e-3, steps=500, ovl=30` | −3.92% | 4/4 |

These compared E110+CD60s vs SDF+CD60s. **Key question:** does this lift
survive when CD has 50 minutes (cascade-class budget)? Running now.

### 3. Profile of cascade pipeline (ibm01, 600s budget)

```
place(): 469s total
├── run_cd_adaptive: 396s (84%)
│   ├── search_axis: 372s
│   └── delta_cost_axis_batch: 346s  ← HOT PATH
│       └── _net_cong_contrib_flat: 156s  ← BIGGEST HOTSPOT
├── cascade_saddle (uses CD internally): 164s (35%)
├── periphery_push_and_polish (uses CD): 165s (35%)
├── run_sa_polish_v2: 23s
└── run_lns_gridbin: 23s
```

**94% of time is single-threaded Python CD polish.** PyTorch ops are dwarfed.

This explains the speed gap to Carrotato:
- Carrotato: GPU Xplace + Triton kernels → no Python loop bottleneck → 3.8 min/bench
- Us: combinatorial CD in Python loops → 55 min/bench mostly waiting for CD

### 4. E110Minimal (E110 + CD only, no cascade) is SHOCKINGLY good

| Bench | E110Minimal @300s | Lane-4 default @3300s | Δ |
|---|---:|---:|---:|
| ibm01 | 0.85204 | 0.85071 | +0.16% |
| ibm04 | 1.00161 | 0.96007 | +4.32% |
| ibm09 | 0.82387 | 0.82548 | −0.20% |
| ibm13 | 0.96575 | 0.93687 | +3.08% |
| **avg** | **0.91082** | **0.89328** | **+1.96%** |

**E110Minimal at 300s achieves ~98% of cascade quality in ~9% of the wall.**
At 3300s budget, expected to close most of the 2% gap. Running now.

## What's running (CPU-overloaded but tolerable)

| Job | PID | jobs | Started | ETA |
|---|---|---|---|---|
| Variant A ovl10 --all | 98000 | 4 | 07:39 | ~12:00 |
| Variant A nocong --all | 98013 | 4 | 07:39 | ~12:00 |
| NG45 default Lane-4 | 98383 | 2 | 07:44 | ~11:00 |
| E110Minimal --all (3300s) | 99737 | 2 | 08:04 | ~13:00 |

CPU oversubscribed (~18 cores requested on 14-core M3), but jobs share fairly.
Expected 1.2-1.5× slowdown vs uncontested.

## Decision tree at noon

```python
ovl10_all = avg from CDLNSSACascadeStackedPeripheryE110Ovl10Placer JSON
nocong_all = avg from CDLNSSACascadeStackedPeripheryE110NoCongPlacer JSON
minimal_all = avg from E110MinimalPlacer JSON
ng45_default = avg from default Lane-4 NG45 JSON
ng45_option_c = 0.6893 (current Option C NG45)

# Decision matrix
if min(ovl10_all, nocong_all) < 1.05 and ng45_default < 0.700:
    # Strong winner on IBM + holds on NG45
    pick winner; run NG45 on it; ship if NG45 < 0.690
elif minimal_all < 1.06:
    # Even minimal beats Option C — fast lane wins
    Ship E110Minimal; can run multi-seed in same budget tomorrow
elif ovl10/nocong tie with cascade BUT NG45 lifts:
    # Submission cfg is bench-specific. Stick with Option C
    Ship Option C
else:
    # E110 doesn't generalize beyond --fast — retreat
    Ship Option C
```

## What's still to test (afternoon queue)

1. **Variant B with ovl10 cfg** — does plateau-pick + cascade compound the ovl10 lift?
2. **Multi-seed ensemble E110Minimal** — same wall budget, 3 different random seeds, pick min.
   Hypothesis: variance-reduction via ensemble might match cascade quality.
3. **--diverse 8-bench sweep** — train on 8 benches incl. ibm17/18, not just --fast.
   Validates that sweep cfg isn't overfitting to easier benches.
4. **Adaptive num_steps** — `num_steps = max(300, num_macros)` so big benches get more descent.
   Tests if the ibm17 +2.95% LOSS is a step-count issue.
5. **Wirelength-only descent** — sanity check. If WL-only beats WL+D+C, the smooth proxy
   approximation is hurting us.

## On per-bench tuning concern (user)

The ovl10 cfg is a SINGLE GLOBAL cfg (one hyperparameter changed from default).
It's tuned on --fast (4 benches), not per-bench. Risk: --fast is small sample;
might not represent full IBM or NG45.

**Validation strategy:**
- ovl10 --all (running): tests on 17 IBM benches
- NG45 on default Lane-4 (running): tests if ANY E110-based approach generalizes
- After noon: NG45 on best cfg

If ovl10 wins --fast (4) AND --all (17) AND NG45 (4), that's 25 benches — strong evidence
of generalization to hidden test cases (also 17 IBM-class + 4 NG45-class).

## On speed gap to Carrotato (user)

Three options to close the gap:

**Option A: Don't.** Use our 55 min/bench. Competition allows 60 min.
  - Pro: simpler, leverages our cascade machinery
  - Con: hard to iterate (~4 hr per --all)

**Option B: Optimize the Python hotspot.**
  - Cython/C ext for `_net_cong_contrib_flat` and `delta_cost_axis_batch`
  - Expected speedup: 5-10× on CD polish phase
  - Cost: 8-16 hours engineering. Risky for May 21 deadline.

**Option C: Skip cascade, use E110Minimal multi-seed.**
  - E110+CD at 4 min × 10 seeds = 40 min per bench
  - Pick min over seeds
  - Pro: Carrotato-class throughput
  - Con: quality TBD (running E110Minimal --all now)

Recommendation: keep Option A for submission, prototype Option C in parallel.
If E110Minimal --all at 3300s ties Option C, Option C is back on the table for
later iteration tomorrow.
