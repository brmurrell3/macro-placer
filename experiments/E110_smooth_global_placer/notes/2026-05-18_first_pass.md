# E110 first-pass results — 2026-05-18 evening

## TL;DR

**Adam descent on E95 DiffProxy → greedy_legalize → 60s CD lifts ibm01,
ibm09, ibm13 vs SDF + 60s CD by 1.4–4.1%.** Regression on ibm04 (+2.6%).
3/4 wins on --fast subset. Aggregate Δ = **−1.27%**.

This is the gradient-lane signal we were looking for. **Continue the
sprint.**

## --fast 4-bench result (M3, 60s CD)

| Bench | hard | canvas | SDF+CD60s | E110+CD60s | E110 raw | Δ |
|---|---:|---|---:|---:|---:|---:|
| ibm01 | 246 | 23×23 | 0.91760 | **0.87703** | 1.078 | **−4.05%** |
| ibm04 | 295 | 34×34 | 1.02482 | 1.05182 | 1.431 | +2.63% |
| ibm09 | 253 | 47×47 | 0.86808 | **0.85603** | 1.038 | **−1.39%** |
| ibm13 | 424 | 56×56 | 1.04287 | **1.01942** | 1.273 | **−2.25%** |
| **avg** | | | **0.96334** | **0.95108** | | **−1.27%** |

Per-bench wall: 90–150s. Total --fast: ~8 min. Compare cascade --fast
single-thread: ~3+ hr.

## What's working

- LSE-HPWL + grid density + ABU-5% RUDY smooth proxy converges via Adam
  from SDF init. Smooth basin reaches 0.73 (ibm01), 0.70 (ibm09).
- γ-anneal `5e-3·cw → 5e-5·cw` works (no instability).
- Overlap-penalty rampup `0 → 50` over first 70% keeps Adam zero-overlap
  during descent on most benches.
- `greedy_macro_legalize` with `step_size_frac=0.005` produces zero
  overlaps on all tested benches.
- CD recovers smooth-to-canonical bias.

## What's broken

- **ibm04 raw post-legalize = 1.43** (very bad). Greedy legalize is
  doing too much work — Adam likely settled in an overlapping config
  the rampup didn't fully resolve. Diagnose with longer steps and
  stronger overlap_lambda.
- **Smooth-to-canonical gap is large at convergence** (0.73 smooth →
  1.01 canonical on ibm01 post-legalize). The 2% docstring number was
  at cascade optimum, not Adam-only. Reality: gap is 20-50% before CD
  polish. CD closes most of it, but not all.

## Default config (winning on 3/4)

```python
SmoothGlobalPlacer(
    num_steps=500,
    lr_frac=0.005,                # Adam lr = 0.005 × canvas_width
    gamma_start_frac=5e-3,
    gamma_end_frac=5e-5,          # tighter anneal than initial 5e-4
    overlap_lambda_start=0.0,
    overlap_lambda_end=50.0,
    overlap_ramp_pct=0.7,
    boundary_lambda=50.0,
    include_congestion=True,
    init='sdf',
    legalize_step_frac=0.005,     # fine spiral step
    legalize_radius_steps=200,
)
```

## Next steps (priority order)

1. **ibm17 single-bench** — does largest IBM (537 macros) crash or run?
   [RUNNING]
2. **ibm04 variant sweep** — stronger overlap_lambda, more steps, random
   init. [RUNNING]
3. **E110 + full cascade polish** (not just 60s CD) — does the lift
   survive?
4. **Lane 4 integration into stacked_periphery** — plateau-pick over
   {E25, E41, E110}.
5. **--all run** — verify 17 IBM + 4 NG45.

## Lane 4 integration sketch

```python
# In stacked_periphery / cascade_stacked place():
init_lanes = []
for lane_name, lane_placer in [
    ('E25_SDF',  E25_placer),
    ('E41_DPO',  E41_placer),
    ('E110_grad', SmoothGlobalPlacer(...)),  # NEW
]:
    if deadline - time.time() < lane_min_budget:
        log(f'  SKIPPING lane {lane_name}: insufficient time')
        continue
    try:
        pos = lane_placer.place(benchmark)
        proxy = compute_proxy_cost(pos, benchmark, plc)['proxy_cost']
        init_lanes.append((lane_name, pos, proxy))
    except Exception as exc:
        log(f'  Lane {lane_name} FAILED: {exc}')
# Plateau pick
best = min(init_lanes, key=lambda x: x[2])
```

E110 lane_min_budget ≈ 200s (descent ~30s + CD ~60s + buffer).

## Risk notes

- ibm04 regression means E110 alone isn't safe — must be a LANE, not
  a replacement.
- 1/4 loss rate on --fast: probably worse on --all without tuning.
- Hardware variance to EPYC still unknown for E110.
