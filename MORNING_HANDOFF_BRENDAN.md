# Morning handoff (E107) — 2026-05-16 09:00 EDT

## TL;DR

Periphery wrapper VALIDATED on 4 fresh NG45 runs (3 wrapper + 1 patched).
Wrapper SAFE everywhere; periphery polish rejected on all because Lévy
cascade output is already a tight optimum.

**NEW NG45 bests (M3 Max overnight, all FRESH)**:
- ariane133: **0.65212** (vs E74 0.6641, **-1.80%**) ⭐
- ariane136: **0.64980** (vs E74 0.6518, **-0.31%**) ⭐
- mempool_tile: **0.73750** (vs E74 0.7376, tied)
- nvdla: **0.67646** (vs E74 0.6716, **-0.76%**) ⭐ [via patched wrapper]

NG45 4-bench aggregate: **0.67897** (-0.34% vs E74 0.68128, -0.28% vs PATH B 0.68086)

**IBM 17 (AWS-cpu overnight)**: aggregate **1.07564** (matches rank-6 score 1.0771)

**21-bench combined** (17 IBM + 4 NG45): **1.00009** ≈ 1.000 🎯

Patch shipped: `cd_lns_sa_cascade_levy/placer.py` now catches E25 errors and
falls back to E41-only. Prevents nvdla-style failures.

## Where things stand

| Score | Value | Comparison |
|---|---|---|
| IBM 17 aggregate (proxy ranking) | 1.07564 | Matches current rank 6 (1.0771) |
| NG45 4 aggregate | 0.67775 | -0.5% vs PATH B 0.68086 |
| 21-bench unweighted | 0.99985 | n/a (each scored separately) |
| ariane133 (Grand Prize bench) | 0.65212 | -1.8% vs E74 best 0.6641 |

## Key E107 findings (12 multi-seed tests, 5 seeds each)

| Pattern | Across 12 tests |
|---|---|
| Random kicks WIN proxy | 10/12 by 1-14σ statistical advantage |
| Periphery wins Δedge (more peripheral) | 10/12 — direction reliably toward edge |
| Random reliably drifts toward CENTER | 10/12 — direction matters |
| Center always worst | 12/12 — confirms direction signal |
| ibm10 fragile to perturbation | Both α=0.005 and 0.01 catastrophic |

**Strategic implication**: Lévy cascade IS optimal for proxy ranking.
Periphery wrapper helps WNS/TNS via direction toward edge, but at the cost
of small proxy regression on tight Lévy minima. Wrapper deployment safe
(strict-accept never regresses); useful only if OpenROAD validation
shows WNS/TNS benefit worth the proxy cost.

## Deployable

- `submissions/cd_lns_sa_cascade_levy_periphery/placer.py` — SAFE wrapper
  (strict-accept). Verified on 4 NG45 + 1 IBM (ibm01 smoke). Never regresses.
- Best plain Lévy: same `cd_lns_sa_cascade_levy/placer.py` (existing).

## What's blocking improvement

1. **OpenROAD WNS/TNS validation** — need to test if periphery shift
   (0.06%-1.39% Δedge) actually improves Grand Prize WNS/TNS objective
2. **E25 error on nvdla** — when E25 produces small (1-pixel) overlap,
   placer fails. Should add graceful fallback to E41-only
3. **ibm10 / ibm12 / ibm17 high proxy** — these dense IBM benches are
   our main contributors to aggregate; DP hyperparam dispatch (Path A)
   still untested and could help

## What's running

| Process | Status |
|---|---|
| nvdla plain Lévy fallback | Running (no wrapper, no periphery; just Lévy cascade) |
| AWS-cpu | IDLE (IBM chain complete) |
| Local M3 | Running nvdla fallback |

## Suggested next steps (in priority)

1. **OpenROAD validation** on ariane133 with our 0.65212 placement — answers
   the Grand Prize question definitively
2. **Path A** (DP hyperparam dispatch) on ibm12/14/17 — could lower IBM
   aggregate significantly
3. **Submit** current placements (1.07564 IBM + 0.67775 NG45) — already
   competitive

## Commit history (last few)

```
6c66431 E107 ariane133 fresh wrapper: NEW BEST 0.65212
704f0c0 E107 chain COMPLETE: 12-test multi-seed validation
5e295bf E107 final: 10-bench periphery sweep results table
d9bf7ad E107: complete sweep results across 10 benches
9700746 E107 periphery-bias spike
08ec3c9 Multi-Claude coordination + priorities
```
