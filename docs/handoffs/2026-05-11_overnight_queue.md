# Overnight autonomous queue

For Claude self-tracking. Last updated 2026-05-11 20:05 PDT.

## Active probes (cloud, ETA ~21:00)

5 cascade variants × 4 benches (ibm10/12/14/17) = 20 data points:

| Variant | Hypothesis | Started |
|---------|-----------|---------|
| wide-saddle | k=1, 6 eps {0.1, 0.3, 1, 3, 10, 30}, 60s polish, max_iter=3 | 19:55 |
| more-cascade | Cascade gets 40% budget (vs 33%), E25/E41 trimmed | 19:56 |
| dual-basin | Cascade from BOTH E25 and E41, take best | 20:00 |
| aggressive-kjoint | K-joint K=8 top_N=15, budget 300s (vs default K=3 150s) | 20:05 |
| finegrain | tight eps {0.05, 0.1, 0.3, 0.7}, 300s polish, max_iter=2 | 20:05 |

## Autonomous decision tree (at each gate)

```
WAKEUP every 60-240 min:
  1. Aggregate all completed probes since last wake → update LEADERBOARD.md
  2. IF any variant wins ≥3/4 hardest benches by ≥0.5%:
       → Launch winner --all (17 IBM) on cloud
       → Schedule next wakeup +240 min
  3. ELSE IF probes still running:
       → Schedule next wakeup +60 min, no action
  4. ELSE (all probes done, no clear winner):
       → Launch dual-basin --all (most algorithmically novel)
       → Queue extended-budget cascade for after
       → Schedule next wakeup +240 min
  5. AFTER --all done:
       → Launch winner --ng45 (4 designs, ~1hr)
       → Then extended-budget cascade --all (b=5400, ~5hr)
       → Schedule next wakeup +240 min
  6. KEEP loop alive: every wakeup must schedule its successor
```

## Queue for after --all wave (if time)

Variants worth probing if cloud frees up before user wake:
- `gamma_perturb`: smooth-proxy gamma=0.001 (vs 0.0005) for sharper Hessian
- `restart_on_plateau`: cascade restarts with random ε kick after no-improve
- `e48_only`: maximum E25+E41 budget, no cascade (baseline of plateau-only)
- `cascade_from_sdf`: skip E25/E41 entirely, cascade from SDF init
- `extended_budget`: cascade b=5400 (overshoot 60-min cap, EPYC ceiling test)

## What NOT to spend time on

Already falsified in LEADERBOARD.md:
- DREAMPlace anything (path B killed, see DP sweep)
- cascade max_iters > 5 (budget saturates)
- cascade --jobs 4 (contention not the bottleneck)
- numpy-grids patch alone (only 1.04× — need bigger refactor)

## How to recover if autonomous loop breaks

If the wakeup chain stops:
1. Cloud probes finish silently — results in `~/parallel_eval/<label>_probe/`
2. Re-launch via TaskCreate or manual ScheduleWakeup
3. LEADERBOARD.md and OVERNIGHT_QUEUE.md are the source of truth

## Commit cadence

Every gate decision should commit + push:
- Updated LEADERBOARD.md
- Any new placer variant or driver script
- Brief explanation in commit message
