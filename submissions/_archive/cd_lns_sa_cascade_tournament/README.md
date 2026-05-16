# cd_lns_sa_cascade_tournament

**Status:** experimental — unvalidated on `--all`. Created 2026-05-14.

## What

Tournament placer: runs three placers in parallel subprocesses and returns
the placement with lowest canonical proxy (zero overlaps required).

Lanes:
1. `cd_lns_sa_cascade_dp_lane/placer.py` (PATH B #1 — single-DP hybrid; champion)
2. `cd_lns_sa_cascade_dual_levy/placer.py` (E97/E100 — heavy-tail saddle)
3. `cd_lns_sa_cascade/placer_adaptive.py` (shipped — canvas-area dispatch)

## Theoretical aggregate

Per-bench best-of among these 3 placers (from 2026-05-14 cloud runs):
- IBM avg: **1.0605** (vs shipped 1.078 → −1.6 %; vs DP-lane champion 1.0665 → −0.6 %)
- NG45 avg: **0.6779** (vs shipped 0.681 → −0.5 %)

## Wall budget

Per-lane budget defaults to 2700s (45 min). Lanes run in parallel → total
wall ≈ 45 min, within the 60-min cap.

Each lane uses ~4 cores (`OMP_NUM_THREADS=4`). Three lanes = ~12 cores. Fits
in 30-core EPYC.

## Risks

- Subprocess overhead per lane (~5 s)
- If DREAMPLACE_ROOT not set, dp_lane gracefully degrades but doesn't lose to adaptive.
- Lanes that fail (e.g., docker subprocess error) are excluded; tournament
  picks best surviving lane.
