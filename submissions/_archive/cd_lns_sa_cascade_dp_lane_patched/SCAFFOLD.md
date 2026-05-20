# SCAFFOLD: cd_lns_sa_cascade_dp_lane_patched

Built 2026-05-19 in response to the "patch DREAMPlace loss" task.
Not promoted; surface for the next sweep / cloud validation only.

## What this is

A 5-lane placer that augments `cd_lns_sa_cascade_dp_lane` with two
**smooth-proxy gradient descent** lanes:

- Lane 4: DREAMPlace basin → SmoothGlobalPlacer (E110) descent → CD
- Lane 5: SDF basin → SmoothGlobalPlacer (E110) descent → CD

The intent is Option C of the patched-DREAMPlace investigation:
"DREAMPlace as init for our smooth-proxy gradient placer." The smooth
descent IS effectively a patched DREAMPlace — same Adam-on-positions
mechanism but with our exact challenge proxy as the loss, rather than
DP's `wl + density_weight * eDensity`.

Lane 5 is always-on and is the local-testable equivalent of Lane 4
(since DREAMPlace requires x86_64 .so and Docker, which doesn't work
on M3).

## Why not promote yet

1. **Not validated on --all.** Only tested on ibm01 single-bench.
2. **No DP testing locally.** DREAMPlace can't run on M3 so Lane 4
   has never been smoke-tested end-to-end. Test on cloud A100 box.
3. **Likely strictly worse than the current champion** when DP isn't
   available: `cd_lns_sa_cascade_stacked_periphery_e110` (the
   current champion 1.0575 IBM / 0.6744 NG45) already has the
   smooth-descent lane as Lane-4 of stacked_periphery, AND has the
   portfolio_saddle + periphery wrapper this placer doesn't.

## Architecture diff vs champion

| Element | This placer | Champion (e110-stacked-periphery) |
|---|---|---|
| Init lane: SDF + E25 polish | YES | YES |
| Init lane: DPO + E41 polish | YES | YES |
| Init lane: DP + stock polish | YES | NO |
| Init lane: DP + smooth descent | YES | NO |
| Init lane: SDF + smooth descent | YES | YES (= Lane 4) |
| Cascade saddle | YES | YES |
| Portfolio saddle | NO | YES |
| Periphery wrapper | NO | YES |

This placer is "DP-lane + smooth descent on top." If DP is unavailable
this is essentially "cascade + smooth descent" (no portfolio, no
periphery), strictly worse than the champion which is "cascade +
portfolio + periphery + smooth descent."

## When to revisit

- A cloud machine with working DREAMPlace install exists, AND
- We want to test if DP basin + smooth descent produces lift on
  benches where the current champion underperforms
  (ibm02 +2.99%, ariane136 +1.76%, ibm10 +1.28%, ibm18 +1.13%,
   ibm08 +1.00%, ibm16 +0.61%, nvdla +0.49% — see E91 SUMMARY)

## How to test

```bash
# Smoke test smooth-descent capability (no DP needed; works on M3)
uv run python experiments/E110_smooth_global_placer/code/smoke_ibm01_patched.py

# End-to-end smoke (no DP available; lane 4 will skip, lanes 1+2+5 active)
uv run python experiments/E110_smooth_global_placer/code/smoke_patched_end2end.py

# Full --fast (requires cloud A100 with DREAMPlace for Lane 4 contribution)
DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
  DP_DOCKER_IMAGE=dreamplace:custom DP_USE_GPU=0 \
  uv run evaluate submissions/cd_lns_sa_cascade_dp_lane_patched/placer.py \
    --fast --json --hypothesis E94_C
```
