# Partial results — 2026-05-19 08:30

## E110Minimal-600s ibm06 result is shocking

Partial mapping (best guess by wall time and bench order):

| Bench (estimated) | Lane-4 default @3300s | E110Minimal @600s | Δ |
|---|---:|---:|---:|
| ibm01 | 0.85071 | 0.84702 | **−0.43%** |
| ibm03 | 0.93180 | 0.97362 | +4.49% |
| ibm06 | 1.11434 | **0.99162** | **−11.0%** !! |
| ibm02 | 1.01928 | 1.03648 | +1.69% |

**E110Minimal at 600s budget appears to BEAT Lane-4 (3300s) on ibm06 by 11%.**

If this pattern holds across more benches, it's revolutionary:
- 5× speedup AND quality improvement
- Suggests cascade is OVER-investing time on E25/E41 polish
- Means the optimal placer might just be E110 + targeted CD polish

## Working theory

The cascade gets stuck on hard benches (ibm06, ibm12, ibm14, etc) because:
- E25/E41 lanes start from SDF/DPO basins that may be poor
- Cascade saddle escape can't break out of bad basin
- Periphery polish marginal

E110 finds a fundamentally different (better) basin via gradient descent.
Long CD polish on E110 basin is enough — don't need E25/E41 lanes at all.

## What we need to verify (next 1-2 hr)

1. Map remaining E110Minimal-600s completions to bench names (proper match)
2. Wait for ibm17, ibm18 (hard benches) to see if minimal scales there
3. Confirm pattern: does E110Minimal beat Lane-4 broadly or just on ibm06?

## ovl10 / nocong --all: 1/17 complete, ~3 more hours

Inner cascade_stacked WINNER lines visible. Full Lane-4 (with E110 lane
+ periphery) takes ~50 min/bench. Expected completion ~12 PM.

## NG45 default Lane-4: 2/4 done (estimated)

- ariane133 or 136: stacked_periphery=0.65539, periphery rejected (ovl=6)
- ariane133 or 136: stacked_periphery=0.64737, periphery accepted

These are MUCH lower than Option C NG45 0.6893. If pattern holds, Lane-4
default may already lift NG45 by 5%+.

## Next actions (after noon)

If E110Minimal-600s avg < 1.05 → ship E110Minimal as primary submission
If ovl10 or nocong < 1.05 → ship that
If everything ties → ship Option C, but record E110Minimal as iteration win

Stage for next iteration:
- **Triple-E110-lane placer** (built, ready to fire) — addresses
  per-bench tuning concern by running 3 sweep-winning cfgs and
  plateau-picking
- **NG45 tests** on each winning cfg (~1.5 hr each at jobs=2)
- **Adaptive num_steps** placer (built)
