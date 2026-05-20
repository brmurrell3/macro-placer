# May 20-21 iteration plan (post-today's results)

## If we ship today (best case)

May 20 morning: EPYC validation on chosen placer
May 20 afternoon: confirm EPYC numbers + final submit
May 21: monitor leaderboard

## If still iterating (likely)

The Lane-4 architecture is solid but each cfg tweak gives <0.5% lift.
To break the 1.05 ceiling we need structural changes. Options ranked by EV:

### A. Better smooth congestion (highest EV)

Known limitation: E88 `_rudy_congestion` uses bbox-uniform routing demand,
not per-net trace. Memory says "Smooth _rudy_congestion bbox-uniform
diverges 3-4× from canonical per-net trace on ibm10/12/17". This is THE
source of smooth-to-canonical bias on hard benches.

**Fix:** Replace bbox-uniform with probabilistic per-net trace, matching
canonical's algorithm.

**Cost:** ~8 hr engineering. Test before May 21 ship.

**Expected lift:** 2-5% on hard benches (ibm12/14/17/18) which currently
drag avg up to 1.05. Could close the gap to <1.03.

### B. MPS backend fix (medium EV)

Currently fails at `_rudy_congestion` device mismatch. Fix is patching
h_coeff/v_coeff allocation to use placement.device. ~1 hr.

Expected: 5-10× faster Adam descent on M3 MPS. Enables multi-seed
ensemble in the same wall budget.

### C. Two-stage descent (medium EV)

Currently γ anneals linearly. Better: two-stage —
- Stage 1: γ=5e-3 to 5e-4, lr=5e-3, 300 steps (find basin)
- Stage 2: γ=5e-5, lr=1e-3, 200 steps (refine)

Mimics ePlace/Xplace's multi-stage approach. Probably +0.5-1% lift.

### D. LBFGS or Newton optimization

LBFGS often finds better basins than Adam on smooth nonconvex problems.
Quasi-Newton uses Hessian approximation = faster convergence.

Tricky in PyTorch but doable. ~3 hr engineering. Might or might not help.

### E. Stronger overlap penalty + spreading

Some macros end up in local minima with marginal overlap. A stronger
penalty during late-stage descent might force better separation.

Already tested (ovl_end=10 vs 50 in sweep). ovl=10 wins. Could try
ovl=5 or even 1 for late stages.

### F. Hierarchical refinement

Place soft macros after hard macros. Or place at coarse grid then refine
fine. Adds complexity but might give +0.5-1%.

## What NOT to do (already tried, falsified or insignificant)

- Per-bench tuning — wrong direction per user concern
- Plain Random init — catastrophic (1.7+ raw on ibm04)
- Pure SDF init — already used, can't improve much
- nocong — consistently loses against cascade
- More portfolio weights — within noise (sweep tested)
- Different RNG seeds (single-seed) — identical results

## Code work needed for ANY structural change

Test infrastructure ready:
- `experiments/E110_smooth_global_placer/code/sweep_harness.py` — config sweep
- `experiments/E110_smooth_global_placer/code/analyze_results.py` — result analysis
- `experiments/E110_smooth_global_placer/code/decide.sh` — auto-decision
- Lane-4 architecture (Variant A) is the integration point — drop in
  new SmoothGlobalPlacer variants and plateau-pick

## Conservative recommendation if everything ties today

Ship Option C unchanged. Trust the M3-verified 1.0575. EPYC validate
tomorrow. Submit by Wed noon.

Rationale: with 2 days to deadline, no benefit to risky cfg changes.
Option C is in the top 10 of public leaderboard already (rank 9 verified
at 1.0771 on judge box).
