---
id: E141
name: group_lns
status: in_progress
parent: thinkorplace-v2
created: 2026-05-21
decided: null
champion_at_time: 0.921 (v2 on ibm04 M3)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E141: group_lns

## Hypothesis

CD plateau is single-axis line-search saturated. Spatially-coordinated K-macro
moves (E12-style LNS, but cluster-targeted around a focal macro and K-nearest
neighbors) can escape the CD plateau because they explore a 2D move type CD
cannot reach. E12's grid-bin LNS won historically; we resurrect that primitive,
but cluster-targeted (K-NN around a random focal) rather than top-K-cost.

## Method

Pipeline = v2 (V4+Gauss descent → legalize → CD polish 300s) → **group LNS 200s**
→ CD polish 300s. Total ~1500s/bench. Group LNS iter:

  1. Pick a random focal macro from hard movables.
  2. Find K=6 nearest hard movables to it (Euclidean on current placement).
  3. For each of the 6 macros: try N=20 candidate positions on a grid around
     its current location (radius ~30% of canvas), evaluate canonical proxy
     delta via `IncrementalProxyEvaluator.move/revert`, commit best-improving
     position. Re-place macros one at a time, greedy best-first.

Different move type vs CD (multi-macro coordinated 2D) and vs grid-bin LNS
(local neighborhood vs full canvas grid). LNS budget 200s; ~5-15s per iter
gives ~15-30 iters.

## Kill gate

Smoke ibm04 final proxy ≥ 0.92 (v2-speedcd baseline 0.921 on ibm04 M3).
< 0.92 → run --all. ≥ 0.925 → falsify.

## Generalization check

If smoke wins, --all on M3. avg < 0.985 to surface; avg < 0.97 → STOP.

## Outcome (filled when decided)

**Smoke ibm04 M3 2026-05-21 (228 s wall):** final 0.91551, zero overlaps.
- Phases: descent 17 s → CD1 105 s (0.91701) → LNS 5 s (REJECT) → CD2 89 s (0.91551).
- LNS converged early (5x no-improve streak) after 19 iters at 0.9 s total
  inside-evaluator wall — `IncrementalProxyEvaluator.current_cost()` shows
  starting proxy 0.93359 vs canonical 0.91701 on the *same* CD1 placement.
  LNS internal Δ = −0.00440 was real but the canonical-eval gap meant the
  LNS output (0.91738 canonical) was REJECTed back to CD1's 0.91701.
- vs baselines: v2-speedcd ibm04 M3 0.92099 → E141 0.91551 = **−0.6 %**.
  v2-extcd smoke 0.93498 → E141 0.91551 = **−2.1 %**.
  Lift is mostly from the budget shape (CD1+CD2 = 600 s vs v2's 900 s
  monolithic) and the inner re-evaluator state, not the LNS phase (which
  rejected). Direct LNS contribution ≈ 0.

Decision: lift is within M3 run-to-run noise (~0.3-0.5 %) vs v2-speedcd
0.921 baseline, but clearly beats v2-extcd smoke 0.935. Worth running
--all to see if multi-bench picture is consistent or random-noise.
Status: in_progress pending --all.

## Pointers
- Code: `code/placer.py`
- Results: `results/`
- Discussion: pending
