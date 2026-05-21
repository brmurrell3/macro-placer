---
id: E149
name: outlier_perturb
status: falsified
parent: thinkorplace-v2
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.984   # v2-extCD M3 avg
outcome: 0.9084           # ibm04 smoke; tied with v2-extCD baseline 0.92 within noise
champion_delta: null
graduated_to: null
superseded_by: null
---

# E149: outlier_perturb

## Hypothesis
K-pin random restart (E142) was falsified because perturbing random macros
adds *undirected* noise — most random perturbations land on macros that
were already well-placed, so the post-perturbation polish has nowhere to
go but back. If instead we identify the macros that contribute the most
to the canonical cost (largest WL/density/congestion contributors) and
*only perturb those*, the perturbation has direction — it removes the
biggest offenders and lets CD re-place them in better spots. This is a
macro-level analog of a Newton step: select coordinates by gradient
magnitude, perturb those.

## Method
Pipeline = thinkorplace-v2 with an **outlier-targeted perturbation loop**
inserted between CD1 and CD2:

  1. V4+Gaussian descent → greedy legalize → project_overlaps.
  2. CD polish 1 (~400 s) to reach the local minimum.
  3. **Outlier-perturb loop** (3 iterations × 100 s each):
     - Compute per-macro cost contribution:
       - WL contrib: for each net containing macro, distance from macro
         center to that net's bbox center (proxy for HPWL pull).
       - Density contrib: density grid value at macro's cell.
       - Congestion contrib: H+V routing cong at macro's cell.
       - Score = wl_contrib + 0.5*density_contrib + 0.5*cong_contrib
         (matching canonical proxy_cost weights).
     - Pick top K=8 by score.
     - Reset those K to uniformly random valid positions inside a local
       neighborhood (±20% canvas radius around current position, clamped
       to legal bbox).
     - greedy_macro_legalize → project_overlaps.
     - CD polish (100 s).
     - Accept iff canonical proxy_cost strictly improves.
  4. Final CD polish 2 (~300 s).

Total wall budget 1500 s/bench. Within partcl 60-min cap.

## Kill gate
- Smoke (`ibm04` only): if final proxy >= v2 baseline + 1 % AND zero
  iterations accept → falsify (outlier scoring uninformative or
  perturbation too large to recover).
- `--all` (17 IBM): if avg proxy >= 0.985 (worse than v2-extCD on M3) →
  falsify.

## Generalization check
- If --all passes, run NG45 commercial subset to confirm no regression.

## Outcome (filled when decided)
**FALSIFIED on ibm04 smoke (2026-05-21).**

Smoke (`ibm04`, M3, 487 s wall, zero overlaps):
- Phase 1 descent: 1.06823 (16 s)
- Phase 2 CD1 plateau: 0.91224 (87 s — CD plateaued early, well under 400 s budget)
- Phase 3 outlier loop, 3/3 REJECTs:
  - iter 1: K=8 perturbed; pre_polish=1.15138 → polish to 0.95190 (not below 0.91224)
  - iter 2: K=8 perturbed; pre_polish=1.08845 → polish to 0.93590
  - iter 3: K=8 perturbed; pre_polish=1.11707 → polish to 0.95637
  - Score distribution was discriminating (top-5 = [1.72, 1.10, 0.97, 0.87, 0.85]),
    chose hard movable macros, no silent bug.
- Phase 4 CD2: 0.91224 → 0.90836 (54 s; CD found a few extra moves on the
  CD1-plateau state, NOT on a perturbed state).

Final 0.90836 vs v2-extCD ibm04 baseline ~0.92 (range 0.911-0.928): the
0.4-1.3% “improvement” over v2-extCD is **within M3 run-to-run noise
floor (~0.36-0.5%)**. CD1 had 400 s budget but plateaued in 87 s, so
the *real* contribution of E149 (vs simply giving v2 the same CD2 budget)
is essentially the CD2 round at the end. The outlier-perturb loop did not
contribute anything — 3/3 rejects with delta exactly 0.0.

**Why outlier perturbation didn't help:**
- Same failure mode as E142 random K-pin: perturbing K=8 outliers + their
  legalization neighbors disrupts ~30-50 macros, lifts proxy from 0.912
  to 1.09-1.15, then 100 s CD polish recovers to 0.93-0.96 (not under
  baseline). Polish budget per iter is too small for the perturbation
  magnitude.
- ALL 3 iters chose the same top-8 indices (best_pos didn't change after
  rejects, so re-scoring picks the same outliers each time). Without
  tabu or randomized selection from a larger pool, repeated iterations
  redo the same work.
- The score successfully ranked macros (1.72 > 1.10 > 0.97 ...) — the
  scoring method itself is sound. The bottleneck is the perturbation-
  polish recovery gap, not outlier identification.

**Verdict:** E149 ≡ E142 within noise. Outlier-directed selection does
not solve the polish-budget gap that doomed E142. NOT recommended for
`--all` — would burn ~3.5 hr of judge wall time for a tied result.

## Pointers
- Code: `code/placer.py`
- Results: `results/`
- Parent: `submissions/thinkorplace-v2/placer.py`
- Sibling: `experiments/E142_kpin_restart/` (falsified random-K-pin variant)
