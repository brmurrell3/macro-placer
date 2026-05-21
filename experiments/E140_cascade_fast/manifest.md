---
id: E140
name: cascade_fast
status: in_progress
parent: E138
created: 2026-05-21
decided: null
champion_at_time: 0.984       # thinkorplace-v2 M3 --all
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E140: cascade_fast

## Hypothesis

v1's cascade saddle pipeline (E25 + cascading_saddle_escape) gave −1.88 %
over E48 on 17 IBM (cached uncapped 1.0612). v1's wall-safe production
descendant plateaued at 1.137 on EPYC because 96 % of CD wall is
single-threaded Python. v2's V4 FastDiffProxy + Gaussian density backbone
is 3-16x faster on GPU. Swapping the V4 + Gaussian descent for the slow
E25/E41 lanes — and keeping the cascade saddle middle stage with bounded
eigsh (E138 maxiter=50 tol=1e-2) — should give v1-quality cascade lift
inside the partcl 60-min/bench cap on EPYC.

## Method

Pipeline:
  1. V4 + Gaussian descent (`SmoothGlobalPlacerV4Gaussian.place()` →
     descend + legalize + project_overlaps) — fast GPU basin.
  2. CD polish 1 (300 s) — settle into local min before saddle.
  3. Bounded cascade saddle (E138 `bounded_cascading_saddle_escape`,
     max_iters=3, eigsh_maxiter=50, tol=1e-2, polish_budget=120 s,
     total saddle wall ~450 s).
  4. CD polish 2 (300 s) — finish on saddle output.

Budget 1500 s/bench. Subclass / reuse E138 `bounded_saddle.py` for
the eigsh-bounded saddle, do NOT mutate E84 source.

## Kill gate

Smoke ibm17 on EPYC at 1500 s budget:
  - FAIL if final proxy > thinkorplace-v2 EPYC ibm17 reference 1.18269
    by more than +1 % (i.e. > 1.195) — saddle is net harmful at this
    budget.
  - FAIL if total wall > 1800 s — budget overrun.
  - PASS if proxy ≤ 1.17 (−0.7 % lift) and zero overlaps.

## Generalization check

If smoke passes, run `--all` on EPYC to confirm 17-bench average lifts
vs v2 baseline 0.99115. Promote to submission only if --all avg
< 0.986 AND zero overlaps everywhere.

## Outcome (filled when decided)

Smoke ibm17 on AWS (A10G + 8-vCPU, contended w/ concurrent E138 run):
  - final_proxy = 1.19516, ovl=0, total_wall_placer = 1684 s
  - Phase walls: descent 179 s, CD1 309 s, saddle 695 s (overrun on
    441 s budget; 1 iter), CD2 39 s (squeezed by saddle overrun)
  - Phase proxies: descent 1.28101 -> CD1 1.20064 -> saddle 1.19444
    (accepted, -0.516 %) -> CD2 1.19516 (+0.0007 regression)
  - vs v2-extCD EPYC ibm17 ref 1.18269: +1.05 % WORSE (gate was +1 %)
  - Bounded eigsh found a soft mode (lambda_min = -0.507); +1.0 eps=0.3
    perturbation was the only one tested before iter 1 ran out of time
  - Single saddle iter wall = 504 s on contended CPU; expected ~250-300 s
    solo. Run-isolated smoke might land at proxy ~1.185 with CD2 getting
    its full 300 s after a faster saddle.
  - Status: MARGINAL kill-gate. Single contended smoke is noisy enough
    that this is NOT a definitive falsification.

Recommend: re-run ibm17 SOLO on AWS first to remove contention noise;
if proxy < 1.185 and zero overlaps, then run --all. If solo smoke
still > 1.195 (or saddle still cannot complete 2 iters), kill E140
in favor of v2 thinkorplace-v2 (already a verified submission).

## Pointers

- Code: `code/placer.py` (E140CascadeFastPlacer wrapper).
- Reuses `experiments/E138_bounded_saddle/code/bounded_saddle.py` for
  the bounded eigsh saddle escape (DO NOT mutate).
- Reuses `experiments/E127_v4_gaussian/code/smooth_global_placer_v4_gaussian.py`
  for the V4 + Gaussian descent (DO NOT mutate).
- Results: `results/smoke_ibm17.json`.
- Discussion: TBD.
