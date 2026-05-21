---
id: E155
name: parallel_ensemble
status: marginal
parent: E152
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.98387
outcome: ibm17 smoke proxy 1.17653 BEATS both lanes (vs v2-extCD 1.18269 = -0.52%, vs E138 1.17996 = -0.29%), BUT wall 3097s = ~52 min exceeds 35-min kill-gate AND Lane A was terminated before completing — only Lane B contributed (degenerate ensemble = E138 alone)
champion_delta: negative (-0.52% on ibm17 vs v2-extCD)
graduated_to: null
superseded_by: null
---

# E155: parallel_ensemble

## Hypothesis
E152 falsified because budget-splitting two sequential lanes destroyed each
lane's CD plateau-convergence (Lane A got 570s vs needed 822s). The offline
ensemble of v2-extCD and E138 still projects -0.33% lift (avg 0.98064 vs
0.98387) **provided each lane runs at its full standalone budget**. The fix
is to run the two lanes in PARALLEL (multiprocessing subprocesses) so each
gets the full ~1500s budget within the 60-min/bench partcl cap.

## Method
In-process two-lane ensemble in `code/placer.py` that fans out two
`multiprocessing.spawn` subprocesses, one per lane:
- Lane A = v2-extCD: V4+Gauss + legalize + CD polish 900s (rng_seed=42).
- Lane B = E138: V4+Gauss + legalize + CD polish 900s + bounded cascade
  saddle 120s + CD polish trailing (rng_seed=142).

Subprocesses receive `benchmark.name` and an output pickle path. Each
subprocess re-loads bench/plc via `find_benchmark_dir` + `load_benchmark_from_dir`
(clean torch context, no shared state). On the judges' 16-vCPU EPYC each
subprocess gets ~8 vCPU; GPU is shared but each lane's CD polish is
single-threaded Python so GPU contention is negligible.

Master joins both processes (wall = max of two lanes ~25 min + setup),
deserializes positions, picks canonical-best, returns tensor.

`spawn` start method ensures each subprocess gets a fresh CUDA context.
Pickling avoids shared CUDA tensor state.

## Kill gate
EPYC ibm17 smoke. Compare to v2-extCD ibm17 1.18269 and E138 ibm17 1.17996.
- Kill: final proxy >= 1.180 (no improvement over both lanes).
- Marginal: 1.176 <= proxy < 1.180.
- Promote to --all: proxy < 1.176 AND total wall < 35 min.

## Generalization check
If smoke passes, run `--all --json --hypothesis E155_full` on EPYC.
Promote only if --all avg < v2-extCD 0.98387 (offline projection: 0.98064).

## Outcome
**MARGINAL — proxy beats both lanes, but wall blows the cap and Lane A
was killed (degenerate ensemble) (2026-05-21).**

EPYC ibm17 smoke result:
- Lane A: KILLED at master deadline (1700s budget). Lane A reached descent
  basin proxy=1.28629 at wall=413s, started CD polish (budget=900s)
  but `run_cd_adaptive` ran longer than its hard_cap on EPYC's 4-vCPU
  contention regime. Master terminated lane0 at the wall, NO result file.
- Lane B: COMPLETED. Descent basin proxy=1.28580 at wall=509s, CD1
  polish ran 1320s (vs 900s budget — `_sweep_macros` overran the cap)
  reaching **1.18058** (already beats v2-extCD 1.18269 by -0.18%).
  Bounded saddle escape found lambda_min=-0.85 in 64s Lanczos, accepted
  perturbation lifting to **1.17930** (-0.00127). CD2 polish 360s ran
  to wall=3093s with final proxy=**1.17653**.
- Master join with timeout=None waited for Lane B beyond deadline.
- PICK=B, final=1.17653, total_wall=3097s.

Comparison to standalones:
- v2-extCD standalone ibm17 1.18269 -> E155 -0.52% (BETTER)
- E138 standalone ibm17 1.17996 -> E155 -0.29% (BETTER)
- Cause: extended Lane B CD1 budget got 1320s (vs E138 standalone 900s)
  on hard bench, hitting deeper plateau. NOT the ensemble — Lane B alone
  with extended CD budget would have produced this.

Kill-gate disposition:
- Proxy < 1.176 → PASS (1.17653)
- Wall < 35 min → FAIL (52 min)
- Promote → BLOCKED on wall.

Root cause for wall blowout:
- On EPYC 8-vCPU with 2-way parallel = 4 vCPU/lane. CD polish (single-
  threaded inner sweep) gets ~2 vCPU active utilization. Lane B CD1
  hard_cap=900s was overrun to 1320s because the cap check is between
  sweeps and a single ibm17 sweep takes ~400s on contention.
- ibm17 is the worst-case (largest 17 IBM bench by macro count). Other
  benches should fit a 35-40 min wall comfortably.

Why Lane A died but Lane B finished:
- Lane A's CD ran longer than the master deadline (1700s). Lane B's
  CD1 also ran past the same point, but the master's join switched
  to timeout=None for the second join, so Lane B was allowed to
  complete. Order-dependent behavior — fragile.

Decision: DO NOT SHIP as v3.

Reasons:
1. Wall (52 min) blows the 60-min/bench cap with thin margin and
   no headroom for hardware variance.
2. The result is NOT a genuine ensemble (Lane A was killed). The lift
   comes from Lane B running with overrun CD budget, not from
   per-bench ensemble selection.
3. The "extended CD budget" effect is interesting but should be
   tested as a single-lane variant (just bump v2-extCD's cd_polish_s
   higher) — that would be a cleaner test of the hypothesis.
4. Ship v2-extCD (verified 0.98387 --all, ~25 min/bench) as the
   safer Tier-1 submission.

Pointer to follow-up: if there is post-deadline appetite, run a single-lane
E156 with v2-extCD's cd_polish_s=1500s and budget_seconds=2000s on EPYC
ibm17. That's the cleaner ablation isolating the CD-overrun effect.

## Pointers
- Code: `code/placer.py` (master + subprocess entry).
- Parents:
  - v2-extCD (`submissions/thinkorplace-v2/placer.py`)
  - E138 (`experiments/E138_bounded_saddle/code/placer.py`)
  - E152 (`experiments/E152_ensemble/code/placer.py`) — falsified sequential version.
