# 2026-05-19 — WireMask Greedy Reposition Breakthrough

## TL;DR

**WireMask greedy 2D reposition** (E113), added as a polish layer after
the cascade-portfolio saddle stack, gives **−0.26% avg lift** on `--fast`
at budget=1500s (Intel Xeon EPYC-like). Two wins, two ties, zero
regressions.

| Bench | Option C | Option C+WM | Δ |
|---|---|---|---|
| ibm01 | 0.88448 | 0.87824 | **−0.71 %** ✓ |
| ibm04 | 0.98665 | 0.98312 | **−0.36 %** ✓ |
| ibm09 | 0.83080 | 0.83098 | +0.02 % (noise) |
| ibm13 | 0.96739 | 0.96734 | −0.01 % (tie) |
| **avg** | **0.91733** | **0.91492** | **−0.26 %** |

The launcher `placer.py` now points to
`submissions/cd_lns_sa_cascade_stacked_wiremask_periphery/`. Validation
of `--all` at budget=1500 in progress (task `b9ff14cg7`).

## What WireMask does

For each movable macro `m`, in order of descending current WL
contribution (criticality-ranked):

1. Generate an N×N grid of candidate (x, y) positions centered on the
   current location, bounded by canvas + local-radius fraction (default
   N=5, radius=15% of canvas diagonal).
2. For each candidate, compute proxy delta via `IPE.delta_cost(m, xy)`.
3. Pick the lowest-delta candidate.
4. Commit if (a) zero overlap created, (b) proxy strictly improved.

Multiple passes until plateau or budget exhausted. `top_k` truncates to
the top-K critical macros for speed (default 300).

## Why WireMask wins where SA-v2 saturates

**Mechanism**: 2D atomic moves unlock placements that SA-v2's per-axis
breakpoint search can't reach. Specifically: if macro A's optimal new
position is at (x', y') and the intermediate states (x', y) or (x, y')
overlap with other macros, SA-v2 rejects the partial moves. WireMask
proposes the (x', y') position atomically; the overlap check happens
*after* the proposed move, with proper rollback if rejected.

**Empirical confirmation (E113 head-to-head, post-CD plateau on ibm01,
180s budget)**:

| Method | Final proxy | Δ from plateau (0.94755) |
|---|---|---|
| SA-v2 alone (180s) | 0.93587 | −1.23% |
| WireMask alone (180s) | 0.93646 | −1.17% |
| **SA-v2 90s + WM 90s** | **0.93044** | **−1.81%** ✓ |

WireMask alone is *slightly weaker* than SA-v2 alone, but **stacked
after SA finds additional non-local moves** SA can't reach. WireMask's
contribution per second is 2.3× SA's at the polished plateau.

On ibm04 (head-to-head), SA-v2 alone beat WM-only and SA-then-WM
within polish-only test. But in the FULL pipeline integration at
budget=1500s, WM still added −0.36% lift on ibm04. Mechanism: cascade
and portfolio saddle phases produce a DIFFERENT polished state than
just post-SA, and WireMask finds additional lift on that state.

## What was falsified (May 17–19 push)

Six breakthrough hypotheses falsified before WireMask landed:

| Hypothesis | Result |
|---|---|
| LP-dual congestion gradient | +0.087% / −0.24% / −0.05% across 3 configs (noise) |
| Population annealing (Hukushima-Iba) | −3.7% loss (resample overhead) |
| Multi-start SA-v2 (N=4 chains) | 0.17% inter-chain spread, no benefit |
| Extended single-chain SA polish | −0.07% on --fast (cascade compression > polish recovery) |
| LSMC structural kicks (K=4 cycle) | −0.29% on --fast (3/4 benches regress) |
| ε-magnitude vmallela hypothesis | ε=1.3 (lit) vs ε=3.0 (current) differ 0.03% — within noise |

Key insight from falsifications: **SA-v2 with best-so-far tracking is
already near-optimal at the local polish stage**. Replacement strategies
fail because they lose ground to the polished SA local minimum. Only
*additive* methods that find moves SA can't make have a chance — WireMask
is that method.

## Architecture (the new champion candidate)

```
SDF init    DPO init
   |           |
   v           v
  CD-LNS-SA   CD-LNS-SA-Kjoint           ← E25/E41 lanes
       \      /
        v   v
       plateau pick                       ← best of E25/E41
           |
           v
   cascade saddle (canonical Hessian)    ← max_iters=5, ε∈(0.3, 1.0, 3.0)
           |
           v
   portfolio saddle (3 non-canonical)    ← cong-focus, density-focus, non-WL
           |
           v
   *** WireMask polish (NEW) ***         ← top_k=300, n_per_axis=5, max 8 passes
           |
           v
   periphery wrapper (conservative)
           |
           v
   final placement (zero overlaps)
```

Budget split at submission default (3300s/bench):
- Stacked phase: 2970s
- WireMask polish: 180s
- Periphery polish: 120s
- Slack: 30s

At smaller budgets, stacked phase compresses; WireMask budget stays
fixed at 180s (significant fraction). Conservative wrappers ensure no
regression: WireMask is REJECTED if it doesn't strictly improve proxy
while keeping zero overlaps.

## Files

- `submissions/cd_lns_sa_cascade_stacked_wiremask_periphery/placer.py` — new champion candidate
- `experiments/E113_wiremask_greedy/code/wiremask_polish.py` — core algorithm
- `experiments/E113_wiremask_greedy/code/test_wiremask.py` — polish-only head-to-head harness
- `experiments/E113_wiremask_greedy/code/direct_test.py` — full-pipeline test harness with budget override
- `placer.py` — launcher updated to point at WM-integrated placer; respects `PLACER_BUDGET` env var
- This handoff: `docs/handoffs/2026-05-19_wiremask_breakthrough.md`

## Next steps

1. **`--all` validation** at budget=1500 (in progress, task `b9ff14cg7`).
   If avg ≥ −0.2% lift vs Option C baseline at same budget, ship.
2. **Option C baseline `--all`** at budget=1500 (next, ~1.8 hr wall).
3. **Full-budget `--all` validation** at budget=3300 (~3-4 hr wall) for
   final submission number.
4. **Optional stacking**: pair-swap polish (E114) after WireMask, if
   time permits.

## Sources

- WireMask-BBO (Shi/Yu/Qian, NeurIPS 2023):
  https://arxiv.org/abs/2306.16844
- EGPlace (Deng et al., ICML 2025): criticality-aware mutation:
  https://proceedings.mlr.press/v267/deng25g.html
