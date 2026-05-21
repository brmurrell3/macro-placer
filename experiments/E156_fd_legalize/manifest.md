---
id: E156
name: fd_legalize
status: in_progress
parent: E127  # V4+Gauss descent comes from E127
created: 2026-05-21
decided: null
champion_at_time: 0.984  # thinkorplace-v2 M3 --all
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E156: Force-directed (FD) macro legalizer

## Hypothesis

The V4+Gauss descent output has many overlapping macros that the post-
descent legalizer must resolve. Currently we use a deterministic greedy
spiral search (`greedy_macro_legalize` from E76). Greedy is by area-
descending; large macros get their original location, then smaller macros
spiral around them.

A force-directed (FD) legalizer is a fundamentally different mechanism:
all macros move simultaneously based on pairwise repulsive forces from
overlapping neighbours. The two algorithms can resolve the same overlap
graph into substantially different layouts:

- Greedy snaps each macro to the closest non-overlapping grid cell,
  potentially producing artifacts (small macros squeezed into corners).
- FD distributes overlap-resolution displacement smoothly across both
  macros in each pair, preserving more of the descent's spatial structure.

If FD produces a different post-legalize basin from greedy, downstream
CD polish has a different starting point and may converge to a different
local optimum.

## Method

Drop-in legalizer replacement in the thinkorplace-v2 pipeline:

  V4+Gauss descent (raw)
    -> fd_legalize (REPLACE greedy_macro_legalize)
    -> CD polish 900s

If FD leaves residual overlaps, fall back to project_overlaps +
greedy_macro_legalize as safety net.

Compare to thinkorplace-v2 final score on EPYC ibm17 (1.18269).

## Kill gate

EPYC ibm17 final proxy >= 1.18269 + 0.005 (i.e., worse than v2-extCD by
more than half a percent) -> falsified.

A useful auxiliary signal: pre-CD basin proxy of fd_legalize output vs
greedy output. If FD's basin is identical or strictly worse, the
hypothesis (that FD reaches a different basin) is incompletely supported.

## Generalization check

If ibm17 lifts >0.5 %, run 2-3 more EPYC benchmarks (ibm10, ibm12, ibm18)
before declaring an ensemble lane candidate. Single-bench wins below 1 %
are within hardware variance.

## Outcome (filled when decided)

### Pre-CD basin signal (informative)

**Local M3 ibm17, same V4+Gauss descent output (seed=42, ovl_lambda=10):**

| Legalizer | Wall (excl. proxy) | Pre-CD basin proxy |
|---|---:|---:|
| greedy_macro_legalize | <1s | 1.28676 |
| **fd_legalize (strength=2.0)** | <1s | **1.27969** (−0.55 %) |

FD basin is structurally different and 0.55 % lower than greedy basin pre-
CD. This validates the hypothesis "FD produces a different basin" but
does not yet decide whether CD polish converges to a better minimum.

**EPYC ibm17 smoke (rng=42, cuda descent → strength=2.0 FD):**
- Pre-legalize ovl_count: 78
- FD: ovl=0 in 4 iters, 0.16s, basin proxy 1.27501
- Final CD-polished proxy: (pending)

## Pointers

- Code: `code/fd_legalize.py`, `code/placer.py`
- Comparison baseline: `submissions/thinkorplace-v2/placer.py` 1.18269 EPYC ibm17
- Predecessor: E152 (in-process ensemble FAILED), E155 (parallel-subprocess ensemble, 1.17x on ibm17 likely)
