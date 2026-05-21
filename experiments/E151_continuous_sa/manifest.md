---
id: E151
name: continuous_sa
status: falsified
parent: E139
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.984    # thinkorplace-v2 EPYC
outcome: ibm04 smoke 0.9130 (=CD1 0.91297). 0 strict best-improvements in 195k SA proposals; T0=0.01·proxy saturates Metropolis (85.8% accept-worse), cur drifts to 1.05. Best-tracking restored. CD2 reproduces CD1 within roundoff.
champion_delta: null
graduated_to: null
superseded_by: null
---

# E151: SA on continuous macro positions (not pair-swaps)

## Hypothesis
E139's SA-on-pair-swaps was falsified because CD's basin is swap-locked
(swapping any two macros worsens proxy by definition of the local
minimum CD achieved on the swap graph). But CD also doesn't consider
**small continuous moves** of a single macro to non-grid positions —
the per-axis CD only commits at the best-of-axis fixed point. A Gaussian
SA proposal `x += N(0, T·step_size); y += N(0, T·step_size)` explores
neighbourhood directions CD never sees, so post-CD SA-continuous should
have a fighting chance to escape.

## Method
Append a 300s SA-continuous phase to thinkorplace-v2 after the 900s CD
polish, plus a 200s CD polish phase if SA improved the proxy. Per
proposal: pick a random movable hard macro, propose a small Gaussian
displacement around its current position, peek cost via
`IncrementalProxyEvaluator.delta_cost()` (non-mutating), commit via
`move()` if accepted by Metropolis, check overlap, revert if illegal.

T0 = 0.01·proxy, T_end = 0.0001·proxy, step_size = 0.005·canvas_width.
Decay calibrated by pace window (same as E139).

## Kill gate
ibm04 smoke: if SA acceptance rate < 1% OR SA best ≥ CD1 baseline +
0.005, kill. If smoke passes, run --all on M3.

## Generalization check
--all M3 < 0.984 average required for graduation.

## Outcome (filled when decided)

**Falsified on ibm04 M3 smoke 2026-05-21 (780s wall):**

| stage | proxy | wall |
|------|------:|----:|
| V4+Gauss basin | 1.07411 | 17s |
| CD polish #1 | **0.91297** | 374s |
| SA-continuous best | 0.91297 (= init) | 290s |
| SA chain-final | 1.02892 (drifted +13%) | – |
| best_restored | True | – |
| CD polish #2 | 0.91297 | 95s |
| **eval-final** | **0.9130** | – |

195513 SA proposals at 812/s. Zero strict best-improvements; best stayed
pinned to CD1's init value the whole time (best_found_at_t=0.0s).
Acceptance rate climbed 59% → 85.8% as σ shrank but `cur` drifted to
1.05 — i.e. SA was a noisy random walk, not basin polish.

Diagnosis of why: at T0 = 0.01·proxy ≈ 0.01 and typical post-CD basin
Δ ≈ 1e-5..1e-4, `exp(-Δ/T) ≈ 0.99..0.999`, so Metropolis accepts
essentially every worse move. SA samples a hot-temperature ergodic
distribution far from the basin; cooling never gets us back. Also σ =
0.005·canvas (≈10 micron for ibm04) is comparable to macro half-width,
which explains 13-25% of proposals being illegal (`rejIlleg=25569` at
end).

Implication: continuous-SA on a post-CD basin needs T0 ≈ 1e-5 (not
1e-2) and σ ≈ 1e-4·canvas (not 5e-3). At those scales it would
likely just reproduce CD's local fixed point with extra wall cost.
The "explore directions CD doesn't see" hypothesis is real, but the
useful exploration radius is below CD's per-axis grid resolution and
cannot escape the CD basin without a structurally different move type
(e.g. cooperative multi-macro shifts).

Not graduating; final 0.9130 is bit-equivalent to running v2 alone with
a slightly different CD budget split. Kill gate triggered (SA best ==
CD1 baseline, no improvement at all).

## Pointers
- Code: `code/placer.py`, `code/sa_continuous.py`
- Template: `experiments/E139_sa_swap/code/`
