---
id: E15
name: pair_swap
status: falsified
parent: ADR-003
created: 2026-04-27
decided: 2026-04-28
champion_at_time: 1.1055
outcome: 0.9414 (--fast); flat vs E16 baseline 0.9425 (Δ=-0.0011, below noise)
champion_delta: null
graduated_to: null
superseded_by: null
---

# E15: pair_swap

## Hypothesis
CD plateaus when every macro is at its single-macro fixed point. Swapping
two macros that share nets is a *coordinated* move outside CD's reachable
set — it can move a pair through a region of the loss landscape that no
single-macro shift can. Per ADR-003 the champion is plateau-bound rather
than budget-bound, so a different move type is the relevant lever.

## Method
After CDAdaptive converges, generate candidate pairs from the connectivity
graph (pairs sharing ≥ `min_shared_nets` nets). Rank by
`shared_net_count × center-to-center distance`, cap to `K = 5 %` of the
candidate set. For each candidate pair, query proxy with positions swapped
via the incremental evaluator; accept if improving. All hyperparameters are
global.

## Kill gate
Zero accepted swaps on `--fast` ⇒ CD's fixed point is stable to swaps too
⇒ kill.

## Generalization check
If `--fast` shows non-trivial accepted swaps, validate on NG45 ariane133.

## Outcome (filled when decided)
**Falsified 2026-04-28.** v2 of the pair-swap mechanism runs cleanly and
finds real candidates (21–53 accepted swaps per benchmark on `--fast`) but
the proxy delta is below noise.

- **v1 was buggy.** The candidate filter used `min_shared_nets = 2`, but
  per `findings.md` §"Algorithmic findings" #7, *most macro pairs share
  exactly 1 net* in the IBM benchmarks. v1 found zero candidates and
  produced output identical to the E16 baseline. Result file
  `results/CDPairSwapPlacer_20260428_091950.json` should NOT be cited.
- **v2 dropped the filter to `min_shared_nets = 1`.** On `--fast` v2 finds
  21–53 real swaps per benchmark and lands at avg **0.9414** — essentially
  flat versus the E16 baseline (`--fast` avg 0.9425 from
  `CDAdaptiveE16Placer` over the same set; Δ = −0.0011, below run-to-run
  noise). Result file `results/CDPairSwapPlacer_20260428_125855.json`.

The literal kill gate (zero accepted swaps) is *not* triggered, but the
gate's intent — "if pair-swap is the move type that escapes CD's per-axis
fixed point, the score moves" — clearly is. CD's plateau is robust to pair
swaps the same way it was robust to single-axis destroy/reinsert and SDF
jitter: the swap candidates that look promising on the connectivity graph
turn out not to be the ones the proxy actually wants moved. **Status
flipped to `falsified`.** Kept on disk as evidence for the writeup's
"different move type, but still inside CD's reach" lesson.

## Pointers
- Code: `code/cd_pair_swap.py`.
- Results: `results/CDPairSwapPlacer_20260428_125855.json` (v2, `--fast`,
  cite this); `results/CDPairSwapPlacer_20260428_091950.json` (**v1
  broken — do not cite**).
- Discussion: `findings.md` "Algorithmic findings" #7; `experiments_overnight.md`
  E15 block.
- Parent: ADR-003 ("CD plateau-bound, not budget-bound").
