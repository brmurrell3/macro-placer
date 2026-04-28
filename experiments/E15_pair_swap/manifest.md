---
id: E15
name: pair_swap
status: in_progress
parent: ADR-003
created: 2026-04-27
decided: null
champion_at_time: 1.1055
outcome: null
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
- **v1 was buggy.** The candidate filter used `min_shared_nets = 2`, but
  per `findings.md` §"Algorithmic findings" #7, *most macro pairs share
  exactly 1 net* in the IBM benchmarks. v1 found zero candidates and
  produced output identical to the E16 baseline. Result file
  `results/CDPairSwapPlacer_20260428_091950.json` should NOT be cited.
- **v2 dropped the filter to `min_shared_nets = 1`.** On `--fast` v2 finds
  21–53 real swaps per benchmark and lands at avg **0.9414** — essentially
  flat versus the E16 baseline (`--fast` avg 0.9425 from
  `CDAdaptiveE16Placer` over the same set). Result file
  `results/CDPairSwapPlacer_20260428_125855.json`.
- The kill gate (zero accepted swaps) is *not* triggered for v2, but the
  proxy delta is below noise. v2 is best read as **marginal / falsified**
  in spirit: real swaps exist but don't move score. The manifest is left
  at `in_progress` because the parent agent's status table still has E15
  v2 listed as "running"; flip to `falsified` once that table catches up.

## Pointers
- Code: `code/cd_pair_swap.py`.
- Results: `results/CDPairSwapPlacer_20260428_125855.json` (v2, `--fast`,
  cite this); `results/CDPairSwapPlacer_20260428_091950.json` (**v1
  broken — do not cite**).
- Discussion: `findings.md` "Algorithmic findings" #7; `experiments_overnight.md`
  E15 block.
- Parent: ADR-003 ("CD plateau-bound, not budget-bound").
