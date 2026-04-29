---
id: E13
name: batched_cd_eval
status: in_progress
parent: E12
created: 2026-04-29
decided: null
champion_at_time: 1.0990
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E13: batched_cd_eval

## Hypothesis

The CD inner sweep at `submissions/cd_lns_gridbin/placer.py:153-181` evaluates
each (col, row) candidate sequentially via `evaluator.move/current_cost/revert`.
If the per-candidate cost can be batched (`[K, ...]`-shaped tensors) and run on
MPS, total wall time for the K-candidate sweep drops enough to bring our
~30 min/benchmark down toward DreamPlace++'s ~37 s/benchmark — without changing
the algorithm or sacrificing the −1.63 % E12 lead.

## Method

Stage the cheapest possible kill test first. Stages only run if the prior one
passes:

**Stage 0 (architecture review, no code).** Confirm
`incremental_evaluator.current_cost()` is pure PyTorch (it is — see
`_wirelength_cost`, `_density_cost`, `_congestion_cost` at lines 650–749).
Confirm `move()` is Python-scalar-heavy (it is — dict updates, `.item()`
calls per affected net at lines 764–867). **Done.** This is what frames the
remaining stages.

**Stage 1 (microbench, ~100 LOC).** On the M3 Max, time on **ibm01** only:
- (a) 200 × `evaluator.move(idx, xy); evaluator.current_cost(); evaluator.revert()`
  on CPU — the current production path.
- (b) 200 × `current_cost()`-only on CPU (skip move/revert) — the kernel-only
  baseline.
- (c) Same as (b) but with the evaluator's tensors moved to MPS — the
  per-candidate MPS ceiling.

This costs us a single ~1 min run. Kill gate fires here.

**Stage 2 (only if Stage 1 passes).** Add a batched cost path:
`current_cost_batch(candidate_xys: [K, 2]) -> [K]` that computes proxy for K
candidate positions of one macro in one tensor op. Time on MPS for K = 64,
256, 1024. If batched MPS at K=256 doesn't beat CPU sequential by ≥5×, kill.

**Stage 3.** Integrate into the LNS reinsert loop only (the simpler call site
than CD's per-macro sweep). Run `--fast`. Compare wall-clock and quality.

## Kill gate

Stage 1: kill if **MPS `current_cost()` per call is slower than CPU
`move+cost+revert` per call**. Rationale: even with perfect batching at
K = grid_col × grid_row ≈ 400, MPS ≥ 2× slower per kernel means batching
needs ≥800× amortization to break even, which exceeds plausible MPS speedup
on ibm-sized tensors. (Realistically a tighter gate of "MPS ≤ 2× CPU
incremental per call" is what we want, since batching has to absorb both
overhead and bookkeeping replacement.)

Stage 2: kill if batched MPS at K=256 doesn't beat CPU sequential by ≥5×.

Stage 3: kill if `--fast` wall time doesn't drop by ≥3× (need a meaningful
shift toward 37 s territory; smaller wins don't justify the rewrite).

## Generalization check

If Stage 3 lands, run `--all` and confirm avg proxy stays ≤ 1.0990 (no
quality regression from FP precision changes on MPS). NG45 only after
champion-tier on `--all`.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/microbench_stage1.py` (Stage 1 only — keeps experiment cheap).
- Background: `submissions/cd_lns_gridbin/placer.py:153-181` (inner sweep),
  `macro_place/incremental_evaluator.py:650-749` (cost kernels — already
  pure PyTorch), `:764-867` (move bookkeeping — Python-scalar).
- Trigger: Partcl LinkedIn post 2026-04-29 about DreamPlace++ (Billy Lee /
  MediaTek) showing 37 s/benchmark via GPU analytical placement.
