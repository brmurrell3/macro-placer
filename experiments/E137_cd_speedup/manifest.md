---
id: E137
name: cd_speedup
status: in_progress
parent: null
created: 2026-05-21
decided: null
champion_at_time: 0.984     # M3 thinkorplace-v2 --all baseline
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E137: cd_speedup

## Hypothesis
The CD polish lane in thinkorplace-v2 is wall-clock dominated by the
single-threaded Python in `IncrementalProxyEvaluator.delta_cost_axis_batch`
(cProfile on ibm04 attributes 32s/33s to `move()`/`revert()`). The prior
agent measured a 30 % per-batch wall reduction by replacing
list.append / list.extend accumulators with preallocated numpy scatter
buffers and adding a small-net fast path for `_net_cong_contrib`. The
v2 extended CD budget gave −0.73 % lift on EPYC by letting CD run longer
before plateau; pairing that with a 30 % faster per-batch wall should
let CD complete more accepted sweeps within the same 1500s budget,
plausibly stacking another lift on hard benches (ibm17 in particular).

## Method
Re-apply the verified bit-exact CD speedup at
`macro_place/incremental_evaluator.py`:
* Add preallocated numpy scatter buffers (density, H_macro, V_macro,
  H_net, V_net) with geometric-growth allocator.
* Replace list.append / list.extend in `delta_cost_axis_batch` with
  direct ndarray writes via a reservation API.
* Replace per-call `torch.tensor([...], dtype=...)` in `_scatter_add`
  with zero-copy `torch.from_numpy(buf[:off])`.
* Add `_net_cong_contrib_to_buf` fast path for nets with ≤4 pins
  (the small-net common case): cached numpy view of `pin_x` / `pin_y`,
  pure-python gcell computation, skipping `torch.floor / clamp / long
  / tolist`.

Public API (`__init__`, `move`, `revert`, `commit`, `current_cost`,
`delta_cost`, `delta_cost_axis_batch`) is preserved exactly. Parity
test (`test/test_incremental_evaluator.py`) must pass.

Test placer is a copy of `submissions/thinkorplace-v2/placer.py` placed
at `experiments/E137_cd_speedup/code/placer.py` — identical knobs and
imports; the speedup lives in `macro_place/incremental_evaluator.py`
which both placers share.

## Kill gate
EPYC `ibm17` final proxy ≥ 1.18269 (the extended-CD-only baseline) —
i.e. no measurable improvement over what the v2 budget bump already
delivered. Speedup re-implementation that fails parity is an automatic
kill.

## Generalization check
If ibm17 smoke is positive, run `--all` on EPYC and compare combined
avg against v2's 0.98387 baseline. Lift ≥ 0.3 % → graduate (ship
combined config in next v2 spin); within ±0.3 % → marginal (keep the
speedup in tree, no champion change).

## Outcome (filled when decided)
TBD.

## Pointers
- Code: `code/placer.py` (= a copy of `submissions/thinkorplace-v2/placer.py`).
- Speedup: `macro_place/incremental_evaluator.py` (`_ScatterBuffer`,
  `_net_cong_contrib_to_buf`, rewritten `delta_cost_axis_batch`).
- Tests: `test/test_incremental_evaluator.py` (must pass).
