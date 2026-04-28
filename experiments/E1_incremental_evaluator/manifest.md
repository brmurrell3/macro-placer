---
id: E1
name: incremental_evaluator
status: graduated
parent: null
created: 2026-04-26
decided: 2026-04-26
champion_at_time: 1.3834
outcome: infrastructure_speedup
champion_delta: null
graduated_to: macro_place/incremental_evaluator.py
superseded_by: null
---

# E1: incremental_evaluator

## Hypothesis
Full-proxy queries cost ~30 s on ibm10. CD requires per-move proxy lookups
in a tight inner loop; without a sub-millisecond delta evaluator, neither
CD nor LNS is feasible at competition scale. An incremental evaluator that
maintains per-net min/max trackers, a bin-density grid with delta updates,
per-net RUDY contributions, and vectorized cumsum smoothing should be both
*orders of magnitude faster* and *bit-for-bit equivalent* to the reference
proxy.

## Method
~930 lines in `macro_place/incremental_evaluator.py`. Per-net min/max
trackers; bin-density grid with delta updates per macro move; per-net
RUDY contributions cached and updated on move; smoothing via vectorized
cumsum. Unit-tested for parity in `test/test_incremental_evaluator.py`.
Single-step `revert()` only (multi-move sequences must invert manually).

## Kill gate
Bit-for-bit parity required. Any divergence > 1e-9 relative on random
moves invalidates downstream CD results.

## Generalization check
Tested on ibm01 + ibm10 with 130 random moves; benchmarked end-to-end via
`scripts/bench_incremental.py`.

## Outcome (filled when decided)
**Graduated to `macro_place/incremental_evaluator.py`.**

- **4 657 × speedup** on ibm10 (6.5 ms / move vs 30 s full eval).
- **Bit-for-bit parity** with `compute_proxy_cost` (worst absolute diff
  1.1 × 10⁻¹⁵ on 130 random moves; ibm01 + ibm10).
- Single-step revert tested within 1 e-9 relative.
- RUDY congestion verified decomposable per single-macro move; smoothing
  is the dominant per-call cost, `move()` itself is much cheaper.
- Load-bearing for E2 (ibm10 breakthrough), E9 (CDAdaptive champion),
  E12 (grid-bin LNS), E15 (pair swap).

## Pointers
- Code: graduated to `macro_place/incremental_evaluator.py`.
- Tests: `test/test_incremental_evaluator.py`, `scripts/bench_incremental.py`.
- Discussion: `writeup/evidence.md` §7.1; `docs/experiment_index.md` (E1 row).
