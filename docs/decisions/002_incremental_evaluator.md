# ADR-002: Build an incremental evaluator with bit-for-bit parity to compute_proxy_cost

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** project owner

## Context

After ADR-001, the next question was how to optimize the joint composite
proxy. DPO answered "differentiable surrogate," and got to 1.3834. The
RUDY fidelity analysis then showed that DPO's congestion gradient is
structurally wrong in direction — top-5 % hotspot overlap with real
congestion is 10.9 % (Jaccard 0.057, near-random), and the congestion-
weight sweep is monotonically worse, falsifying the "scale the gradient"
fix.

The next move — coordinate descent on the *real* proxy — was infeasible
at face value. `compute_proxy_cost` takes ~30 s per call on ibm10. CD
needs thousands of move evaluations per sweep, and the per-bench budget
is on the order of minutes to hours. At 30 s/move, a 600 s budget yields
~2 sweeps total. Far too few to converge.

The only path was infrastructure: an evaluator that, given a single
macro move, recomputes the proxy in milliseconds while remaining
identical to the reference implementation.

## Decision

Build `macro_place/incremental_evaluator.py` (~930 lines) with
per-net min/max trackers (incremental HPWL), a bin-density grid with
delta updates, per-net RUDY congestion contributions, and smoothing via
vectorized cumsum. Verify bit-for-bit parity against
`compute_proxy_cost` before using it for any accept/reject decision:
worst absolute diff 1.1e-15 across 130 random moves on ibm01 + ibm10,
with revert tested within 1e-9 relative.

The parity check is the load-bearing part. An "approximately equal"
incremental evaluator would re-introduce exactly the kind of structural
gap that ADR-001 was a response to.

## Consequences

Positive: 4657x speedup on ibm10 (6.5 ms/move vs 30 s full eval) gates
everything downstream. CD on ibm10 went from infeasible to 13 sweeps in
the 2400 s E2 run, dropping proxy from 1.4112 (SDF init) to 1.0632
(-24.7 %). Same algorithm class (coordinate descent) — different
infrastructure. This is the clearest single demonstration of
"infrastructure unlocks algorithm" in the project.

Negative: the evaluator is ~930 lines of stateful, bug-prone code that
must stay in sync with `compute_proxy_cost`. Any future change to the
proxy definition requires re-validating parity. Smoothing is the
dominant per-call cost; further speedups likely require rewriting that
path.

Alternatives ruled out: continuing to optimize a differentiable
surrogate (DPO, ADR-001 keeps this open in principle, but the RUDY
fidelity analysis made it the wrong path); approximate incremental
evaluation (would re-create the structural-gap problem); larger compute
(M3 Max wall-clock budget is the bottleneck, not throughput).

## Evidence

- `writeup/evidence.md` §7.1 (incremental evaluator: 4657x speedup,
  bit-for-bit parity)
- `writeup/evidence.md` §7.2 (E2 ibm10 breakthrough run, 1.4112 to
  1.0632)
- `writeup/contributions.md` §8 (incremental proxy evaluator, 4657x)
- `writeup/contributions.md` §9 (full-proxy CD)
