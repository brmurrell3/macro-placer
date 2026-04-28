# ADR-006: Bypass approximations rather than improving them

**Status:** Accepted
**Date:** 2026-04-27 (the design principle, articulated post-CD)
**Deciders:** project owner

## Context

Two structurally-wrong approximations were diagnosed and bypassed (not
fixed) on the same problem:

1. **LP-HPWL.** The polyhedra system used LP-HPWL to rank polyhedra
   and moves. The Miftari analysis on ibm01 showed rho(LP-HPWL,
   refined proxy) = -0.001 across 24 feasible topologies. The natural
   "fix" — improving the LP relaxation, the surrogate, or the
   topology extraction — was tested in the 22-experiment overnight
   sweep and produced 22 results within +/-0.5 % of baseline. The
   bypass was ADR-001: optimize the joint composite proxy directly
   via DPO.

2. **DPO's RUDY congestion.** DPO replaced LP-HPWL with a
   differentiable surrogate of the full proxy, and the RUDY congestion
   model has 10.9 % top-5 % hotspot overlap with real congestion
   (Jaccard 0.057, near-random); 3.1x magnitude gap; 28 % of real
   congestion comes from macro blockage that RUDY ignores entirely.
   The natural "fix" — scaling the congestion weight — was tested in
   a 0.5 to 1.5 sweep and was monotonically worse, falsifying it: the
   gradient *direction* is wrong, not the magnitude. The bypass was
   ADR-002: build an incremental evaluator and run CD on the real
   proxy.

The pattern is identical: a structural mismatch (rho = -0.001 in case 1;
near-random hotspot overlap in case 2) is diagnosed by quantitative
analysis, the natural amplification fix is falsified, and the resolution
is to evaluate the truth instead of building a better lie.

## Decision

When an approximation is structurally wrong (not noisy), prefer exact
evaluation backed by faster data structures over a more accurate
approximation.

"Structurally wrong" is operationalized as: a correlation analysis
shows the approximation does not even rank options correctly (LP-HPWL),
or a spatial overlap analysis shows the approximation disagrees with
truth on which inputs are critical (RUDY hotspots). Magnitude errors
are tunable; direction errors are not.

## Consequences

Positive: drove the two pivots that produced the -24.2 % vs RePlAce
result (1.4578 -> 1.1055). Both pivots succeeded; both natural-fix
paths were falsified. This is the design principle that explains why
our trajectory differs from a literature-based one — DREAMPlace++,
ReFine, and similar approaches lean on improved differentiable
surrogates.

Negative: rules out future "build a better RUDY" tracks unless the
evidence shifts. Costs: exact evaluation is harder to vectorize than
batched gradient kernels — per-move incremental updates rather than
GPU-batched forward/backward passes. Does not generalize beyond
problems where an incremental evaluator can be built (problems with
non-decomposable, non-incremental cost functions are out of scope for
this principle).

Alternatives ruled out: improving LP-HPWL (22 experiments within
+/-0.5 %); improving RUDY by reweighting (monotonically worse);
improving RUDY by adding macro-blockage and L-routing terms (a more
accurate RUDY would still be an approximation, and the incremental
evaluator already gives exact in milliseconds).

This ADR is the design principle behind ADR-001 and ADR-002. Those
ADRs record the specific decisions; this one records the meta-rule
that produced both.

## Evidence

- `writeup/evidence.md` §4.2 (LP-HPWL rho = -0.001, Miftari analysis)
- `writeup/evidence.md` §5.1 (RUDY hotspot overlap 10.9 %, Jaccard
  0.057, 3.1x magnitude gap)
- `writeup/evidence.md` §5.2 (congestion-weight sweep 0.5 to 1.5,
  monotonically worse)
- `writeup/evidence.md` §7 (CD breakthrough: incremental evaluator +
  full-proxy CD)
- `writeup/contributions.md` §10 ("bypass, don't fix" as a design
  principle)
- ADR-001 (the LP-HPWL bypass)
- ADR-002 (the RUDY bypass via the incremental evaluator)
