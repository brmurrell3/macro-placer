# ADR-001: Optimize the full composite proxy directly, not LP-HPWL

**Status:** Accepted
**Date:** 2026-04-23
**Deciders:** project owner

## Context

The polyhedra navigation system reached an avg proxy of 1.4867 on `--all`
(Phase 2, 300 s) and stalled. A 22-experiment overnight sweep across three
independent subproblems — surrogate accuracy (SP3), initial topology (SP1),
and LP formulation (SP4) — landed every result within +/-0.5 % of the
1.4921 baseline. Global best 1.4918. The system was at a plateau, and the
plateau was structural, not parametric.

A Miftari-style correlation analysis on ibm01 then explained why. Across
24 feasible topologies (Hamming 0-2249 from base), LP-HPWL has rho =
-0.001 with the refined proxy. LP-HPWL correlates strongly with refined
WL (rho = +0.852) but anti-correlates with refined density (rho = -0.536),
and the two effects nearly cancel. LP-HPWL is essentially blind to
refined congestion (rho = +0.072), and refined congestion is what
dominates: the component breakdown across all 17 benchmarks shows
congestion at 74.9 % of proxy cost on average (64-81 % per benchmark),
density at 19.4 %, WL at only 5.8 %.

Cheap proxies for LP-HPWL are accurate (rho = 0.86, 42700x speedup), but
LP-HPWL itself carries no usable signal about the metric we are scored on.

## Decision

Abandon any algorithm that ranks polyhedra or moves by LP-HPWL. Optimize
the joint composite f(p) = WL + 0.5 D + 0.5 C directly, including the
congestion term in whatever objective the optimizer sees.

This is the foundational decision of the DPO and CD generations: every
move-acceptance, gradient, and ranking after this point evaluates the
full composite, not a component proxy.

## Consequences

Positive: enabled the DPO generation (1.3834) and the CD generation
(1.1055), the two structural pivots that produced the -24.2 % vs RePlAce
result. Removed the temptation to keep tuning surrogates or LP
formulations that operate on a structurally wrong objective.

Negative: rules out a class of well-developed techniques that depend on
LP duals or LP relaxations of HPWL — Lagrangian bounds from LP duals,
LP-relaxation MCMC weighted by LP cost, partial-commitment LP, dual-
informed targeting. These all looked promising on paper and several were
implemented in the SP4 stage; none survived.

Alternatives ruled out: improving the surrogate (SP3 stage, all within
noise); changing the initial topology fed to LP (SP1 stage, only
congestion-aware extraction finished, at 1.4930); modifying the LP
formulation (SP4 stage, best 1.4918, marginal). All 22 experiments
across these three axes were within +/-0.5 % of baseline.

## Evidence

- `writeup/evidence.md` §4.1 (22-experiment overnight sweep)
- `writeup/evidence.md` §4.2 (Miftari correlation analysis, ibm01)
- `writeup/evidence.md` §3.3 (per-benchmark component breakdown,
  congestion 74.9 % avg)
- `writeup/contributions.md` §2 (barrier diagnosis: the objective
  mismatch)
- `writeup/contributions.md` §5 (non-decomposability of the proxy cost)
