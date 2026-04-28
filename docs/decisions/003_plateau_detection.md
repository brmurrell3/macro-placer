# ADR-003: Use per-benchmark plateau detection, not a fixed wall-clock budget

**Status:** Accepted
**Date:** 2026-04-27
**Deciders:** project owner

## Context

CDOnly with a fixed 600 s/bench budget produced 1.1193 avg on `--all`
(0 overlaps, every benchmark beat DPO). Inspecting per-benchmark sweep
deltas at the 600 s cap revealed two failure modes of the fixed budget:
hard benchmarks (ibm17, ibm18, ibm14, ibm12) had sweep-deltas of
0.001-0.003 — still descending, just out of time — and easy benchmarks
(ibm04, ibm09, ibm11) plateaued within 5-6 minutes and spent the
remaining 4-5 minutes producing no improvement.

This is a budget-allocation problem, not an algorithm problem. The
algorithm (CD on the incremental evaluator) was the same; the question
was when each benchmark should stop.

A fixed budget tuned on IBM is also brittle to dataset shift: the hidden
NG45 commercial designs are unseen, and any fixed schedule fits a prior
that may not hold. The competition rule (1 hour per benchmark) gives a
hard cap but no guidance on how to spend the time within it.

## Decision

Replace the fixed budget with per-run, per-benchmark plateau detection.
Defaults `(min_time_s=300, hard_cap_s=3600, patience=3,
plateau_threshold=0.005)`: each benchmark runs for at least 5 minutes,
then exits when the last 3 sweep-deltas all drop below 0.005, or when
the 1-hour competition cap is reached — whichever comes first.

This is a per-run policy, not a tuned schedule. It adapts to whatever
benchmark the optimizer sees rather than being fitted to one dataset.

## Consequences

Positive: 1.1193 to 1.1055 on `--all` (-1.23 %), with hard benchmarks
gaining 1.7-3.6 % from the extra time and easy benchmarks tied within
+/-0.4 %. All 17 IBM benchmarks exit via plateau; none hit the 3600 s
cap. Total wall 17480 s = 4.85 hr, well inside the 17-hr competition
envelope (17 benches x 1 hr). Per-bench wall ranges from 322 s (ibm01)
to 2238 s (ibm17). Per-benchmark adaptivity transfers to NG45 without
retuning: the policy is a function of the descent trajectory, not the
benchmark identity.

Negative: ~70 % more wall than CDOnly's fixed 600 s budget (10316 s to
17480 s). The extra time is spent on benchmarks where it pays off, but
total throughput on a dev machine is lower. The plateau-detection logic
adds a small amount of state to the placer.

Alternatives ruled out: a tighter fixed budget (would re-create the
mid-descent cutoff on hard benchmarks); a per-benchmark tuned budget
(violates the no-per-benchmark-tuning contest rule, and would not
transfer to NG45); aggressive early stopping (easy benchmarks already
plateau within the 5 min minimum).

## Evidence

- `writeup/evidence.md` §7.5 (E9 CDAdaptive defaults and result)
- `writeup/evidence.md` §2.1 (per-bench table: 17/17 plateau, none hit
  cap, 17480 s total)
- `writeup/evidence.md` §11 (compute envelope: 4.85 hr vs 17 hr cap)
- `writeup/contributions.md` §12 (per-benchmark plateau detection, E9)
- `results/CDAdaptivePlacer_20260427_132214.json`
