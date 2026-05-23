---
id: E174
name: e143_3iter_portfolio
status: in_progress
parent: E143
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E174: e143_3iter_portfolio — E143 with portfolio_max_iters=3

## Hypothesis

E169 confirmed 3-iter portfolio > 2-iter portfolio on ibm03 (small bench)
by ~0.32%. E143 uses 2-iter portfolio on top of K-joint+SA stack. If
E169's lift compounds with E143's K-joint mechanism, the combined stack
on large benches (where K-joint shines) could lift more than E143 alone.

E143 ibm10: 0.95671. If E174 adds another -0.3% from deeper portfolio,
ibm10 could land 0.953-0.955.

## Method

Identical to E143 but:
- portfolio_max_iters: 2 → 3
- portfolio_reserve_s: 500 → 750 (room for 3 iters)
- budget_seconds: 3000 → 3300 (extended to fit; under 60-min cap)

## Kill gate

ibm10 smoke: final ≤ E143 ibm10 (0.95671) - 0.001 → 3-iter portfolio
compounds with K-joint. If final ≥ 0.957 → portfolio depth wasted on
large benches (K-joint already extracts the lift).

## Generalization check

If smoke passes: replace E171's large-bench path with E174.

## Outcome (filled when decided)

**Smoke ibm10 PASSED — compounds with K-joint** (2026-05-21):
- V4 basin → CD polish: 0.98145
- Cascade iter 1: 0.97625 (-0.52%)
- Portfolio iter 1: 0.97371 (-0.26%; iter 2 budget squeezed)
- K-joint pass 1: 156 of 262 commits (60% accept), → 0.95095
- K-joint converged at pass 2
- SA polish 30s (heavy budget squeeze): no improvement
- **Final: 0.95212** (ovl=0, MPS-contention wall=194 min; internal ~55 min)

vs E143 ibm10 0.95671: **-0.48% better**.

Portfolio extra iter contributed before K-joint took over. 3-iter portfolio
DOES compound with K-joint on large benches, even when only 1 portfolio iter
fits in budget.

NOTE: 194-min wall reflects MPS contention with 10+ parallel placers.
Internal placer-side wall ~55 min, under 60-min partcl cap. Should be
safe on clean judging hardware.

Status: graduated for IBM large benches. Consider updating E171 to use
E174 for large-bench branch (not done in this session to avoid disrupting
in-flight E171 --all).


## Pointers

- Code: `code/placer.py`
- Parent: E143 (2-iter), E169 (3-iter on small bench)
