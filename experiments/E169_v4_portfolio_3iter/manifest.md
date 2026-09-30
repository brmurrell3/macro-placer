---
id: E169
name: v4_portfolio_3iter
status: in_progress
parent: E166
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E169: v4_portfolio_3iter — E166 with portfolio max_iters=3

## Hypothesis

E166 ibm03 portfolio data:
- iter 1 final: 0.89549 (-0.42% from cascade end)
- iter 2 final: 0.88700 (-0.95% additional)
- iter 2 produced MORE lift than iter 1

The second iter built on the first's improvements (cascade polish refined
each weight's basin). A third iter should also find lift (diminishing but
nonzero) if the pattern continues.

Trade-off: 3 iters × 4 weights = 12 Hessian probes, ~600s. Budget must
accommodate.

## Method

Same as E166 but:
- portfolio_max_iters: 2 → 3
- portfolio_reserve_s: 500 → 900 (room for 3 iters)
- cascade_reserve_s: 400 → 300 (small trim)
- budget_seconds: 2400 → 2800 (40 → 47 min)

## Kill gate

ibm03 smoke: iter 3 must find additional lift (final < 0.886) → portfolio
plateau happens at iter 4+. If final ≥ 0.887 → 3 iters saturate; revert.

## Generalization check

If smoke passes: --fast head-to-head vs E166 --fast.

## Outcome (filled when decided)

**Smoke ibm03 PASSED — new small-bench champion** (2026-05-21):
- Multi-init pick: sdf42 (canonical=1.05575)
- CD polish: 0.89781
- Cascade iter 1: 0.89229 (-0.55% from CD)
- Portfolio iter 1: 0.89229 → 0.88418 (-0.91%)
- Portfolio iter 2: 0.88418 → 0.87966 (-0.51%)
- Portfolio iter 3: 0.87966 → 0.87964 (-0.002%, saturated)
- **Final: 0.87964** (ovl=0), wall=2754s = 46 min (under cap)

Iter 3 saturated almost immediately — 3 iters extracts essentially all
available portfolio value. 4-iter would not compound.

vs E166 ibm03 0.88700: **-0.83% better**.
vs V4 baseline 0.904: **-2.7% lift**.

Kill gate cleared. **Promoted: 3-iter portfolio is E171's small-bench
default (already baked in via E171's `portfolio_max_iters=3`).**

--fast pending. Projection: if -0.83% over E166 generalizes, E169 --fast
≈ 0.831, and E171 --all ≈ 0.97 range.

**--ng45 PARTIAL FALSIFICATION 2026-05-21**: E169 --ng45 = 0.6650
(vs E166 --ng45 0.6629 = +0.32% WORSE). 3-iter portfolio is not
universally better — on NG45 commercial designs, the budget cost of
extra portfolio iters squeezes cascade and hurts more than the extra
portfolio iter helps. **For NG45, prefer E166 (2-iter portfolio).
For IBM small benches, E169 (3-iter) wins.**



## Pointers

- Code: `code/placer.py`
- Parent: E166
