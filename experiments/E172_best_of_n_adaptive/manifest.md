---
id: E172
name: best_of_n_adaptive
status: in_progress
parent: E171
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E172: best_of_n_adaptive — best-of-2 wrapper around E171

## Hypothesis

E171 adaptive stack uses single rng_seed=42 for cascade/portfolio/K-joint
RNG. Different seeds produce different Lévy ε samples in portfolio and
different K-joint tuple selections — could find different basins.

Running E171 twice with different seeds and picking the better canonical
result is a stochastic restart strategy. Doubles wall but each run gets
half the budget. Trade: parallel exploration vs single deeper search.

## Method

Run E171 twice with budget=1500s each:
- Run 1: rng_seed=42 (default)
- Run 2: rng_seed=123

Return whichever has lower canonical proxy (and zero overlaps).
Total wall budget 3000s = 50 min (under 60-min cap).

## Kill gate

ibm03 smoke: final ≤ E171 ibm03 - 0.001 → multi-restart adds value.
If final ≥ E171 → single run was enough at 1500s; multi-restart isn't
worth the budget split.

## Generalization check

--fast: aggregate ≤ E171 --fast - 0.2%.

## Outcome (filled when decided)

(pending E171 + this smoke)

## Pointers

- Code: `code/placer.py`
- Parent: E171
