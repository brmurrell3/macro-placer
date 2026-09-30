---
id: E168
name: v4_wider_portfolio
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

# E168: v4_wider_portfolio — 7-weight portfolio + budget reallocation

## Hypothesis

E166 showed portfolio's 4 weights ALL independently productive:
- canonical (-0.30%), cong-focus (-0.45%), density-focus (-0.16%),
  non-WL (-0.45%) on ibm03.

If each non-canonical weight finds a new escape direction, adding more
weights — especially the SINGLE-COMPONENT extreme weights (1,0,0),
(0,1,0), (0,0,1) — should find directions orthogonal to all 4 existing
weights. Single-component weights probe pure WL, pure density, pure
congestion Hessians.

Budget reallocation: portfolio 500s→800s, cascade 400s→200s. Cascade
gave only -0.33% on ibm03 vs portfolio's -1.36% — clear ROI signal to
move budget toward portfolio.

## Method

Same as E166 but:
- 7-weight portfolio: 4 from E166 + (1, 0, 0), (0, 1, 0), (0, 0, 1)
- portfolio_reserve_s: 800 (was 500)
- cascade_reserve_s: 200 (was 400)
- cascade_max_iters: 1 (was 2)

Other phases unchanged.

Budget 2400s (40 min, under 60-min cap):
- 4 lanes: 800s
- CD polish: 500s
- Cascade: 200s
- Portfolio: 800s
- Safety: 100s

## Kill gate

ibm03 smoke: final ≤ E166 ibm03 (0.88700) - 0.001 → portfolio width adds
material lift. If final ≥ 0.887 → portfolio is saturated at 4 weights.

## Generalization check

--fast: aggregate ≤ E166 --fast - 0.2% AND no single-bench regression.

## Outcome (filled when decided)

**Smoke ibm03 FALSIFIED** (2026-05-21):
- Multi-init pick: sdf42 (canonical=1.05010)
- CD polish: 0.89446 (deeper than E166's 0.89926 due to longer CD budget)
- Cascade: no improvement (only 200s budget; cascade couldn't fit useful iters)
- Portfolio 7 weights, iter 1:
  - w[0..3] (mixed): no improvement
  - w[4]=(1,0,0) pure WL: λ_w ≥ -tol, skipped (not soft mode)
  - w[5]=(0,1,0) pure density: no improvement
  - w[6]=(0,0,1) pure congestion: no improvement
- "no improvement this iter; saturated"
- **Final: 0.89446** (vs E166 ibm03 0.8870 = +0.84% WORSE)

**Two compounding failures**:
1. Trimmed cascade (200s) → cascade couldn't find iter 2 with eps grid
2. Extended CD (500s) → basin too deep for portfolio to find escape
   directions (headroom-competition pattern from E144)

Wider portfolio (7 weights) didn't help — pure-component weights either
already covered by canonical/mixed weights, or basin too polished to
escape from. **The original 4 mixed weights in E166 are optimal**.

Status: killed. E169 (same 4 weights, 3 iters, balanced budget) is the
correct extension.


## Pointers

- Code: `code/placer.py`
- Parent: E166
