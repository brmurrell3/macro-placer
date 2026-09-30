---
id: E167
name: v4_poisson_pick
status: in_progress
parent: E164
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E167: v4_poisson_pick — Poisson-augmented multi-seed basin scoring

## Hypothesis

E164 picks multi-seed lanes by canonical proxy alone. Canonical proxy
uses ABU-5% (top-5% mean) for congestion, which is a local indicator.
Poisson-FFT congestion (DCGP DAC'25 style) gives a globally-smooth
"pressure field" derived from solving ∇²φ = -ρ on net+macro route demand.
Multi-objective pick using `canonical + α*poisson` may select basins
that look marginally worse on canonical alone but are structurally
better for cascade/CD polish (lower global pressure → easier escape).

## Method

Same as E164 (3-seed V4 + cascade saddle) but lane pick uses combined
score `canonical_proxy + α*poisson_congestion` where:
- canonical_proxy = TILOS PlacementCost (our objective)
- poisson_congestion = top-5% |φ|, φ from FFT Poisson solve on
  V_total + H_total demand grids
- α = 0.1 (tunable; α=0 reduces to E164)

Pipeline:
1. Run V4 for each seed → 3 basins
2. For each basin: compute canonical AND poisson scores
3. Pick lane with lowest combined score
4. CD polish + cascade saddle (same as E164)

Budget: 1500s (same as E164). Poisson eval adds ~2s per basin (negligible).

## Kill gate

ibm03 smoke: poisson pick must select a DIFFERENT lane than canonical
pick on at least 1 of {ibm03, ibm10, ibm17} (mechanism shows novelty),
AND final ≤ E164 final + 0.001.

If poisson always picks same lane as canonical → α too low or signals
correlated. If picks differently but final is worse → poisson signal
misleading.

## Generalization check

--fast: poisson α=0.1 vs α=0 (E164) head-to-head. If different picks
and better final on ≥ 2 of 4 benches, promote.

## Outcome (filled when decided)

**Smoke ibm03 FALSIFIED — implementation bug** (2026-05-21):
- V4 basin: proxy=0.90087 ovl=0
- After Poisson refine (100 Adam steps, α=0.5): **proxy=1.25681 ovl=0 (+39.5%!)**
- Refinement loss was `poisson_alpha * poisson + overlap_lambda * overlap`
  ONLY — completely OMITTED WL, density, and canonical congestion from
  the gradient. Adam minimized Poisson alone, which has no relationship
  with canonical proxy beyond their congestion components.

This is not a falsification of the DCGP mechanism, only of the
implementation. To properly test, the refinement loss must include the
ORIGINAL V4 loss (WL + 0.5*density + 0.5*canonical_cong + overlap)
plus the Poisson penalty as an extra term.

Status: superseded by E170 (fixed implementation).

Confirmed final: 0.91251 ovl=0 wall=1018s. CD+cascade couldn't fully
recover from refine's +0.36 destruction. +2.9% worse than E166 ibm03 (0.8870).



## Pointers

- Code: `code/placer.py`
- Poisson primitive: `code/poisson_congestion.py`
- Parent: E164 (`experiments/E164_v4_multiseed_cascade/`)
