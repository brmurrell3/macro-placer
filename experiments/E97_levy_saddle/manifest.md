---
id: E97
name: levy_saddle
status: in_progress
parent: E84
created: 2026-05-13
decided: null
champion_at_time: 1.078
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E97: levy_saddle — Heavy-tail ε magnitudes for cascade saddle escape

## Hypothesis
The E84 cascade saddle uses a fixed Gaussian ε grid `(0.3, 1.0, 3.0)`. Heavy-tail
(half-Cauchy) magnitudes give magnitude diversity that the fixed grid misses:
~50% draws near σ, occasional 5-10× large jumps (escape distant basins) and
small ~0.1× probes (find adjacent valleys). Same eigvec direction; only the
distribution over ε changes.

## Method
Replace the fixed `eps_values` argument of `cascading_saddle_escape` with K
samples from `|Cauchy(0, σ)|` (inverse CDF: `σ·tan(π/2·u)`, clipped to
`[0.01·σ, 50·σ]`). Defaults: K=3 magnitudes (matches Gaussian's count for fair
wall budget), σ=1.0 (matches median of Gaussian grid).

## Kill gate
Spike H2H on ibm01 + ibm03 + ibm10 (cached or fresh plateau). If Lévy's best
proxy ≥ Gaussian's (at fair K and same wall) on any 2 of 3, kill.

## Generalization check
After spike validation, integrate as drop-in placer at
`submissions/cd_lns_sa_cascade_levy/placer.py`. Run `uv run evaluate ... --all`
on EPYC. Promote if aggregate improvement vs cascade-adaptive 1.078 is positive
and no regression > 1% on any individual bench.

## Outcome (filled when decided)

### Spike H2H (2026-05-13, same wall, same eigvec, seed=42, polish_budget=60s)

| Bench | Gauss (K=variable) | Lévy K=3 | Δ in Lévy's favor |
|-------|-------------------:|----------:|-------------------:|
| ibm01 | best=0.91148 (-0.22%, K=3, 195s) | best=0.91045 (-0.35%, K=6, 290s)* | -0.13% |
| ibm03 | best=0.99555 (-0.091%, K=6, 202s) | best=0.99323 (-0.317%, K=3, 202s) | -0.23% |
| ibm10 | best=1.08900 (-1.99%, K=6, 199s) | best=1.08696 (-2.19%, K=3, 201s) | -0.20% |

*ibm01 H2H reversed K values; Lévy K=6 was the initial spike before K-fair
runs landed.

**Result: 3/3 benches show Lévy wins by 0.13-0.23%.** Lévy at K=3 beats
Gaussian at K=6 on ibm03 and ibm10 — magnitude diversity from heavy tails
outweighs sheer count of Gaussian probes.

### Production integration
`submissions/cd_lns_sa_cascade_levy/placer.py` is the wall-safe drop-in (same
E25 → E41 → Lévy saddle pipeline, identical fallback chain).

Local smoke (budget_seconds=400) ibm01: proxy=0.87808 ovl=0 wall=385s.
Cloud --fast --json --hypothesis E97_levy_fast running on EPYC; ibm01
already at 0.86931 after eps=0.824 in saddle phase (-2.92% from E41 plateau).

**Composition variants ready** (awaiting validation):
- `submissions/cd_lns_sa_cascade_dual_levy/placer.py` — adds cong-focus eigvec
  per E100 portfolio finding (2-weight portfolio with K=3 Lévy)
- `submissions/cd_lns_sa_cascade_dp_levy/placer.py` — composes with PATH B
  DP-lane hybrid; BLOCKED on cloud DREAMPlace rebuild

## Pointers
- Code (spike): `code/levy_saddle.py`
- Code (production): `submissions/cd_lns_sa_cascade_levy/placer.py`
- Results: `results/ibm0{1,3,10}_{gaussian,levy}_seed42.json`
- Parent E84: `experiments/E84_cascading_saddle/`
- Plan: `PLAN.md`
