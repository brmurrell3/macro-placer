# CDLNSSACascadeMultidirPlacer (submission entry candidate)

**Status:** experimental. Local cached-input validation passed (cascade-multidir 3-iter on ibm01 = **-0.614 %** lift). Cloud single-pass --fast aggregate marginal (~0.05-0.10 %); cloud cascade-multidir end-to-end test still pending.

## What it does

Wraps the current submission floor (`submissions/cd_lns_sa_cascade/placer_adaptive.py`)
with one additional phase: multi-direction Hessian saddle escape (E90)
on the cascade output.

```
Phase A (parent CDLNSSACascadeAdaptivePlacer):
   E25 → E41 → best plateau → single-direction cascading saddle escape
   = current submission floor (IBM 1.137 / NG45 0.6925 verified)

Phase B (new):
   cascade_multidir polish — iterate multi-direction saddle escape until
   plateau or budget exhausted
```

## Why it should help

Single-direction saddle escape (E74/E84) perturbs along ONE softest
eigenvector at a time and never combines them. At the cascade-converged
plateau, the smooth-proxy Hessian has multiple negative-curvature
eigvecs (PATH A diagnostic, ibm01: `λ = [-0.14, -0.08, -0.07, -0.06]`
for k=4). Combinations of these soft modes reach basins single-direction
misses.

**Spike result on cached cascade ibm01** (E90, `c543056`):
- Cascade-cached input: canon 0.84528, 0 ovl (cascade saturated on
  single-direction).
- Multi-direction polish: best polished canon **0.84231** (rank-2
  sv=(1,1,0) eps=2.0) = **+0.352 % lift**.
- 16/20 attempts found lifts; best 3 of top 4 were rank-2 same-sign
  pairs of soft modes 1 and 2.

## Wall budget split

Total `budget_seconds` (default 3000s = 50 min, fits 60-min cap with
10-min margin) splits as:

- 80 % → cascade-adaptive (= 2400s)
- 20 % → multi-direction polish (= 600s)

Per-attempt polish budget is 60 s; with K=3 eigvecs, rank-≥2 sign vectors,
eps ∈ {0.5, 2.0}, the multi-direction phase runs ~10 attempts × 60s = 10 min
per cascade-multidir iteration, ~2 iterations within the 600 s envelope.

## Falls back gracefully

If multidir fails or runs out of budget, the cascade-adaptive result is
returned unchanged. Multidir is **strictly additive** — never returns
worse than the parent.

## Pending decisions

- `--fast` aggregate ≥ 0.3 % gates promotion to `--all`.
- `--all` aggregate ≥ 0.3 % gates promotion to `--ng45`.
- `--ng45` must not regress ariane133 (historical failure point — see
  e54/e62 in memory).
- If all gates pass, this becomes the new submission entry, replacing
  `submissions/cd_lns_sa_cascade/placer_adaptive.py`.
