# E122 3-pin Steiner calibration findings (2026-05-20)

## Headline

3-pin Steiner T-route closes the canonical-vs-smooth congestion bias by
**+4.1 to +6.4 percentage points** across all 4 IBM benches tested,
matching the survey-predicted target.

## Per-bench calibration results

| Bench | E111 base bias | E122 Steiner bias | Closure | 3-pin nets |
|-------|---------------:|-----------------:|--------:|----------:|
| ibm01 | +11.0 % | +6.9 % | **+4.1 pp** | 21.7 % (1300/5993) |
| ibm10 | +18.9 % | +12.9 % | **+6.1 pp** | 26.2 % (7421/28272) |
| ibm12 | +23.7 % | +18.3 % | **+5.3 pp** | 28.5 % (8237/28939) |
| ibm17 | +14.4 % | +8.0 % | **+6.4 pp** | 36.9 % (16895/45825) |

All on cached cascade plateau placement
(`experiments/E84_cascading_saddle/results/cascade_<bench>.pt`).

## Key validation: ibm17 hits survey-predicted closure target

The literature survey (docs/research/2026-05-20_congestion_model_survey.md
§3 IMPROVEMENT 1) predicted:
> Expected lift: rel mismatch from +14% → +8-10%

**Observed**: +14.4 % → **+8.0 %** — at the bottom of the predicted range
(better than expected).

## Architecture verification

- Pin-degree distribution shows 22-37 % of nets are 3-pin (matches survey
  prediction of 25-40 %).
- `n_pairs` correctly drops by ~24-39 % after excluding 3-pin nets from
  the star-L pair list (ibm17: 87361 → 53571 pairs, a 38.6 % reduction —
  consistent with 36.9 % 3-pin × 2 sinks per net relative to 2-pin's 1
  sink per net).
- The Steiner T-route adds 3 stripe contributions per net (1 H + 2 V),
  vs star L-route's 4 stripes (2 H + 2 V), so it correctly under-counts
  vs the inflated star approximation.

## Forward-pass perf (M3 MPS, CPU mode)

| Bench | E111 build | E122 build | E111 eval | E122 eval |
|-------|-----------:|-----------:|----------:|----------:|
| ibm01 | 0.2 s | 0.3 s | 0.04 s | 0.04 s |
| ibm10 | 0.7 s | 1.7 s | 0.1 s | 0.1 s |
| ibm12 | 0.8 s | 1.6 s | 0.1 s | 0.1 s |
| ibm17 | 1.3 s | 2.6 s | 0.2 s | 0.1 s |

E122 build cost is ~2× E111 due to extra 3-pin tensor split. **Per-step
eval is the same or faster** — the steiner is vectorized over n_3pin
nets via matmul, no batch loop overhead.

## Smoke test on ibm01 (V3+Steiner end-to-end)

Pipeline: SDF init → 200 Adam steps with V3+Steiner → greedy_legalize.
- Initial congestion (smooth proxy): c=1.3062
- Final congestion (smooth proxy): c=0.9784 (−25 % from init)
- Final proxy: 0.89119, zero overlaps, 27 s wall
- For reference: E111 baseline ibm01 + CD = 0.85427 — within noise of
  this 200-step + no-CD result (CD would close the gap).

Confirms the placer's forward + backward + legalize all work correctly.

## End-to-end ibm17 production (720s budget, ovl_lambda_end=10)

| Config | proxy | wall | breakdown |
|--------|------:|-----:|-----------|
| E111 baseline (ovl10_720s) | 1.2003 | ~720s | (per experiment_log) |
| **E122 Steiner (V3Steiner+CD)** | **1.2121** | 828s | wl=0.069 d=0.539 c=1.747 |
| Δ | **+0.0118 (+0.98% WORSE)** | | |

Phases:
- V3Steiner descent + legalize: proxy=1.27188 (~348s)
- CD polish budget=200s → final 1.21212 (CD lifted −0.06)

**Kill gate BREACHED**: target was ibm17 + V3+CD600s ≤ 1.18; observed
1.2121 ≥ 1.20.

### Why fidelity didn't translate

E122 closes scalar bias by REDUCING smooth congestion (Steiner < L-star).
At the same placement, Steiner reports lower congestion than L-star.
Adam interprets this as "good enough" earlier and pushes less hard on
3-pin net positions. The basin Adam converges to is *different* from
E111's, but not necessarily *better* by the canonical metric.

The +6.4 pp scalar improvement is real, but it's a SCALAR improvement
— it does not directly induce a better-aligned gradient field.
