---
id: E121
name: mcf_congestion
status: falsified
parent: E111
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.05750
outcome: ibm17 + CD600s = 1.19975, matches V3Min ovl10 720s baseline 1.20032 within noise (-0.03%). MCF with rudy_mix=1.0 collapses to canonical L-route (E111-equivalent); MCF with rudy_mix<1.0 had worse calibration. Bug-fix lift (hard min/max) is real but separable from MCF — see Outcome.
champion_delta: 0.0
graduated_to: null
superseded_by: null
---

# E121: Multi-Commodity Flow Congestion (cross-domain)

## Hypothesis

VLSI placement literature uses RUDY/ABU congestion (one deterministic
L-route per pin pair). Network optimization literature treats routing
as **multi-commodity flow (MCF)**: each pair is a commodity with one
unit of flow distributed over MANY plausible paths, weighted by length.
A smooth MCF approximation gives a fundamentally different gradient
signature than RUDY:

- RUDY peaks on source-row + sink-col (L-route corner): cell at
  (src_row, sink_col) gets HEAVY weight; cells off the L get zero.
- MCF (uniform over monotonic paths) peaks on the BBOX DIAGONAL:
  cell weight follows the hypergeometric distribution
  `C(d_src, |Δc_src|) * C(d_snk, |Δc_snk|) / C(D, |Δc|)`
  where `D = |Δr| + |Δc|`. This spreads flow over the whole bbox.

The MCF gradient pushes pins to **broad bbox-balance**, where RUDY
pushes to **L-corner-avoidance**. On hard benches where canonical
congestion isn't dominated by single corners (ibm17 has dense
peripheral macros across many corners), the broader MCF signal should
be more aligned with the canonical top-5% structure.

## Method

Implement a smooth MCF approximation: for each (source, sink) pair,
the expected flow through cell (r,c) under uniform distribution over
monotonic lattice paths is the **transportation hypergeometric**:

  P((r,c) on path) = exp(lgamma(d_src+1) - lgamma(|Δr_src|+1) - lgamma(|Δc_src|+1)
                       + lgamma(d_snk+1) - lgamma(|Δr_snk|+1) - lgamma(|Δc_snk|+1)
                       - lgamma(D+1) + lgamma(|Δr|+1) + lgamma(|Δc|+1))

where d_src = |Δr_src| + |Δc_src|, etc. This is **uniform** over
monotonic paths (all have length D, so length-softmax is uninformative
here). To get a **temperature** parameter that interpolates between
RUDY (L-routes) and uniform MCF, we mix:

  P(c | τ) = (1 - α(τ)) * P_RUDY(c) + α(τ) * P_MCF_uniform(c)

with α(τ) = sigmoid((τ - 1) * k). At low τ (sharp), behaves like RUDY;
at high τ (smooth), behaves like uniform MCF. Differentiable everywhere.

**Why this is novel:**
1. RUDY/ABU comes from EDA (Spindler & Johannes 2007, ICCAD).
2. MCF / network flow comes from operations research (Ford-Fulkerson
   1956; Garg-Konemann 2007 for the fast approximation scheme).
3. The transportation polytope / monotonic-path distribution is from
   combinatorial probability (Polya 1937).
4. No public placer fuses MCF + smooth proxy + temperature interpolation.

## Kill gate

- Smooth MCF must be within ±20% of canonical congestion on cached
  cascade placements for ibm10/12/17 (the three hardest benches).
- ibm17 + CD600s using MCF-smooth must beat V3Min ovl10 720s baseline
  (1.20). Target < 1.18.
- If calibration is >40% off → kill (MCF distribution doesn't match
  canonical top-5%, which uses L-route demand only).
- If ibm17 ≥ 1.20 → kill (no lift over RUDY-based E111).

## Generalization check

If ibm17 lifts, run --fast (4 benches) then --all (17 IBM) at the V3Min
ovl10 720s budget and confirm aggregate avg ≤ 1.000. Single-bench-only
wins are insufficient (E54/E62 falsification pattern).

## Outcome — FALSIFIED 2026-05-20

### ibm17 end-to-end (rudy_mix=1.0 K=3 τ=0.05) — primary kill gate

| Metric | E121 MCF rudy_mix=1.0 | V3Min ovl10 720s (E111) baseline |
|---|---:|---:|
| RAW (smooth + legalize) | 1.28346 | 1.30396 |
| + CD 600s polish | **1.19975** | 1.20032 |
| wall (descent+legal+polish) | 1273 s | ~720 s |

**Δ = −0.05 % (essentially tied within M3 noise floor ~0.5%).**
Kill gate: ibm17 < 1.20 with target < 1.18. We hit 1.19975 — ON the
kill-gate boundary but no improvement over the E111 baseline.

### Calibration (2026-05-20)

After fixing a bug in `_soft_min_max` (degenerate args produce phantom
stripe of width ~2*log(2)/beta ≈ 0.23 cells) — switched to exact
`torch.minimum/maximum` for the H1/H2 stripe ranges:

| Config | ibm10 | ibm12 | ibm17 | max\|Δ\| |
|---|---:|---:|---:|---:|
| rudy_mix=1.00 K=3 τ=0.05 | +16.8% | +20.3% | +12.6% | **20.3%** |
| rudy_mix=0.50 K=5 τ=0.15 | +19.8% | +26.6% | +15.1% | 26.6% |
| rudy_mix=0.00 K=7 τ=0.30 | +20.7% | +29.4% | +16.7% | 29.4% |

vs E111 baseline (per_net_trace): +18.9% / +23.7% / +14.4% — MCF
rudy_mix=1.0 with the hard-min/max fix is better than E111 on all 3
benches.

### Why MCF didn't lift over E111

1. **At rudy_mix=1.0 the MCF model collapses to canonical L-route**
   (bend column = c_snk). With the soft-min/max bug fixed, this matches
   E111 within ~5% calibration. The end-to-end result (1.19975) confirms
   that this RUDY-equivalent path matches V3Min baseline (1.20032).

2. **At rudy_mix < 1.0, MCF distributes path mass across K bend columns.**
   This produces WORSE calibration (max\|Δ\| 26-29% vs 20% at rudy_mix=1)
   because the multi-bend distribution spreads V demand across cols and
   adds H demand on snk_row (vs canonical's pure src_row H stripe).
   The wider gradient signature is less aligned with canonical's L-route
   structure than RUDY/E111's tighter L-route. **MCF mixing was tested
   only via calibration; given the calibration was worse, an end-to-end
   run with rudy_mix < 1 was not pursued.**

3. **The 5-10% calibration improvement at rudy_mix=1.0 (20% vs E111's
   23%) is a hard-min/max bug fix in MCF that is INDEPENDENT of MCF's
   multi-bend distribution.** This fix could be back-ported to E111's
   `per_net_trace_proxy.py` for a similar effect — see "Spin-off" below.

### Decision: KILL

MCF as a multi-commodity-flow congestion proxy is falsified. The
distinctive multi-bend distribution gives worse calibration than RUDY,
and the only end-to-end win came from the rudy_mix=1.0 configuration
which is canonical-L-route-equivalent.

### Spin-off (independent of MCF)

The `_soft_min_max` phantom-stripe bug found while building this also
exists in `per_net_trace_proxy.py` (E111). Fixing it there would
reduce E111's calibration bias by 5-10% on hard benches. **Whether
that translates to end-to-end lift requires testing.** Not pursued here
because the ibm17 test on the equivalent (MCF rudy_mix=1.0) showed only
tied performance vs E111, so the gradient direction was already aligned
enough not to be the binding constraint on hard benches.

## Pointers

- Code: `code/mcf_congestion.py` — MCF distribution + L-route mixture
- Code: `code/calibrate.py` — smooth vs canonical on cached placements
- Code: `code/smooth_global_placer_mcf.py` — V3-shaped placer with MCF
- Code: `code/test_ibm17.py` — ibm17 + CD600s smoke
- Results: `results/calibration.json`, `results/ibm17_smoke.json`
