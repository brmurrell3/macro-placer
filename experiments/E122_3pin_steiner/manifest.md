---
id: E122
name: 3pin_steiner
status: marginal
parent: E111
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.0575
outcome: ibm17 E111 baseline 1.2003 → E122 1.2121 (+0.98% WORSE, despite +6.4pp scalar bias closure). Fidelity vs basin disconnect.
champion_delta: +0.0118
graduated_to: null
superseded_by: null
---

# E122: 3-pin Steiner T-route — surgical fix on E111

## Hypothesis

E111's per-net-trace congestion treats every net as a star of L-routes
(pin0 → each other pin). For 3-pin nets, canonical's
`PlacementCost.__three_pin_net_routing` (plc_client_os.py:1354-1390)
instead computes a **Steiner shape** (L or T) which RE-USES the middle
pin's row/col as a routing trunk — substantially less demand than two
star L-routes.

Per the 2026-05-20 literature survey
(`docs/research/2026-05-20_congestion_model_survey.md` §3 IMPROVEMENT 1),
25-40% of IBM benchmark nets are 3-pin, and this topology mismatch is the
**#1 source** of our remaining 14-24% canonical bias on hard benches.

Adding a soft T-route Steiner contribution for 3-pin nets (the "Simpler
alternative" in the survey: T-route only, no 4-case split) should close
the canonical bias from +14% to +8-10% on ibm17, with downstream ibm17 +
CD600s polish improving from 1.20 to ≤ 1.18.

## Method

1. **`code/three_pin_steiner.py`** — `PerNetTraceCongestionSteiner3Pin`,
   a subclass of E111's `PerNetTraceCongestion`. At init:
   - Re-extract `NetData`, split nets by `len(pins)` into {2, 3, 4+}.
   - Save 3-pin nets as `(macro_a/b/c, off_a/b/c, weight)` tensors.
   - Rebuild parent's pair lists EXCLUDING 3-pin nets (so star routing
     doesn't double-count).

   Per Adam-step:
   - Inherited `_trace_route_congestion` handles 2-pin + 4+pin star
     L-routes (unchanged from E111).
   - Override adds 3-pin Steiner via soft median selection on x-coords:
     * `α_left = softmax(-β·x, dim=1)`, `α_right = softmax(+β·x, dim=1)`,
       `α_mid = (1 - α_left - α_right)+`.
     * `x_left = Σ α_left·x`, `x_right = Σ α_right·x`, `y_mid = Σ α_mid·y`.
     * H trunk at `y_mid` over `[col_left, col_right]`.
     * V branch at `col_left` over `[min(row_left, row_mid), max(...)]`.
     * V branch at `col_right` over `[min(row_mid, row_right), max(...)]`.
   - Sum 3-pin contribution into V_route, H_route. Rest of pipeline
     (macro footprint + box smoothing + top-K) is unchanged.

2. **Calibration test** (in `three_pin_steiner.py::calibrate`): compares
   E111 base vs E122 Steiner vs canonical on ibm17 cached cascade
   placement; reports scalar mismatch + 16-perturb Pearson/Spearman.

3. **Production pipeline test**
   (`submissions/e111_minimal_steiner3pin/placer.py`): swaps
   E111's PerNetTraceCongestion for the Steiner version in
   `SmoothGlobalPlacerV3`, runs Adam descent + greedy_macro_legalize + CD
   polish, target ibm17 < 1.18 at 720s budget.

## Kill gate

- **Calibration**: smooth-canonical bias on ibm17 cached cascade
  placement >= 12% (currently ~14%) → fix is not closing the
  topology gap as expected.
- **End-to-end**: ibm17 + V3+CD600s ≥ 1.20 (current baseline) →
  the proxy fidelity gain does not translate to placement quality.
- If either kill gate fires AND Pearson/Spearman ρ improves: marginal
  status (fidelity gain ≠ basin quality gain).

## Generalization check

Before any promotion: validate on ibm10 + ibm12 (the other "hard" benches
in the survey), confirm Steiner bias < base bias on both, no regression
on ibm17 V3+CD pipeline. If only one of three improves: marginal.

## Outcome (decided 2026-05-20)

**MARGINAL** — fidelity gain confirmed, basin quality does NOT improve.

### Calibration: bias closure achieved (target hit)

| Bench | E111 base | E122 Steiner | Closure | 3-pin% |
|-------|---------:|---------:|--------:|------:|
| ibm01 | +11.0% | +6.9% | +4.1 pp | 21.7% |
| ibm10 | +18.9% | +12.9% | +6.1 pp | 26.2% |
| ibm12 | +23.7% | +18.3% | +5.3 pp | 28.5% |
| ibm17 | +14.4% | +8.0% | **+6.4 pp** | 36.9% |

ibm17 closure (+14% → +8%) is exactly at the bottom of the survey's
predicted "+8-10% target" range. On SDF-init placement the closure is
even larger (ibm17: +7.7% → +0.8%, +6.9 pp; ibm01: +4.6% → +0.3%,
+4.3 pp). The implementation works as designed.

### End-to-end: basin quality fails (kill gate breached)

| Pipeline | ibm17 proxy | wall |
|----------|------------:|-----:|
| E111 baseline (ovl10_720s) | **1.2003** | ~720s |
| E122 V3Steiner ovl10_720s | **1.2121** | 828s |
| Δ | **+0.0118 (+0.98% WORSE)** | |

Kill gate `ibm17 + CD600s ≥ 1.20` is breached: E122 gives 1.2121 > 1.20.

### Diagnosis: fidelity vs gradient quality

The Steiner T-route gives a more accurate canonical-matching SCALAR, but
when used as the Adam descent objective, it produces a less useful
GRADIENT. Specifically:

- Steiner counts ~30-50% less demand per 3-pin net than the star L-route
  (T-route shares the middle pin's row/col as a trunk, vs star which
  routes each pin pair independently with full L-route).
- This means Adam sees a LOWER smooth congestion at the same placement
  vs E111 — so Adam stops pushing as hard on 3-pin net positions.
- Result: Adam converges to a basin where canonical congestion is
  similar BUT canonical density/WL are slightly worse, because the
  reduced "pressure" let other parts of the gradient field drift.

Survey IMPROVEMENT 2 §3 already flagged this risk:
> caveat: This *moves* away from canonical L-only and toward Westra's
> probabilistic model. It is NOT canonical-faithful. Canonical always
> picks exactly the source.row + sink.col L … Worth a smoke test on
> ibm17: if E111-V3 + Westra → CD-60s polish ≤ E111-V3-L-only +
> CD-60s polish, ship the change.

E122 is the analog of this risk for L-vs-Steiner. The risk has materialized.

### Decision: KEEP AS RESEARCH ARTIFACT, DO NOT GRADUATE

Per the survey's "gradient quality is the lever" — the Steiner produces
a smoother SCALAR but a less aligned GRADIENT. To capture the scalar
benefit without harming the gradient, one would need to either:

1. Use E122 as a 2nd-stage refinement: descend with E111 (L-only) for
   most of the trajectory, then a few late-stage steps with E122 to
   fine-tune. (Not implemented here.)
2. Use Steiner as a regularizer with small weight on the main objective.
   (Not implemented here.)
3. Use the full 4-case canonical splitting (per IMPROVEMENT 1's
   "case-split full" variant) which should preserve gradient direction
   better. (Not implemented — would be M-effort.)

These extensions are listed in `notes/next_steps.md` but are NOT being
pursued for the May 21 deadline. The headline finding is that the
**+6.4 pp calibration closure does not translate into placement
quality on the current single-stage Adam pipeline**.

### Generalization not tested

Only ibm17 was tested end-to-end (kill gate failed before broader
sweep). Per E54 lesson, we did NOT test on NG45 — but the negative
result on the hardest single-bench is enough to falsify the hypothesis
that scalar fidelity → basin quality.

## Pointers

- Code: `code/three_pin_steiner.py`
- Production: `submissions/e111_minimal_steiner3pin/placer.py`
- Survey: `docs/research/2026-05-20_congestion_model_survey.md` §3 IMPR. 1
- Canonical reference: `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py:1354-1390`
- Results: `results/calibration.json`, `results/ibm17_smoke.log`
