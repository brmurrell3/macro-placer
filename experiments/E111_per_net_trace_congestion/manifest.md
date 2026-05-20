---
id: E111
name: per_net_trace_congestion
status: in_progress
parent: E110
created: 2026-05-19
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E111: per-net-trace differentiable congestion

## Hypothesis

The DPO-era smooth `_rudy_congestion` spreads each net's routing demand
**uniformly over the bbox**. The canonical `PlacementCost.get_routing`
**traces an L-route** per (source, sink) pair: H stripe at the source's
row, V stripe at the sink's column. On hard benches the bbox-uniform
diverges from canonical by 3-4× (memory: `diff_proxy_rudy_mismatch`).

Replacing the smooth congestion in E110's SmoothGlobalPlacer with a
per-net-trace differentiable proxy that closely matches canonical should
give Adam a more canonical-aligned gradient on hard benches (ibm10/12/17),
and produce a basin that is structurally closer to what the cascade
polish exploits.

## Method

1. **`code/per_net_trace_proxy.py`** — `PerNetTraceCongestion` class.
   For each (source, sink) pair: soft cell-assignment (Gaussian σ=0.5×cell),
   smooth row_min/row_max (logsumexp β=6), sigmoid in-range indicator with
   the canonical `range(lo, hi)` boundary, accumulate H demand on source
   row × range(cols), V demand on sink col × range(rows). Macro footprint
   contribution via clamp-overlap × routing_alloc. ±smooth_range box
   smoothing via separable conv with edge-count correction. Concat-V+H
   top-5% mean.

2. **`code/smooth_global_placer_v3.py`** — mirrors E110's
   SmoothGlobalPlacer. `DiffProxyV3` extends `DiffProxyV2` overriding
   `cost()` to use `PerNetTraceCongestion`. Otherwise identical
   (Adam descent → greedy_macro_legalize → return).

3. **`code/diagnose.py`** — strip components to attribute gap.
4. **`code/compare_baselines.py`** — calibrate vs canonical on cached
   cascade placement; report scalar mismatch + Δcong correlation.
5. **`code/test_ibm17.py`** — head-to-head E110 vs V3 on ibm17 with
   60s CD polish.

## Kill gate

ibm17 raw smooth basin proxy ≥ 1.50 OR ibm17 + 60s CD polish ≥ 1.40 →
kill (the new proxy doesn't help, ship Option C).

## Generalization check

Must validate on at least ibm10/12 (the other "hard" benches) and at
least one NG45 design (ariane133) before promoting. Single-bench cherry
picks are not enough (E54 falsification pattern).

## Outcome — preliminary 2026-05-19

**ibm17 SMOKE TEST PASSED, dominantly:**

| Variant | Raw proxy | + 60s CD | Wall (raw + polish) |
|---|---:|---:|---:|
| E110 SmoothGlobalPlacer (bbox-uniform) | 1.756 | 1.522 | 213 + 128 s |
| **E111 V3 (per-net trace)** | **1.304** | **1.246** | 184 + 128 s |
| Reference: Lane-4 default --all ibm17 | — | 1.324 | (full cascade) |

- Raw smooth basin **−25.7%** vs E110 (target was <1.50).
- After 60s CD polish **−18.1%** vs E110 (target was <1.40).
- After 60s CD polish ALSO BEATS the full lane-4 cascade output of 1.324
  on this bench (1.246 vs 1.324 = **−5.9%**).

**Calibration vs canonical (Δcong correlation on 16 random
perturbations of cached cascade placement):**

| Bench | canonical | bbox (E88) | trace (E111) | trace-pearson | trace-spearman |
|---|---:|---:|---:|---:|---:|
| ibm17 | 1.895 | 6.815 (+260%) | 2.167 (+14%) | +0.59 | +0.59 |
| ibm12 | 1.654 | (~+200%) | 2.046 (+24%) | TBD | TBD |
| ibm10 | 1.301 | (~+200%) | 1.548 (+19%) | TBD | TBD |

The smooth proxy now tracks canonical with absolute mismatch <25% on
hard benches (vs >200% before), and Spearman ρ ≈ 0.59 on ibm17 (vs 0.19
for bbox).

**Next steps** (in priority order):

1. Validate on ibm10 + ibm12 (other hard benches) end-to-end with CD60s.
2. Run on ariane133 (NG45) — failure on this would falsify generalization,
   per E54 lesson.
3. Compare to Lane-4 default by integrating V3 as Lane-5 in the
   stacked_periphery cascade plateau-pick.
4. If wins hold, consider --fast then --all M3 runs to confirm aggregate.

## Pointers

- `code/per_net_trace_proxy.py` — the congestion proxy
- `code/smooth_global_placer_v3.py` — placer wrapper
- `code/test_ibm17.py` — head-to-head test
- `code/diagnose.py` — component attribution
- `code/compare_baselines.py` — calibration vs canonical
- `results/ibm17_smoke.json` — raw output
- `results/ibm17_smoke.log` — full descent trace + summary

## Status of next experiment

Smoke test PASSED. Need ibm10/ibm12 + NG45 generalization before claim
of structural improvement.
