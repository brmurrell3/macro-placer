---
id: E92
name: dp_tilos_rudy
status: scoped
parent: PATH B B-R1 — diagnostic add TILOS-RUDY as DP loss term
created: 2026-05-12
decided: null
champion_at_time: 1.0612 (cascade uncapped, ibm)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E92: dp_tilos_rudy — add canonical-aligned RUDY as DP loss term (B-R1)

## Hypothesis

DP's basic global placement obj_fn is `wl + density_weight * density`. RUDY
exists in DP (`dreamplace.ops.rudy.Rudy`) but is used only for area
adjustment in routability_opt mode — it's *not* a gradient-contributing
loss term. The autopsy framed "DP-RUDY vs TILOS-RUDY" as a mismatch, but
the real diagnosis is finer: stock DP **doesn't optimize congestion at
all**, which is 0.5x weight in the canonical proxy.

If we add a congestion term to DP's obj_fn with TILOS-aligned semantics
(net weights, smoothing radius, top-K aggregation), DP's gradient
direction aligns better with canonical at convergence — and the polish
gap to cascade should narrow.

## Method

1. Fork `dreamplace/PlaceObj.py` in our DP install:
   - Add `build_rudy_loss_op` that wraps `rudy.Rudy` with canonical
     TILOS weights (from `macro_place._plc` — net weight 5e-4 by default).
   - Modify `obj_fn` to compute scalar congestion via
     `softmax(rudy_map / τ).flatten() · rudy_map.flatten()` (smooth
     top-K analog), or `(rudy_map - cap).relu().pow(2).sum()` (capacity
     penalty).
   - Add `congestion_weight` param (start at density_weight × 0.5 to
     match canonical 0.5x congestion weight).
2. Calibration check: at a fixed cascade placement, compute (a) our
   modified DP obj, (b) canonical proxy. They should be linearly
   correlated within ~5 %.
3. Re-run E91 B-R0' pipeline with the patched DP (stock DP → full polish
   → cascade) on ibm10 + ibm14 + ibm12.

## Kill gate

- B-R0' (E91) verdict gates this experiment. If stock DP+full-polish
  already lands ≤ cascade, B-R1 is unnecessary (already aligned enough
  via polish).
- If stock DP+full-polish is still > cascade by ≥3 %, run B-R1.
  Decision: gap closes by >half (to <1.5 %) → routing was load-bearing,
  commit to B-R3 (full canonical losses on DP). Else density is
  dominant → commit to B-R2 (top-K density replacement).

## Implementation notes

The integration point in PlaceObj.py is `obj_fn` (line 322 area). Stock
form:

    obj = wl + density_weight * density

Patched form:

    obj = wl + density_weight * density + congestion_weight * congestion
    where congestion = scalar_reduction(rudy.Rudy()(pin_pos))

`rudy.Rudy` already takes `net_weights` — TILOS-aligned weights come
from `macro_place._plc.PlacementCost` (read at startup and passed in
the bookshelf .wts file from `tilos_to_bookshelf._write_wts`). Verify
those weights survive the round-trip.

## Pointers

- DP RUDY op: `dreamplace/ops/rudy/`
- DP PlaceObj.obj_fn: `dreamplace/PlaceObj.py:322-405`
- DP build helper: `build_route_utilization_map` at line 957
- Canonical congestion: `macro_place/_plc.py` (PlacementCost.get_congestion_cost)
- TILOS .wts format: `experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py:_write_wts`
