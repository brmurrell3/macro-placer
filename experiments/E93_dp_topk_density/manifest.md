---
id: E93
name: dp_topk_density
status: scoped
parent: PATH B B-R2 — replace eDensity with differentiable top-K density
created: 2026-05-12
decided: null
champion_at_time: 1.0612 (cascade uncapped, ibm)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E93: dp_topk_density — replace DP's eDensity with canonical top-K density (B-R2)

## Hypothesis

DP's `electric_potential.ElectricPotential` (eDensity) penalizes
non-uniform spread via electrostatic field (FFT-Poisson). Canonical
density (`PlacementCost.get_density_cost`) is **top-K density** — sum
of the K highest bin densities, no penalty for under-utilized bins.

For our problem (200–537 macros, very sparse vs std-cell counts), the
"uniform spread" objective is structurally misaligned: we don't want
to spread macros uniformly, we want to avoid any bin exceeding
threshold. Replacing eDensity with differentiable top-K aligns DP's
gradient with the canonical density surface.

## Method

1. Fork `dreamplace/PlaceObj.py`:
   - Add `build_topk_density_op` that computes a 2D density map from
     cell positions (torch-native, differentiable via standard
     scatter-add), then reduces via
     `topk_density = softmax(map / τ) * map` summed over top-K bins.
   - Modify `obj_fn` to use this in place of (or alongside) the
     eDensity term.
2. Density-map computation: per-cell rect → bins overlapping rect, area
   contribution to each bin = cell_w * cell_h * frac_overlap. Use
   torch `index_add_` for accumulation.
3. Test that gradients flow from top-K loss back to cell positions.
4. Calibration check: at fixed placement, our top-K density vs canonical
   `get_density_cost` should agree within ~3 % across a range of
   placements.
5. Re-run E91 pipeline with the patched DP. Compare to E91 (stock-DP)
   and cascade.

## Kill gate

Gated on E91 outcome. Only commit if E91 shows stock-DP + full-polish
remains > cascade by ≥3 % *and* B-R1 (E92) shows routing alone
doesn't close the gap.

## Implementation notes

- DP's eDensity is deeply integrated (FFT, Poisson solver, area-adjust
  loops). Replacing it cleanly may require adding a new "density_mode"
  param to obj_fn rather than ripping out eDensity.
- The density gradient via eDensity is *long-range repulsive* (anyone
  overcrowded pushes everyone). Top-K gives *short-range repulsive*
  (only top-K bins push their occupants). Convergence dynamics may
  differ substantially — start with lower density_weight.

## Pointers

- DP eDensity op: `dreamplace/ops/electric_potential/`
- DP obj_fn: `dreamplace/PlaceObj.py:322-405`
- Canonical density: `macro_place/_plc.py::PlacementCost.get_density_cost`
- Bookshelf-format density: not used by canonical (canonical uses .pb.txt)
