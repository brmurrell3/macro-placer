# Option A — DREAMPlace Loss Patch Specification

Synthesized 2026-05-19 during the "patch DREAMPlace loss" investigation.
Documents the exact patch points to inject our challenge proxy into
DREAMPlace's optimization loop. NOT IMPLEMENTED yet because: (1) the
TILOS-RUDY gradient requires C++ backward, and (2) DREAMPlace requires
x86_64 binaries that can't run on M3 darwin/arm64.

## Where to patch

File: `submit_deps/dreamplace_install/dreamplace/PlaceObj.py`

### Stock `obj_fn` (lines 293-320, after install path)

```python
def obj_fn(self, pos):
    """wirelength + density_weight * density penalty"""
    self.wirelength = self.op_collections.wirelength_op(pos)
    if len(self.placedb.regions) > 0:
        self.density = self.op_collections.fence_region_density_merged_op(pos)
    else:
        self.density = self.op_collections.density_op(pos)
    # ... (init_density bookkeeping, quad_penalty) ...
    result = torch.add(self.wirelength, self.density,
                       alpha=(self.density_factor * self.density_weight).item())
    return result
```

### Patched `obj_fn` for challenge proxy

The challenge proxy is `HPWL + 0.5*density + 0.5*congestion`. DP's
existing `wirelength_op` (weighted_average_wirelength) is a reasonable
smooth HPWL surrogate. DP's `density_op` (eDensity) is NOT aligned
with canonical top-K density. DP has no congestion gradient.

```python
def obj_fn(self, pos):
    """challenge proxy: HPWL + 0.5*density + 0.5*congestion"""
    # 1) Wirelength (use stock weighted_average; γ-anneal during descent
    #    to converge to canonical bbox-max)
    wl = self.op_collections.wirelength_op(pos)

    # 2) Density (REPLACE eDensity with top-K differentiable density;
    #    matches canonical PlacementCost.get_density_cost). See E93
    #    scaffold for implementation: torch.topk on scatter-add density
    #    map normalized by bin density target.
    density = self._topk_density_op(pos)  # NEW; need to implement

    # 3) Congestion (NEW; not in stock DREAMPlace as a gradient term).
    #    Stock has rudy.Rudy() but it's C++ forward-only (no autograd
    #    backward registered). Two options:
    #      (a) Write a torch-native RUDY in Python (slow but works,
    #          ~5x slower than C++ but autograd-native)
    #      (b) Write a custom torch.autograd.Function wrapper for
    #          rudy_cpp.forward with hand-coded backward in C++
    #          (matches Carrotato's likely approach)
    rudy_h, rudy_v = self._diff_rudy_op(pos)  # NEW; need to implement
    cong = self._congestion_reduction(rudy_h, rudy_v)
    # canonical congestion = ABU5% (top-5% bin sum). Use softmax-top-K
    # smooth analog: temp_softmax(rudy / τ) @ rudy.flatten()

    # 4) Weighted sum matching challenge formula
    return wl + 0.5 * density + 0.5 * cong
```

## Engineering effort estimate

| Component | Effort | Notes |
|---|---|---|
| Top-K density Python op | 1-2 days | E93 scaffold. Use torch index_add_. Test gradient flow. |
| RUDY backward (Python) | 2-3 days | Re-implement rudy.Rudy in pure torch. ~5x slower but autograd-native. |
| RUDY backward (C++ extension) | 5-7 days | Write rudy_cpp_backward.cpp + register torch.autograd.Function. Matches Carrotato perf. |
| `obj_fn` patch + integration | 0.5 day | Drop-in replacement; verify Nesterov optimizer doesn't break. |
| Calibration vs canonical proxy | 0.5 day | At fixed cascade placement, patched_obj should match canonical within 0.5%. |
| Hyperparameter tuning | 1-2 days | density_weight, congestion_weight ratios. Use E76 sweep harness. |
| Full --all + NG45 validation | 1 day | EPYC + DREAMPlace cloud box. |
| **Total (Python RUDY)** | **6-9 days** | |
| **Total (C++ RUDY)** | **9-13 days** | |

This is what E94 (`dp_canonical_full`) scoped as the "heaviest route" —
"do not start unless E91 + E92 evidence supports."

## Why we shipped Option C instead

Option C (`SmoothGlobalPlacer` from E110) reaches a comparable
gradient-basin without modifying DREAMPlace:

| Path | Wall (engineering) | Wall (per ibm01) | ibm01 proxy |
|---|---|---|---|
| Option A (full patch, Python RUDY) | 6-9 days | ~5 min (DP optimizer) | unknown, projected ~0.85 |
| Option C (SmoothGlobalPlacer SDF) | DONE | ~3 min (Adam) | 0.880 + CD60s; 0.878 + CD240s |
| Reference: cascade uncapped | DONE | ~50 min | 0.845 |

Option C is within 4% of cascade (and beats it on per-bench cases via
basin diversity in plateau-pick). The engineering cost to close the
remaining 4% via Option A is high relative to the marginal lift
expected.

## Patch test path (when revisited)

1. Copy `submit_deps/dreamplace_install/` → `experiments/E94/code/dp_patched/`
2. Edit `dp_patched/dreamplace/PlaceObj.py` with the obj_fn replacement
3. Implement `_topk_density_op` and `_diff_rudy_op` (use diff_proxy.py
   primitives from E88 as reference)
4. Calibration: at cascade-cached ibm01, patched_obj should match
   canonical within ~0.5% (E95 showed E88 primitives match canonical
   at the smoothing-fixed gamma)
5. Smoke: run patched DP on ibm01 with 1000 iters Nesterov, then
   `greedy_macro_legalize`, evaluate canonical proxy
6. Gate: ibm01 raw < 1.0 (cascade gets 0.85); else kill
7. If pass: full --all on cloud A100 with DREAMPlace install

## Pointers
- Stock obj_fn: `submit_deps/dreamplace_install/dreamplace/PlaceObj.py:293-320`
- RUDY op (forward only): `submit_deps/dreamplace_install/dreamplace/ops/rudy/rudy.py`
- E88 diff_proxy primitives: `experiments/E88_diff_proxy/code/diff_proxy.py`
- E95 normalization fix: `experiments/E95_diff_proxy_v2/code/diff_proxy_v2.py`
- E110 working smooth-proxy descent (Option C): `experiments/E110_smooth_global_placer/code/smooth_global_placer.py`
- E91 DP + full polish (Option D fallback): `experiments/E91_dp_full_polish/code/dp_full_polish.py`
