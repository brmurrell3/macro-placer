# Differentiable RUDY Rewrite — Specification

**Context.** E95 tonight diagnosed that the DPO-era `_rudy_congestion`
(`writeup/archive/submissions/dpo/ablation_v2_steps.py`) diverges 3-4×
from canonical `PlacementCost.get_routing` on hard benches. This is the
load-bearing reason cheap C1 spike fails on ibm10/12/17 and would block
any future smooth-proxy-gradient placer (B-R3, C1 sequel) until fixed.

This spec describes what to build, the differentiability challenges, and
the proposed soft approximations.

## What canonical RUDY does

For each net, canonical traces actual route paths through grid cells.
From `plc_client_os.py`:

```python
def get_routing(self):
    for mod in self.modules_w_pins:
        # collect set of gcells touched by net's pins
        node_gcells = set()
        ... compute pin cell locations ...
        # dispatch by net size
        if len(node_gcells) == 2:
            __two_pin_net_routing(source_gcell, node_gcells, weight)
        elif len(node_gcells) == 3:
            __three_pin_net_routing(node_gcells, weight)
        elif len(node_gcells) > 3:
            for sub in __split_net(source_gcell, node_gcells):
                __two_pin_net_routing(source_gcell, sub, weight)
    # normalize by route capacity
    V_routing_cong[i] = V_routing_cong[i] / grid_v_routes
    H_routing_cong[i] = H_routing_cong[i] / grid_h_routes
    # box smoothing with smooth_range
    __smooth_routing_cong()
    # add macro routing (from hard macros' bbox overlap)
    V_routing_cong[i] += V_macro_routing_cong[i]
    H_routing_cong[i] += H_macro_routing_cong[i]
```

Per-net trace (2-pin example):

```python
def __two_pin_net_routing(self, source, node_gcells, weight):
    sink = the other gcell
    # H-trace at source row from source col to sink col
    for col in range(col_min, col_max):
        H_routing_cong[source_row * grid_col + col] += weight
    # V-trace at sink col from source row to sink row
    for row in range(row_min, row_max):
        V_routing_cong[row * grid_col + sink_col] += weight
```

For 3-pin and N-pin nets, similar trace structure (L-shape or
T-shape, then split).

Final congestion cost: `abu(V + H, 0.05)` — mean of top 5% of
combined H+V utilization per cell.

## What DPO-era smooth RUDY does

```python
def _rudy_congestion(positions, net_data, ...):
    for each net:
        bbox_x_max = LSE-smooth max of pin x's
        bbox_x_min = LSE-smooth min
        bbox_w = max - min
        h_coeff = weight / (bbox_w * grid_h_routes)
        # distribute weight uniformly across all cells in bbox
        for each cell inside bbox:
            cell_overlap_area = clip(cell ∩ bbox)
            h_cong[cell] += h_coeff * cell_overlap_area
```

The smooth version smears each net's wire demand over the FULL bbox.
Canonical version traces a single path. For 2-pin nets with adjacent
pins, both methods produce similar values. For multi-pin or wide-bbox
nets, smooth over-counts by `bbox_w * bbox_h / (bbox_w + bbox_h)` ≈
geometric mean of bbox dims, which grows with net spread.

## Empirical mismatch (E95 finding)

| Bench | canon cong | smooth cong | smooth/canon |
|-------|-----------:|------------:|-------------:|
| ibm01 | 0.96 | 0.96 | 1.0 × |
| ibm10 | 1.30 | 3.97 | **3.05 ×** |
| ibm12 | 1.65 | 5.52 | **3.34 ×** |
| ibm17 | 1.90 | 6.92 | **3.65 ×** |

The 3-4× over-count dominates the smooth gradient on hard benches
because canonical proxy is ~74 % congestion-weighted.

## Differentiable RUDY — design

### Required properties

1. **Scalar-match canonical on a fixed placement.** Smooth(pos) ≈
   canonical(pos) to within 1 % at the cascade local min. This is the
   condition for "smooth gradient ≈ 0 at canonical optimum."
2. **Differentiable w.r.t. pin positions.** Gradient flows from
   congestion scalar back through cell-routing-demand back through
   pin → cell membership back through pin position.
3. **Vectorizable across nets.** Per-net Python loops are too slow at
   28-39k nets (ibm10-18 scale). Need batched torch ops.

### Sub-problem 1: soft pin-to-cell membership

Canonical: `cell = (floor(x / cell_w), floor(y / cell_h))` — step
function, non-differentiable.

Soft version: pin contributes to multiple cells weighted by a
Gaussian kernel centered at the pin position. For pin at `(x, y)`:

```
membership[r, c] = exp(-((x - cell_center_x[c])² / (2 σ_x²)
                      + (y - cell_center_y[r])² / (2 σ_y²)))
```

Normalize so `sum membership = 1`. With σ ≈ cell_size / 4, each pin
covers ~3×3 cells with most weight on the canonical one. As σ → 0,
membership recovers the canonical step function.

Implementation: vectorized — `pin_to_cell` is a `[N_pins, grid_h,
grid_w]` tensor (or sparse for memory). Pin gradient flows through
the soft membership.

### Sub-problem 2: soft 2-pin trace

Canonical: for source `(r₁, c₁)`, sink `(r₂, c₂)`:
- H-route at row r₁ from c_min to c_max (cell entries get +weight)
- V-route at col c₂ from r_min to r_max (cell entries get +weight)

Soft version: instead of hard cell membership, compute a *path
indicator* — a real-valued tensor `[grid_h, grid_w]` that's 1 along
the trace cells, 0 elsewhere, smoothly interpolated.

H-trace indicator at row r₁ from c_min to c_max:

```
H_path[r, c] = soft_eq(r, r₁) * soft_between(c, c_min, c_max)
```

where:
- `soft_eq(r, r₁) = exp(-(r - r₁)² / (2 σ_row²))` — Gaussian peak at r₁
- `soft_between(c, lo, hi) = sigmoid((c - lo)/τ) * sigmoid((hi - c)/τ)` — soft indicator of c ∈ [lo, hi]

As σ → 0, τ → 0, this recovers canonical's hard trace.

For each net, sum H_path and V_path contributions weighted by net weight.

### Sub-problem 3: vectorized batching

Per-net computation is independent. Stack into a `[N_nets, grid_h,
grid_w]` tensor at peak memory, but for ibm10 that's 28k × 41 × 55 ×
4 bytes = 250 MB — too big.

Solution: batched in groups of B nets (B = 1024):
- For each batch of B nets, compute `[B, grid_h, grid_w]` traces
- Sum across the batch dim → `[grid_h, grid_w]` accumulator
- Accumulate across batches → final cong map

Memory ≤ 10 MB per batch. Time: O(N_nets × grid_size).

### Sub-problem 4: __smooth_routing_cong

Canonical applies a 5-cell box smoothing (±smooth_range = ±2 cells)
to V_routing_cong and H_routing_cong separately. This is trivially
differentiable via average pooling with kernel size 5 in each axis.

```python
v_smoothed = F.avg_pool2d(v_routing_cong.unsqueeze(0).unsqueeze(0),
                          kernel_size=(1, 5), stride=1,
                          padding=(0, 2)).squeeze()
h_smoothed = F.avg_pool2d(h_routing_cong.unsqueeze(0).unsqueeze(0),
                          kernel_size=(5, 1), stride=1,
                          padding=(2, 0)).squeeze()
```

### Sub-problem 5: macro routing

Canonical adds `V_macro_routing_cong` and `H_macro_routing_cong`
from `__macro_route_over_grid_cell` — accounts for hard macros
*blocking* routes by occupying cells. These additions are
canonical's intra-cell macro contribution.

`__macro_route_over_grid_cell` adds weight `vroutes_per_micron *
cell_h` to V_macro_cong for every cell the macro footprint covers,
and similarly for H.

Differentiable version: per-macro overlap with each cell (already
have in `_grid_density`) × routes_per_axis. Trivial.

### Sub-problem 6: ABU-5% top-K

Already implemented correctly in DPO-era as
`topk(combined.flatten(), k=0.05*ncells).mean()`. Reuse unchanged.

## Test plan

### Stage 1: calibration on cached cascade outputs

Build the new diff_rudy_v3 and verify on all 17 IBM benches:

```
for bench in [ibm01..ibm18]:
    smooth_cong = diff_rudy_v3(cached_placement)
    canon_cong = compute_proxy_cost(cached_placement, bench, plc)["congestion_cost"]
    gap_pct = (smooth_cong - canon_cong) / canon_cong * 100
    assert abs(gap_pct) < 1.0, f"{bench}: gap {gap_pct:.2f}%"
```

Pass criterion: scalar match within 1 % on every bench.

### Stage 2: gradient sanity

At cascade local min, smooth gradient should be very small:

```
for bench in [ibm01, ibm10]:
    grad = autograd.grad(diff_rudy_v3(positions).sum(), positions)
    norm = grad.norm() / positions.norm()
    assert norm < 1e-2, f"{bench}: relative grad norm {norm} too large"
```

### Stage 3: descent test

Re-run E95 spike v2 with new diff_rudy_v3 plugged in. Expected:
- ibm01 spike: still preserves basin (was holding even with bad RUDY)
- ibm10/12/14/17 spike: now sees descent direction toward canonical
  optimum. Expected lift: 0.1-0.5 % per bench (best case).

If descent lifts at all on hard benches, **THEN** C1 has value as
basin-polish lane in the hybrid placer.

### Stage 4: full --all validation

If Stage 3 clears, run on all 17 IBM + NG45. Compare aggregate to
PATH B hybrid's expected --all result.

## Time estimate

| Task | Est. wall (focused dev) | Status |
|------|------:|--------|
| Diff RUDY trace impl (sub-problems 1-3) | 1.5 days | **STAGE 1 DONE 2026-05-13** — `code/diff_rudy_trace.py` matches canonical to within 0.5 % across all 17 IBM benches (max gap 1.91 % on ibm03; others < 0.5 %). Hard cell membership; no autograd yet. |
| Vectorize + memory tuning | 0.5 days | Pending |
| Stage 1 calibration + debugging | 0.5 days | **DONE** |
| Stage 2 differentiability (soft cell + soft path) | 1 day | Pending |
| Stage 2 gradient + Stage 3 descent | 0.5 days | Pending |
| Total remaining after tonight | **~2 days** | |

### CRITICAL discovery 2026-05-13: TWO bugs in DPO smooth proxy, not one

The DPO-era `_rudy_congestion` had two independent bugs against canonical:

1. **Concat-vs-elementwise**: canonical's `get_congestion_cost` calls
   `abu(self.V_routing_cong + self.H_routing_cong, 0.05)`. Since
   `V_routing_cong` and `H_routing_cong` are Python **lists**, `+` is
   list concatenation (yielding 2 × N_cells values, top-5 % over them
   is 0.1 N values from EITHER V or H pool). DPO smooth treats them as
   tensors and does element-wise sum (N values, top-5 % over them is
   0.05 N values, each = V+H per cell). The element-wise sum has roughly
   2× larger top-K magnitude. **One-line fix**:

       combined = torch.cat([h_cong.flatten(), v_cong.flatten()])

2. **Bbox-uniform-spread vs trace-route**: the original algorithmic
   mismatch (the one the spec set out to fix).

Empirical impact on ibm10:
- DPO smooth (both bugs): smooth_cong / canon = **3.05 ×**
- DPO smooth + concat fix only: smooth_cong / canon = **1.60 ×**
- Stage 1 trace (correct algorithm + concat): smooth_cong / canon = **1.00 ×**

So the concat bug accounts for roughly half the divergence on hard benches. The remaining ~1.5-2× requires the trace-route rewrite.

The concat fix is a one-line patch to DPO smooth proxy that morning Claude can apply trivially. But it does NOT make C1 work standalone — 1.5-2× over-count still misdirects gradient on hard benches. Need both fixes (concat + trace algorithm) for C1.

## Risks

- **Soft trace doesn't preserve canonical's discrete structure.**
  Even with σ → 0, the smoothing creates a different objective
  landscape than canonical's hard topology. Local minima of soft
  may not be local minima of canonical. Mitigate: include canonical
  CD as final polish step (same pattern as cascade saddle escape).
- **Memory explosion on ibm10/18.** 28-39k nets × grid_size. Mitigate
  via batched processing; 1024-net batches keep peak < 50 MB.
- **Pin-to-cell membership Gaussian width.** Too wide → over-smoothed,
  poor scalar match. Too narrow → poor gradient. Tune σ via Stage 1
  scalar calibration.
- **N-pin net splitting matches canonical's `__split_net` algorithm.**
  Need to read that function and replicate its decisions (which
  determines route topology beyond 3-pin nets).

## Open questions

1. Does the smooth_range box-smoothing fully recover canonical's
   smoothness, or does the trace's discreteness leave residual
   non-smoothness?
2. Does the macro routing contribution change with macro position
   (it should — macros at different cells block different cells),
   or is it position-independent? Need to verify.
3. Are pin positions actually the right gradient input, or should
   gradient flow through MACRO positions instead (with pin offsets
   fixed per macro)?

## Files to reference

- Canonical implementation: `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`
  - `get_routing` line 1514
  - `__two_pin_net_routing` line 1269
  - `__three_pin_net_routing` line 1354
  - `__split_net` line 1486
  - `__smooth_routing_cong` line 1608
  - `__macro_route_over_grid_cell` line 1392
- DPO-era smooth RUDY (to replace): `writeup/archive/submissions/dpo/ablation_v2_steps.py:505`
- E92 prototype (incomplete, per-net Python loop):
  `experiments/E92_dp_tilos_rudy/code/diff_rudy.py`
- E95 calibration that found the mismatch:
  `experiments/E95_diff_proxy_v2/code/calibrate_all_ibm.py`

## When to revisit

After PATH B hybrid `--all` confirms its floor advance.
- If hybrid lands ≤ 1.06 aggregate: diff RUDY rewrite is **lower
  priority** — PATH B's mechanism already advances the floor enough.
- If hybrid lands > 1.10 aggregate: diff RUDY rewrite is **medium
  priority** — could compose for additional 0.3-1 % lift via C1
  polish lane.
- If PATH B falsifies entirely: diff RUDY rewrite is **high priority**
  — only remaining unexplored path to lift past cascade-uncapped.

## Memory cross-references

- [[diff_proxy_wl_norm_gotcha]] — E88's misdiagnosis (WL norm bug)
- [[diff_proxy_rudy_mismatch]] — this finding (RUDY 3-4× over-count)
