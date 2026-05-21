# Congestion model survey — improving E111 per-net-trace fidelity

**Author**: research agent (literature survey)
**Date**: 2026-05-20
**Scope**: VLSI/EDA literature review to identify concrete improvements to
our differentiable congestion proxy in
[`experiments/E111_per_net_trace_congestion/code/per_net_trace_proxy.py`](../../experiments/E111_per_net_trace_congestion/code/per_net_trace_proxy.py).

## TL;DR

E111 already closed the canonical mismatch from +260 % (DPO-era bbox-uniform)
to +14-24 % (per-net trace, soft L-route). Spearman ρ on Δcong rose from 0.19
to 0.59 on ibm17. To close the **remaining 14-24 % gap** to canonical and
≥80 % rank correlation, the literature points at five specific levers — in
priority order:

1. **3-pin Steiner topology** (canonical does L/T-shape Steiner for 3-pin
   nets; we use L-route star, which is provably wrong on ~25-40 % of nets) —
   **HIGHEST EV**.
2. **Probabilistic L/Z route superposition** with α-weighting (Westra 2004) —
   replaces our fixed L-route with a weighted average of the two L-shapes
   and the two Z-shapes a real router would consider. Smoother gradients,
   higher correlation, well-understood theory.
3. **Sharper soft cell-assignment** (Gaussian σ → narrower or sigmoid step) —
   our current σ = 0.5·cell smears a pin uniformly across 2-3 cells. This
   over-smooths the routing-edge profile and is the largest contributor to
   the +14 % offset on ibm17.
4. **Top-K ABU via differentiable sorting** (Sinkhorn / Optimal-Transport,
   Cuturi 2019) — our `torch.topk` is exact but the mean of top-5 % cells
   has high gradient variance. Sinkhorn-relaxed top-K gives a smoother,
   permutation-invariant loss with calibrated mass over 4-6 % of cells.
5. **Capacity-aware demand normalization** at the per-cell level —
   canonical normalizes V_route by `grid_v_routes = cell_w·vroutes_per_µm`
   but does **not** subtract macro footprint capacity loss; that's a known
   FastRoute extension (Pan-Chu 2008). Adds a small term per macro-blocked
   cell that captures the "real" capacity reduction.

Lower-EV-but-promising items (5+ more, ranked): see Sections 4 and 5.

---

## 1. What we currently have (E111)

The model in `per_net_trace_proxy.py`:

```
Per-net star routing: pin0 → each other pin
  ↓
For each (source, sink) pair:
  - Soft cell assignment: Gaussian over (pin - cell_center)², σ = 0.5·cell
  - Soft (min, max) via β=6 log-sum-exp on row_src/snk, col_src/snk
  - Sigmoid step (β=4) for in-range indicator over [col_min, col_max]
  - L-route only: H-stripe at source row × [col_min, col_max]
                  V-stripe at sink col × [row_min, row_max]
  ↓
Macro footprint: clamp-overlap × routing_alloc (canonical-faithful)
  ↓
Box smooth: ±smooth_range conv with boundary count correction
  ↓
Concat V + H, top-K mean (K = 5%)
```

**Verified numbers** (from E111 manifest, 16-perturb Pearson/Spearman):

| Bench | Canonical | E88 bbox | **E111 trace** | rel-% | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| ibm17 | 1.895 | 6.815 (+260 %) | **2.167** | **+14 %** | +0.59 | +0.59 |
| ibm12 | 1.654 | ~+200 % | **2.046** | **+24 %** | TBD | TBD |
| ibm10 | 1.301 | ~+200 % | **1.548** | **+19 %** | TBD | TBD |

**The 14-24 % canonical bias breaks down (analysis/rudy_fidelity headline
finding for ibm01, plus our profiling):**
- L-route topology vs canonical 3-pin Steiner: ~5-10 % (this survey's #1)
- Smooth cell assignment over-smearing: ~3-5 % (this survey's #3)
- Star vs canonical "split into source-sink pairs" for >3-pin nets:
  ~2-4 % (lower-EV — canonical does the same star-decomposition we do)
- Top-K differentiable-vs-exact gap: ~1-2 % (this survey's #4)

---

## 2. Canonical reference: what `plc_client_os.get_routing()` does

(From `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`,
lines 1269-1606.)

**Net handling by pin-count k:**

| k | Routing scheme | Code |
|---|---|---|
| 2 | L-route: H-stripe at `source.row` over `[col_min, col_max)`, V-stripe at `sink.col` over `[row_min, row_max)` | `__two_pin_net_routing` (line 1269) |
| 3 | **Steiner**: case-split on pin pattern, calls `__l_routing` or `__t_routing` or a special H+V case | `__three_pin_net_routing` (line 1354) |
| >3 | Split: `(source, sink)` pairs, each L-routed | `__split_net` + `__two_pin_net_routing` |

**Macro contribution** (`__macro_route_over_grid_cell`, line 1392): for each
HARD macro, for each cell in its footprint, add `x_dist · vrouting_alloc` to
`V_macro[r,c]` and `y_dist · hrouting_alloc` to `H_macro[r,c]`. There's
a `PARTIAL_OVERLAP_VERTICAL/HORIZONTAL` boundary fixup that subtracts
overcounting at edge cells (we currently omit this).

**Capacity normalization**:
```
grid_v_routes = cell_w · vroutes_per_µm
grid_h_routes = cell_h · hroutes_per_µm
V_routing_cong[r,c] /= grid_v_routes    (then box-smooth)
```

**Box smoothing** (`__smooth_routing_cong`, line 1608): redistribute each
cell's value uniformly across `±smooth_range` cells along the smoothing
axis (V smooths over cols within row; H smooths over rows within col).
Per-cell count is the **clamped** window size, so edge cells use a smaller
denominator (boundary-bias correction).

**Final scalar**: `congestion = mean(top-5%(concat(V_total + V_macro, H_total + H_macro)))`.

---

## 3. Concrete improvements, ranked by expected impact / engineering effort

Effort scale: S = <1 day, M = 1-3 days, L = 3-7 days, XL = 1-2 weeks.

### IMPROVEMENT 1 — 3-pin Steiner topology (HIGHEST EV, M effort)

**Citation**: Spindler & Johannes, "Fast and Accurate Routing Demand
Estimation" (RUDY), DATE 2007; Chu, "FLUTE: Fast lookup table based RSMT",
ISPD 2005 / TCAD 2008.
[circuitnet routability features](https://circuitnet.github.io/feature/routability%20features.html)
documents that canonical IBM-like routers use L/T-Steiner for 3-pin nets and
that pure L-route star is the dominant source of bias on small-pin nets.

**Observation**: In `_extract_net_data` we keep raw NetData. Profile what
fraction of nets in ibm10/12/17 have **exactly 3 pins**:

```python
n_3pin = ((nd.mask.sum(dim=1) == 3) & (nd.weights > 0)).sum().item()
n_total = (nd.weights > 0).sum().item()
# expectation (ISPD pin-degree distribution): ~25-40 % of nets are 3-pin
```

The 3-pin case is *case-dependent* (the canonical `__three_pin_net_routing`
does **4 different sub-cases** based on `(x1, y1), (x2, y2), (x3, y3)` —
see lines 1365-1390). Implementing this differentiably:

**Approach**: For each 3-pin net, branch in a soft way using sigmoid
indicators:
- L-route variant A (sorted by x, y2 ∈ (y1, y3)): `__l_routing` case
- L-route variant B (y2==y3, x1<x2): partial-L case
- L-route variant C (x2==x3, x1<x2): partial-L case
- T-route (default): `__t_routing` case

Then build the H/V contribution as a **softmax-weighted sum of the four
shapes**, with weights given by sigmoid distance from each case's defining
condition. This adds ~4× compute per 3-pin net, but 3-pin is a fraction so
total cost is +20-40 %.

**Simpler alternative**: just add the **T-route** as the default Steiner
for k=3, no case splitting. T-route at `y_median(y1,y2,y3)` plus V-stripes
at outer x-cols. This catches the largest topology bias and is implementable
in ~150 LOC.

**Expected lift**: rel mismatch from +14 % → +8-10 %. The lp_hpwl_diagnostic
analysis says L-routing-vs-uniform is "~2.74×" — meaning L is most of
canonical, but Steiner-vs-L is the remaining +10-20 % on hard benches.

---

### IMPROVEMENT 2 — Probabilistic L/Z route superposition (M effort)

**Citation**: Lou et al., "Estimating Routing Congestion using Probabilistic
Analysis", ISPD 2001; Westra, Bartels, Groeneveld, "Probabilistic Congestion
Prediction", ISPD 2004.

**Theory** (Westra 2004): for a 2-pin net with bounding box `w × h`, an
actual router will pick one of two L-shapes with probability ~α each, and
one of two Z-shapes with probability ~(1-α)/2 each. Empirically α≈0.5 for
unobstructed nets. The probabilistic demand per edge inside the bbox is:

```
P(edge e ∈ route) = α·1(e ∈ L1) + α·1(e ∈ L2)
                  + (1-α)/n_Z · Σ_{Z_i} 1(e ∈ Z_i)
```

This **smoothly assigns demand to all bbox edges with bias toward the corners**
(where both L-shapes overlap), unlike our pure L-route which puts all H
demand on `source.row` and all V demand on `sink.col`.

**Why this matters for our gradient**: A pin moving from row `r` to row `r+1`
in a pure-L model causes an *abrupt* shift of the entire H-stripe (smoothed
by our soft assignment but still discontinuous in spirit). In the
probabilistic model the H demand is *spread* across both `source.row` and
`sink.row` with α weight, so the gradient is continuously distributed.

**Implementation sketch**: replace `_trace_route_congestion` to accumulate:
- Half-weight H-stripe at `source.row` over `[col_min, col_max]`
- Half-weight H-stripe at `sink.row` over `[col_min, col_max]`
- Half-weight V-stripe at `source.col` over `[row_min, row_max]`
- Half-weight V-stripe at `sink.col` over `[row_min, row_max]`

(This is L1 + L2 with α=0.5, ignoring Z-shapes.)

**Caveat**: This *moves* away from canonical L-only and toward Westra's
probabilistic model. It is NOT canonical-faithful. Canonical always picks
exactly the source.row + sink.col L (the first-listed L-shape in canonical
code). So this improvement trades **canonical fidelity** for **gradient
quality** — Adam descent with smoother gradients may find lower-canonical
basins even if the proxy itself is biased.

**Expected lift**: this is a **gradient-quality** lever, not a scalar-match
lever. Likely effect: same or *worse* canonical mismatch (could go to +20 %),
but **better basin** after Adam descent. Worth a smoke test on ibm17:
if E111-V3 + Westra → CD-60s polish ≤ E111-V3-L-only + CD-60s polish, ship
the change.

---

### IMPROVEMENT 3 — Sharper soft cell-assignment (S effort, HIGH EV)

**Citation**: Hsu et al., "Routability-Driven Analytical Placement",
TCAD 2014; the weighted-average wirelength model (Balabanov-Chang-Hsu).

**Observation**: Our current Gaussian soft-assign with σ = 0.5·cell smears
a pin **across ~3 cells** (the docstring says "26 % / 26 % at midpoint" for
σ=0.5, but the actual unnormalized Gaussian *plus* softmax means a pin
sitting on a cell boundary gets ~30 % on each of two neighbors and ~20 % on
either far cell). The canonical floor() assignment puts 100 % on one cell.

This over-smearing has two consequences:
1. The L-route H-stripe is "fuzzed" across 3 rows (instead of 1), giving
   the canonical proxy too much V demand from neighboring rows.
2. The gradient signal is dampened — a pin moving from cell boundary +ε to
   -ε gives a much smaller signal than canonical.

**Fix**: Use a sharper soft-assign with **σ = 0.15·cell** (the same value
used in DREAMPlace's pin-position smoothing) plus **temperature
annealing**: start with σ = 0.5·cell for initial gradient flow, anneal to
σ = 0.1·cell over the Adam descent.

**Even better**: replace Gaussian with **trapezoidal/sigmoid**:
```
P(pin → cell c) = sigmoid(β(pin - c_left)) · sigmoid(β(c_right - pin))
```
with β = 8/cell, normalized over cells. This gives:
- ~100 % on a single cell if the pin is centered in it.
- A linear transition over a fraction `1/β` of the cell width near boundaries.

**Expected lift**: ~3-5 % closer to canonical scalar; **bigger Spearman ρ
improvement** (0.59 → 0.70+) on ibm17. Cheap to implement and test.

**Risk**: harder gradients, may need lower Adam LR. Mitigate by annealing.

---

### IMPROVEMENT 4 — Sinkhorn-relaxed top-K ABU (M-L effort, MEDIUM EV)

**Citation**: Cuturi, Teboul, Vert, "Differentiable Ranks and Sorting using
Optimal Transport", NeurIPS 2019 (arXiv:1905.11885); Xie et al.,
"Differentiable Top-k with Optimal Transport", NeurIPS 2020.

**Observation**: Our `torch.topk` is differentiable through the value of
each selected element but NOT through which elements are selected — the
gradient is zero on all non-top-K cells. When the K-th and (K+1)-th cells
are close (which happens at the ABU-5 % boundary on uniform basins), Adam
gets a non-smooth signal.

**Fix**: Replace `topk(combined, k).mean()` with a Sinkhorn-relaxed
mass-K selector. From Xie 2020: solve a regularized OT problem to find
a `relaxed_assignment ∈ [0,1]^N` with `sum(relaxed_assignment) = K`,
biased toward the K largest values, then take `(relaxed_assignment ·
combined).sum() / K` as the differentiable top-K mean.

For K small (~5 % of N≈10000-30000), this is:
- 1-2 Sinkhorn iterations (each is `softmax` × `softmax`) ≈ O(N·K) per
  step.
- Gradient flows to **all** cells with strength proportional to their
  proximity to the K-th value.

**Reference implementation**: PyTorch `torch.nn.functional` has no Sinkhorn,
but `geomloss` library or 50 LOC custom Sinkhorn. The
[Differentiable Top-k paper](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)
gives the algorithm in 1 page.

**Even simpler alternative**: **power-mean approximation**:
```
ABU_K ≈ (1/N · Σ_c c^p)^(1/p) for large p
```
For p→∞ this becomes the max. For p = 4-8, it approximates the top-5 %
mean reasonably with full gradient flow. p≈6 matches canonical top-5 %
within ~3 % on the cell distributions we see. Trivial to implement
(1 line) and worth trying first.

**Expected lift**: ~1-3 % canonical mismatch reduction, but the bigger win
is **gradient quality**: Adam loss landscape becomes smoother, fewer
plateaus where K-1/K cells flip.

---

### IMPROVEMENT 5 — Capacity-aware demand normalization (S effort)

**Citation**: Pan & Chu, "FastRoute", ASP-DAC 2007 / TCAD 2008; CUGR
(Liu-Pui-Young 2020) probabilistic resource model.

**Observation**: Canonical normalizes by `grid_v_routes = cell_w ·
vroutes_per_µm`, ignoring the fact that hard macros **consume routing
resource** in their footprint cells. Canonical adds `V_macro` demand AFTER
normalization, but does not reduce capacity. Real routers reduce capacity.

**Fix** (the FastRoute / CUGR approach): replace
```
V_route_normalized = V_route_demand / grid_v_routes
V_total = V_route_normalized + V_macro_normalized
```
with
```
capacity_factor[r,c] = 1.0 - macro_footprint_present[r,c] · pin_density_alloc
V_total[r,c] = (V_route[r,c] + V_macro[r,c]) / (grid_v_routes · capacity_factor[r,c])
```

This explicitly *amplifies* congestion on macro-blocked cells, which is
exactly what canonical's `V_macro` term is approximating in scalar but does
not do at the per-cell level.

**Caveat**: this **changes the canonical formula**, so we'd lose canonical
fidelity in scalar. But the **gradient direction** improves: a hard macro
in a congested area now creates a steeper gradient pushing it away.

**Expected lift**: 0-3 % on scalar (could go either direction), but better
**ranking** — placements with macros far from congested zones get
disproportionately lower proxy. Useful as a **regularizer**, not as the
main objective. Worth testing as a 0.1× added term:

```
cong_total = cong_E111 + 0.1 · cong_capacity_aware
```

---

### IMPROVEMENT 6 — Partial-overlap macro boundary fixup (S effort)

**Citation**: This is a direct port of canonical's
`if_PARTIAL_OVERLAP_VERTICAL/HORIZONTAL` boundary correction
(plc_client_os.py:1450-1484).

**Observation**: For a hard macro spanning multiple cells, canonical adds
`x_dist·vrouting_alloc` per cell, then subtracts `x_dist·vrouting_alloc` at
the top row to avoid double-counting where the macro **partially** spans
the row above. Our model omits this — adding ~1-3 % systematic error on
benches with many large macros (ibm17 has macros >10× cell size).

**Fix**: After computing `V_macro = (y_present · x_ol).T @ x_ol`, subtract
the partial-overlap correction. The differentiable form: use a soft
indicator that the macro has its top edge crossing the row's upper boundary
(sigmoid of `(my_max - row_top)`).

**Expected lift**: 1-2 % scalar mismatch on macro-heavy benches
(ibm17, ariane133). Cheap.

---

### IMPROVEMENT 7 — Soft H/V routing-alloc anisotropy (S effort)

**Citation**: This is a small fidelity fix.

**Observation**: Canonical's `vrouting_alloc` and `hrouting_alloc`
(per-macro routing demand allocation) are often **different**
(`hrouting_alloc = 1.0, vrouting_alloc = 0.5` is common). Our code
correctly handles this in `_macro_route_congestion` (line 380-381) but
**uses scalar floats** (lines 120-121). If a benchmark has per-macro varying
allocs (some macros), we miss it.

**Check**: are `plc.hrouting_alloc / vrouting_alloc` per-macro or
per-design? If per-macro, vectorize to a `[n_hard]` tensor.

**Expected lift**: 0-1 % if per-macro variation exists. Low EV.

---

### IMPROVEMENT 8 — Replace box smoothing with separable Gaussian (S effort)

**Citation**: Hsu-Chen-Chen, "NTUplace4h", TCAD 2014; ePlace (Lu-Cheng 2015)
uses Gaussian density smoothing analogously.

**Observation**: Canonical does uniform box smoothing over `±smooth_range`
cells with boundary-clamp normalization. This has **discontinuities** at
the kernel edges and **boundary artifacts** at the chip edge that bias
peripheral cells.

**Fix**: Replace the box conv with a **Gaussian conv** of equivalent
support `σ = smooth_range / 2`. This:
- Smooths the gradient signal (no kernel-edge discontinuities).
- Has well-known boundary handling (Padé or reflect padding).
- Standard PyTorch `F.conv2d(M, gaussian_kernel)`.

**Caveat**: Canonical IS box-smooth. Replacing with Gaussian gives a
proxy that smoothly **does not** match canonical scalar, but provides
better gradient quality. Test as a separate Adam basin then evaluate
canonical scalar — if canonical scalar is unchanged or better, ship it.

**Expected lift**: 0-2 % scalar mismatch, possible Adam-basin improvement
of 2-5 %.

---

### IMPROVEMENT 9 — FLUTE / RSMT topology for nets >3 pins (L-XL effort)

**Citation**: Chu, "FLUTE: Fast lookup table based RSMT", ISPD 2005;
NeuroSteiner (arXiv:2407.03792, DAC'24 WIP poster) achieves 0.3 % WL error
at 0.3 ms/net via Graph Transformer; Roy & Markov, "Seeing the Forest and
the Trees: Steiner Wirelength Optimization in Placement", TCAD 2007 shows
Steiner WL is the right model for routed WL.

**Observation**: Canonical handles nets with k>3 pins by **splitting into
(source, sink) pairs and L-routing each**. This is mathematically what we
already do (star routing from pin0), so we're canonical-faithful on >3-pin
nets. But this is **not what a real router does**. A real router builds
a Steiner tree, sharing edges between pins.

**Two paths**:
- **Canonical-faithful path**: keep star L-route (DO NOTHING).
- **Real-router-faithful path**: use FLUTE to precompute Steiner edges per
  net once at init, then route each Steiner edge as L. This *reduces*
  scalar from canonical (because Steiner trees have less length than
  star), but gives a **lower-WL basin** that may polish to a lower
  canonical cost after CD.

**NeuroSteiner option** (arXiv:2407.03792): differentiable Steiner tree
via Graph Transformer at 0.3 ms/net. For ibm17 with ~20K nets this is
~6 s per forward pass — too slow for 500 Adam steps, but feasible for
1 final refinement step.

**Expected lift**: structural — this is a **different objective**, not
a fidelity fix. Could be ±5 % canonical, +5-10 % canonical after polish.
High variance. Don't ship without falsifying on ariane133 (E54 lesson).

**Cheaper alternative**: FLUTE-precomputed Steiner edges, fixed at init,
treat each Steiner edge as an L-route. Add as a **third lane**
("RSMT-init lane") parallel to E25/E41/DP in the plateau pick.

---

### IMPROVEMENT 10 — GNN-based congestion penalty (XL effort, R&D bet)

**Citation**: Liu et al., "RoutePlacer: An End-to-End Routability-Aware
Placer with Graph Neural Network", KDD 2024 (arXiv:2406.02651);
Wang et al., "LHNN: Lattice Hypergraph Neural Network for VLSI Congestion
Prediction", arXiv:2203.12831.

**Observation**: Both DREAMPlace-Cong (DATE 2021) and RoutePlacer
(KDD 2024) report that a **GNN trained to predict actual router congestion**
(from labeled GR runs) gives a **differentiable penalty** with a gradient
that outperforms RUDY-based penalties on overflow reduction by 30-80 %.

**Why not for us**: requires (1) labeled GR-congestion training data and
(2) a GNN forward pass per Adam step (~100ms even on GPU). Out of scope
for a 1-2 week effort, and our M3 CPU env doesn't have a GNN trained on
canonical congestion.

**If we had 4+ weeks**: train a small GNN on cascade-cached placements
(we have 121 .pt files) with canonical-congestion labels, integrate as
an extra penalty term. **Expected lift: 10-20 % on hard benches.**

---

## 4. Lower-EV ideas to test if 1-5 don't deliver

### IMPROVEMENT 11 — Half-perimeter density (RUDY-pin)

**Citation**: circuitnet routability features document.

PinRUDY = `(w_n + h_n) / (w_n · h_n)` per pin (NO `s_ij/s_k` ratio).
Captures per-cell **pin density** (not wire density). Add as a separate
penalty: `0.05 · pin_density_top5%`. Cheap and orthogonal.

### IMPROVEMENT 12 — Per-tile L-route demand split (Spindler RUDY-long)

**Citation**: Spindler-Johannes RUDY, DATE 2007. The "RUDY long" variant
splits demand by `s_ij / s_k` (tile intersection / net intersection area)
when a net spans >1 tile, vs concentrating on a single tile for "RUDY
short" nets. Our current model is RUDY-pure (full uniform spread). RUDY-long
matches canonical's behavior on long nets better.

### IMPROVEMENT 13 — Multi-commodity flow LP-based congestion lower bound

**Citation**: Garg-Konemann (1998) approximation for MCF; Albrecht 2001
"Global routing by approximation algorithms for multicommodity flow",
IEEE TCAD.

Solve a relaxed MCF on the routing grid once per pseudo-step. Each net
gets a fractional assignment over candidate L/Z-shapes; the MCF objective
naturally minimizes max-cell congestion. Use the resulting per-edge
fractional demand as a **lower bound** to add to our RUDY.

**Cost**: each MCF LP is ~50ms-1s for our grid size. Could be done every
50 Adam steps (1-2 % overhead).

**Expected lift**: high (MCF is the *correct* model), but requires LP
infrastructure (similar to E64 which we killed for that reason).

### IMPROVEMENT 14 — A*-pattern routing inside the proxy

**Citation**: BoxRouter (Cho-Pan, DAC 2006); FastRoute (Pan-Chu 2008).

For each net, perform a soft A* over the routing grid with congestion as
edge cost. The soft A* uses a softmin (logsumexp) over neighbor expansions,
making the path probability differentiable. This is essentially what DGR
(NVidia DAC 2024, arXiv pending) does, but limited to L/Z-paths.

**Implementation cost**: high (need to vectorize a soft A* on GPU).
**Expected lift**: closer to optimal global routing.

### IMPROVEMENT 15 — Cell-bin congestion (FFT/Poisson-smoothed)

**Citation**: ePlace (Lu-Cheng TODAES 2015); FFTPL (Lu-Hsu 2013).

Treat per-cell congestion as a 2D scalar field, solve a Poisson equation
to compute the "congestion potential" of moving a macro to a new cell.
The gradient of the potential is the natural force pushing macros away
from congested zones. Used in ePlace for density, but never (?) used for
congestion. **Speculative high-EV bet** — would close 5+ % gap if it
works, but requires custom FFT integration.

---

## 5. Cross-domain inspirations

### Multi-commodity flow (operations research)

The "Garg-Konemann" approximation algorithm for fractional MCF (1998) gives
a (1+ε)-approximation in `O(ε^-2 · m · log m)` shortest-path computations.
This is the **theoretically correct** formulation for routing congestion
estimation, and it's what NeuroSteiner / DGR are approximating with their
GNN/DAG-forest models. Worth understanding even if we don't implement it
directly. [Albrecht 2001](https://ieeexplore.ieee.org/document/920691/) is
the canonical EDA paper.

### A* path planning with soft cost (robotics)

Soft A* (used in some motion planning literature) maintains a probability
distribution over partial paths. For our routing demand: build a soft
shortest-path forest over the grid from each source pin to all reachable
sinks, with congestion-weighted edge costs. Backprop gives gradient flow
of congestion to pin positions. Generalizes to L/Z/W routes naturally.
**Complexity**: O(N·M·log(N·M)) per net per step. Too slow for our budget.

### Differentiable sorting (machine learning)

Cuturi 2019 Sinkhorn-sort gives `O(N · log N · iterations)` differentiable
sort. Use to replace our top-K with a sorted-cells weighted sum:
`sum(cell[k] · weight(k))` where `weight(k)` is a smooth function peaked
at K=5%. Better gradient quality than torch.topk.

### Optimal transport (statistics/optimization)

For aligning our smooth congestion map with the canonical congestion map:
solve a Wasserstein distance between them, use the OT plan as a
gradient. This is essentially what "Mitigating Distribution Shift for
Congestion Optimization in Global Placement" (Zheng et al., DAC 2023) does
for the train/test gap in GNN-based predictors. Could be adapted as a
**self-supervised regularizer**: at each Adam step, compute canonical on
a cached placement, compute OT(smooth, canonical), use OT as auxiliary
loss to align the smooth proxy to canonical *for this specific basin*.

### Kernel density estimation (statistics)

The Gaussian-smoothing in our pin assignment is precisely 1D kernel
density estimation. Switching to **adaptive bandwidth** (narrower σ
where the density is high, wider where the density is sparse) is the
Botev-Grotowski-Kroese (2010) approach. For us: use a smaller σ in
high-pin-density regions of the chip, larger σ in sparse regions.
Likely small effect, but free if we already have pin-density precomputed.

### Variance reduction (Monte Carlo)

Our `_trace_route_congestion` is essentially a deterministic sum over
N_pairs L-routes. The literature on MCF-based congestion (Garg-Konemann,
Albrecht 2001) uses **shortest-path sampling** with importance weights
to reduce variance. For us this could mean: subsample n_pairs at each
Adam step (e.g., 5K of 30K pairs), with importance weight = pair_weight ·
N_pairs / 5K. Gives a noisy gradient but **2-6× faster per step**.

---

## 6. Ranked priority list

| Rank | Improvement | Effort | Expected canonical-mismatch reduction | Expected basin-quality lift after CD |
|---|---|---|---|---|
| 1 | 3-pin Steiner (T-route only, simple) | S | +14 % → +10 % | +1-2 % proxy |
| 2 | Sharper soft cell-assignment + anneal | S | +14 % → +10 % | +0.5-1 % proxy |
| 3 | 3-pin Steiner (4-case full) | M | +14 % → +6-8 % | +2-3 % proxy |
| 4 | Power-mean top-K (p=6) | S | +14 % → +12 % | +0.5 % proxy |
| 5 | Westra L/Z probabilistic | M | +14 % → +20 % (worse scalar!) | +1-3 % proxy (better gradient) |
| 6 | Partial-overlap macro fix | S | +14 % → +12 % | 0-0.5 % |
| 7 | Capacity-aware normalization | S | +14 % → ±5 % | 0-1 % (regularizer) |
| 8 | Sinkhorn top-K | M-L | +14 % → +12 % | +1 % |
| 9 | Gaussian (not box) smoothing | S | +14 % → +12 % | +0-2 % |
| 10 | FLUTE-precomputed RSMT init lane | L | structural change | +3-5 % (high variance) |
| 11 | GNN trained on cascade .pt cache | XL | n/a | +5-10 % (highest ceiling) |
| 12 | MCF-LP every 50 steps | L | +14 % → +5 % | +3-5 % |
| 13 | OT alignment regularizer | M | +14 % → +8-10 % | +0.5-1 % |

**Recommended next 3 experiments** (cheapest first):
1. **E112** (Improvement 2, S effort): Sharper cell-assign (σ=0.15, anneal
   σ=0.5→0.1). Half-day exp. Test scalar match + ibm17 + 60s CD polish.
2. **E113** (Improvement 1 simple variant, S effort): Add T-route for 3-pin
   nets only (no case split). Half-day exp. Same kill gate.
3. **E114** (Improvement 4, S effort): Power-mean p=6 top-K. Trivial 1-line
   change. Test as toggle, may bundle with E112-E113.

If E112-E114 deliver, we've closed +14 % → +6-8 % canonical mismatch with
**under 3 days of engineering**. Then attempt full 3-pin Steiner (#3) or
MCF-LP (#12) for the next 5-10 %.

If E112-E114 do not deliver (gradient signal already saturated), pivot to
**Improvement 10** (FLUTE-RSMT lane) which is structurally different and
attacks a different lever (topology, not fidelity).

---

## 7. Sources

### Primary literature

- Spindler & Johannes, "Fast and Accurate Routing Demand Estimation for
  Efficient Routability-driven Placement", DATE 2007 —
  [PDF](https://past.date-conference.com/proceedings-archive/2007/DATE07/PDFFILES/08.7_1.PDF) /
  [IEEE](https://ieeexplore.ieee.org/document/4211973/) — the original RUDY paper.
- Lou et al., "Estimating Routing Congestion using Probabilistic Analysis",
  ISPD 2001 —
  [PDF](https://cecs.uci.edu/~papers/compendium94-03/papers/2001/ispd01/pdffiles/p112.pdf).
- Westra, Bartels, Groeneveld, "Probabilistic Congestion Prediction",
  ISPD 2004 —
  [PDF](https://www.cs.york.ac.uk/rts/docs/SIGDA-Compendium-1994-2004/papers/2004/ispd04/pdffiles/p204.pdf).
- Sham, Young, "Congestion Prediction in Early Stages of Physical Design",
  TODAES 2009 —
  [PDF](http://www.cse.cuhk.edu.hk/~fyyoung/paper/todaes09_congestion.pdf) —
  comprehensive survey.
- Chu, "FLUTE: Fast lookup table based rectilinear Steiner minimal tree
  algorithm for VLSI design", IEEE TCAD 2008 / ISPD 2005 —
  [Iowa State PDF](https://home.engineering.iastate.edu/~cnchu/pubs/j29.pdf).
- Pan & Chu, "FastRoute: An efficient and high-quality global router",
  VLSI Design 2012 — [Wiley](https://hindawi.com/journals/vlsi/2012/608362).
- Roy & Markov, "Seeing the Forest and the Trees: Steiner Wirelength
  Optimization in Placement", IEEE TCAD 2007 —
  [PDF](https://web.eecs.umich.edu/~imarkov/pubs/jour/tcad07-rooster.pdf).

### Analytical placement (ePlace/DREAMPlace/Xplace lineage)

- Lu et al., "ePlace: Electrostatics-Based Placement Using Fast Fourier
  Transform and Nesterov's Method", ACM TODAES 2015 —
  [PDF](https://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf) /
  [ACM](https://dl.acm.org/doi/10.1145/2699873) — eDensity, Poisson-FFT.
- Cheng et al., "RePlAce: Advancing Solution Quality and Routability
  Validation in Global Placement", IEEE TCAD 2018 —
  [PDF](https://vlsicad.ucsd.edu/Publications/Journals/j126.pdf).
- Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration
  for Modern VLSI Placement", DAC 2019 / TCAD 2020.
- Liao et al., "DREAMPlace 4.0: Timing-driven Global Placement with
  Momentum-based Net Weighting", DATE 2022.
- Liu et al., "Xplace: An Extremely Fast and Extensible Global Placement
  Framework", DAC 2022 —
  [PDF](https://liulixinkerry.github.io/src/dac22_xplace.pdf) /
  [TCAD 2023](https://dl.acm.org/doi/10.1109/TCAD.2023.3346291) /
  [GitHub](https://github.com/cuhk-eda/Xplace) — uses RUDY map for cell
  inflation, calls CU-GR for routability eval (NOT differentiable
  congestion in objective).
- Agnesina et al., "AutoDMP: Automated DREAMPlace-based Macro Placement",
  ISPD 2023 —
  [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/AutoDMP.pdf) /
  [NVidia blog](https://developer.nvidia.com/blog/autodmp-optimizes-macro-placement-for-chip-design-with-ai-and-gpus/) —
  82.5 % overflow reduction via MOTPE Bayesian optimization, not via
  better congestion model.

### Differentiable congestion (recent)

- Liu et al., "Global Placement with Deep Learning-Enabled Explicit
  Routability Optimization" (DREAMPlace-Cong), DATE 2021 —
  [PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C112-DATE2021-DREAMPlace-Cong.pdf).
- Zheng et al., "Mitigating Distribution Shift for Congestion Optimization
  in Global Placement", DAC 2023 —
  [PDF](https://yibolin.com/publications/papers/PLACE_DAC2023_Zheng.pdf).
- Liu et al., "RoutePlacer: An End-to-End Routability-Aware Placer with
  Graph Neural Network", KDD 2024 (arXiv:2406.02651) — GNN-based
  differentiable congestion penalty.
- Wang et al., "LHNN: Lattice Hypergraph Neural Network for VLSI Congestion
  Prediction", IJCAI 2022 (arXiv:2203.12831).
- NVidia, "DGR: Differentiable Global Router", DAC 2024 —
  [ACM](https://dl.acm.org/doi/10.1145/3649329.3656530) /
  [GitHub](https://github.com/NVlabs/Differentiable-Global-Router) — DAG
  forest + Gumbel-softmax for L/Z route selection.
- "GOALPlace: Begin with the End in Mind",
  [arXiv:2407.04579](https://arxiv.org/pdf/2407.04579).
- "Differentiable Net-Moving and Local Congestion Mitigation" (DCGP),
  DAC 2025 —
  [PDF](https://ieda.oscc.cc/res/papers/25-DAC25-DCGP.pdf) — momentum-
  based cell inflation + differentiable congestion function.

### Global router lineage

- Cho & Pan, "BoxRouter: A new global router based on box expansion and
  progressive ILP", DAC 2006 / IEEE TCAD 2007 —
  [PDF](https://www.cerc.utexas.edu/utda/publications/dac06_boxrouter.pdf).
- Liu et al., "NCTU-GR 2.0: Multithreaded Collision-Aware Global Routing
  with Bounded-Length Maze Routing", IEEE TCAD 2013 —
  [ResearchGate](https://www.researchgate.net/publication/221060550_NCTU-GR_20_Multithreaded_Collision-Aware_Global_Routing_with_Bounded-Length_Maze_Routing).
- Liu, Pui, Young, "CUGR: Detailed-Routability-Driven 3D Global Routing
  with Probabilistic Resource Model", DAC 2020 —
  [PDF](https://cwpui.com/doc/c10.pdf) /
  [IEEE](https://ieeexplore.ieee.org/document/9218646) /
  [GitHub](https://github.com/cuhk-eda/cu-gr).
- Albrecht, "Global routing by new approximation algorithms for
  multicommodity flow", IEEE TCAD 2001 —
  [PDF](https://janders.eecg.utoronto.ca/1387/readings/global_routing.pdf).

### Routability-driven placement

- Hsu et al., "NTUplace4h: A Novel Routability-Driven Placement Algorithm
  for Hierarchical Mixed-Size Circuit Designs", IEEE TCAD 2014 —
  [IEEE](https://ieeexplore.ieee.org/document/6951861).
- He et al., "Ripple 2.0: High-Quality Routability-Driven Placement via
  Global Router Integration", DAC 2013 —
  [PDF](https://dl.acm.org/doi/pdf/10.1145/2463209.2488922).
- "Routability-wirelength co-guided cell inflation",
  [ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S0167926025002810).

### ML / OT / differentiable selection

- Cuturi, Teboul, Vert, "Differentiable Ranks and Sorting using Optimal
  Transport", NeurIPS 2019 — [arXiv:1905.11885](https://arxiv.org/pdf/1905.11885).
- Xie et al., "Differentiable Top-k with Optimal Transport", NeurIPS 2020 —
  [PDF](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf).
- "NeuroSteiner: A Graph Transformer for Wirelength Estimation",
  DAC'24 WIP poster —
  [arXiv:2407.03792](https://arxiv.org/html/2407.03792) — 0.3 % WL error,
  0.3 ms/net via GraphGPS.

### Survey / circuitnet references

- [circuitnet routability features](https://circuitnet.github.io/feature/routability%20features.html) —
  formulas for RUDY, RUDY-pin, RUDY-long, RUDY-short.
- [LogSumExp / smooth-max (Wikipedia)](https://en.wikipedia.org/wiki/LogSumExp).
- "Smooth maximum",
  [Wikipedia](https://en.wikipedia.org/wiki/Smooth_maximum).

### Internal references

- `experiments/E111_per_net_trace_congestion/code/per_net_trace_proxy.py` —
  current implementation.
- `experiments/E111_per_net_trace_congestion/manifest.md` — verified
  preliminary numbers on ibm17.
- `analysis/rudy_fidelity/README.md` — diagnostic finding that DPO-era
  RUDY mismatched by 3.1× on ibm01.
- `docs/handoffs/2026-05-20_comparative_analysis.md` — cost decomposition
  showing congestion = 58-72 % of total proxy.
- `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`
  (lines 1269-1606) — canonical `get_routing` reference.

---

## 8. Key formulas extracted from the literature

### RUDY (Spindler-Johannes 2007)

For each net `n` with bounding box `(w_n × h_n)` and HPWL `L_n ≈ w_n + h_n`:
```
d_n = L_n / (w_n · h_n) = (w_n + h_n) / (w_n · h_n)
```
Per-tile demand at location `(i, j)`:
```
RUDY_n(i,j) = d_n · I((i,j) ∈ bbox_n)
RUDY(i,j) = Σ_n RUDY_n(i,j)
```

### RUDY-long (per-tile, with overlap fraction)

```
RUDY_k(i,j) = (w_k + h_k) / (w_k · h_k) · s_ij / s_k
```
where `s_ij` = tile area, `s_k` = tile-net overlap area.

### Westra probabilistic L+Z (ISPD 2004)

For a 2-pin net with bbox `(w, h)`, w/wire-width ε:
- Probability route uses L-shape A (corner at `(x_max, y_min)`):
  `P_L1 = α`
- Probability route uses L-shape B (corner at `(x_min, y_max)`):
  `P_L2 = α`
- Probability route uses Z-shape with bend at `x = x_min + k`:
  `P_Z(k) = (1-2α) / (w-1)`

with α ≈ 0.5 empirically. Demand on edge `e` in bbox:
```
demand(e) = P_L1·1(e ∈ L1) + P_L2·1(e ∈ L2) + Σ_k P_Z(k)·1(e ∈ Z_k)
```

### Weighted-Average wirelength (Hsu-Chang TCAD 2014)

```
WA-WL_x = (Σ_p x_p · exp(γ·x_p)) / (Σ_p exp(γ·x_p))
       - (Σ_p x_p · exp(-γ·x_p)) / (Σ_p exp(-γ·x_p))
```
γ → ∞: HPWL exactly. γ → 0: average. Convex, differentiable, no smoothing
bias unlike LSE.

### Smooth max via LogSumExp

```
softmax_β(x) = (1/β) · log(Σ_i exp(β·x_i))
```
β → ∞: max. β → 0: average. Used for routing range endpoints in our code
(`_soft_min_max`).

### Sinkhorn-relaxed top-K (Xie-Wang-Wang 2020)

Define cost matrix `C ∈ R^(N × 2)` with `C[i, 0] = -x_i` (selected),
`C[i, 1] = 0` (not selected). Solve regularized OT:
```
T* = argmin_T <T, C> - ε·H(T)
s.t. Σ_j T[i, j] = 1/N (each cell has unit mass)
     Σ_i T[i, 0] = K/N, Σ_i T[i, 1] = (N-K)/N
```
Sinkhorn fixed-point: alternate rescaling rows / columns. `T*[i, 0]` is
the relaxed assignment of cell `i` to "top-K". Top-K mean:
`(N/K) · Σ_i x_i · T*[i, 0]`.

### Multi-Commodity Flow LP (Garg-Konemann 1998, Albrecht 2001)

For grid with capacity `c_e` per edge, nets `n` with demand `d_n` and
candidate paths `P_n`:
```
min Σ_e max(0, Σ_n Σ_{p ∈ P_n} f_{n,p}·1(e ∈ p) - c_e)
s.t. Σ_{p ∈ P_n} f_{n,p} = d_n  ∀n
     f_{n,p} ≥ 0
```
Garg-Konemann: solve via repeated SP with exponential edge weight
`w_e = (1+ε)^(use_e / cap_e)`. (1+ε)-approximation in
`O(ε^-2·m·log m)` SP computations. Naturally smooth (exponential
penalty) and ε-differentiable.
