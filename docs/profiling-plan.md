# Profiling & Benchmarking Plan

Last updated: 2026-04-04

**Goal:** Identify exactly where the polyhedra navigation placer loses to RePlAce and which
improvements have the highest leverage. Profiling only — no algorithm changes.

---

## 1. Cost Component Breakdown

**Effort: Zero** — data already exists in experiment_log.jsonl.

Decompose proxy cost per benchmark into weighted contributions:
- `wl_contribution = 1.0 * wirelength_cost` (~5% of proxy)
- `den_contribution = 0.5 * density_cost` (~30-35% of proxy)
- `cong_contribution = 0.5 * congestion_cost` (~50-65% of proxy)

Sub-analyses:
- **V vs H congestion split** — `get_V_congestion_cost()` / `get_H_congestion_cost()` per benchmark. Is horizontal or vertical routing the bottleneck?
- **Density distribution** — `plc.get_grid_cells_density()` full grid array. Measure: hot cell count (>90th percentile), max cell density, density variance, spatial clustering of hot cells.
- **Per-net HPWL distribution** — identify worst nets driving wirelength cost.

**Output:** Component contribution table + ranked list of which component to attack per benchmark.

---

## 2. Per-Benchmark Gap Analysis vs Baselines

**Effort: Zero** — baselines in evaluate.py, our results in experiment_log.jsonl.

For each of 17 benchmarks, compute:
- Gap vs RePlAce: `(ours - replace) / replace * 100`
- Gap vs SA: `(ours - sa) / sa * 100`
- Per-component gap vs RePlAce (approximate — we don't have RePlAce component data)

Triage into categories:
- **Winning** (ibm02, ibm10, ibm12): understand why we win here — what's structurally different?
- **Close** (ibm03, ibm04, ibm07, ibm08, ibm09, ibm13-16, ibm18): highest-leverage targets
- **Hard losses** (ibm01, ibm06, ibm11, ibm17): diagnose root cause

Correlate benchmark structural properties (num macros, utilization, net density, avg net degree)
with gap size. Which features predict where we lose?

**Output:** Gap table ranked by improvement potential + structural correlation analysis.

---

## 3. Surrogate vs Ground Truth Accuracy

**Effort: Moderate** — requires instrumentation in `greedy_descent`.

The GridSurrogate filters candidates before expensive full evaluation. If it ranks poorly,
we miss good moves. **Known discrepancies:**

| Component | Surrogate | Ground truth (PlacementCost) |
|-----------|-----------|------------------------------|
| Density | Top 10% of ALL cells | Top 10% of NON-ZERO cells |
| Congestion | ABU(5%) of concat V+H arrays | ABU(5%) of `V + H` element-wise sum |
| Wirelength | `total_net_count` from plc.nets | `self.net_cnt` |

Measurements:
- Scatter plot: surrogate estimate vs true cost for all verified candidates
- Rank correlation (Spearman) — does surrogate preserve ordering?
- Per-component accuracy (surrogate WL vs true WL, etc.)
- Filtering effectiveness: what fraction of surrogate-top-5 are also in true-top-5?
- Systematic bias: does surrogate consistently over/under-estimate?

**Output:** Correlation metrics + calibration plot + recommendation (increase resolution, fix formula, or accept as-is).

---

## 4. Convergence Curves (Cost After Each Step)

**Effort: Light instrumentation** — log checkpoints in `greedy_descent`.

Track proxy cost progression through the pipeline:

| Checkpoint | Record |
|------------|--------|
| After SDF init | proxy, wl, den, cong |
| After LP solve | proxy, wl, den, cong (LP positions, not used directly) |
| After each navigation improvement | proxy, wl, den, cong, iteration#, elapsed_time |
| Final result | proxy, wl, den, cong |

Key questions:
- How much quality comes from SDF init vs navigation?
- Where does improvement plateau? Is the time budget used effectively?
- Do individual moves trade off components (improve WL but worsen congestion)?
- What's the marginal value of additional time?

**Output:** Per-benchmark convergence CSV: `(iteration, elapsed, proxy, wl, den, cong)`.

---

## 5. Computational Profiling (Time Per Phase)

**Effort: Moderate** — manual instrumentation or cProfile/py-spy.

The placer has 5 phases:

| Phase | Expected time | Notes |
|-------|---------------|-------|
| SDF initial placement | ~3s | Fixed cost |
| Assignment extraction | ~0.01-0.1s | O(n^2) pairwise |
| LP solver build | ~0.1-0.5s | Constraint construction |
| LP solve (HiGHS) | 1.7-9s | Scales with benchmark size |
| Navigation | 30-300s | **Dominates** |

Within navigation, profile sub-operations:
- **Candidate generation** (rank_candidates + propose_cluster_flips): per-iteration cost
- **Projection** (_project_flip + cascade): per-candidate cost
- **Surrogate evaluation** (evaluate_move): per-candidate cost (~0.1ms claimed)
- **Full proxy evaluation** (_eval_proxy → compute_proxy_cost): per-verification cost (~1s)
- **LP re-solve** (periodic dual refresh): frequency and cost
- **Assignment re-extraction** (after improvement): cost per improvement

Derived metrics:
- Number of iterations vs improvements vs time budget exhaustion
- Surrogate evals per iteration, full evals per iteration
- Time-to-first-improvement and time-to-last-improvement

Tools: `cProfile` for function-level call counts, `py-spy` for flame graphs, manual
`time.perf_counter_ns` in the hot loop.

**Output:** Phase timing table per benchmark + navigation sub-operation breakdown.

---

## 6. Sensitivity Analysis

**Effort: High** — requires multiple runs per parameter.

### 6a. Seed variance
Run 5-10 seeds on fast subset. If variance >1%, ensemble strategies (run N seeds, take best) are worth pursuing.

### 6b. Time budget
Run with nav_time = 60, 120, 300, 600, 1200s. Plot proxy cost vs time budget to find the knee.

### 6c. Navigation hyperparameters

| Parameter | Current | Values to test |
|-----------|---------|----------------|
| `alpha` (dual weighting) | 1.0 | 0.5, 1.0, 2.0 |
| `top_k` candidates | 150 | 50, 100, 150, 300 |
| `top_k_verify` | 5 | 1, 3, 5, 10, 20 |
| Cluster flip proposals | 20 | 10, 20, 40 |
| Stale iteration threshold | 100 | 50, 100, 200 |
| Dual refresh frequency | every 5 improvements | 3, 5, 10 |

### 6d. LP solver parameters

| Parameter | Current | Values to test |
|-----------|---------|----------------|
| Separation epsilon | 0.002 | 0.0, 0.001, 0.01, 0.1 |
| LP time limit | 60s | 10, 30, 60, 120 |
| Min-displacement margin | 5.0 | 1.0, 5.0, 20.0 |

### 6e. Density refinement
Currently `refine=False`. Test enabling with `density_steps` = 50, 80, 150 and `lr` = 0.05, 0.1, 0.2.

### 6f. Initial placement source
Compare SDF init vs greedy shelf-pack init — how much does initial topology quality affect final results?

**Output:** Parameter sensitivity tables + identified high-leverage knobs.

---

## 7. Benchmark Structural Characterization

**Effort: Moderate** — extract from Benchmark objects.

Build a table for all 17 benchmarks:

| Property | How to extract |
|----------|----------------|
| Num hard macros | `benchmark.num_hard_macros` |
| Num soft macros | `benchmark.num_soft_macros` |
| Num nets | `benchmark.num_nets` |
| Canvas area | `canvas_width * canvas_height` |
| Total hard macro area | `sum(macro_sizes[:n_hard, 0] * macro_sizes[:n_hard, 1])` |
| Utilization | macro_area / canvas_area |
| Avg net degree | `mean([len(net) for net in net_nodes])` |
| Max net degree | `max(...)` |
| Grid size | `grid_rows * grid_cols` |
| Num fixed macros | `macro_fixed.sum()` |
| Avg macro aspect ratio | `mean(max(w,h)/min(w,h))` |
| Num hard macro pairs | `n_hard * (n_hard - 1) / 2` |

Correlate each property with: proxy cost, gap vs RePlAce, runtime, number of navigation improvements.

**Output:** Benchmark characterization table + correlation analysis identifying difficulty predictors.

---

## 8. Placement Quality Metrics

**Effort: Moderate** — extract from placement + Benchmark.

- **Canvas utilization**: hard macro area / canvas area, bounding box of placed macros / canvas
- **Spread**: center of mass vs canvas center, std dev of x/y coordinates
- **Net stretch**: per-net actual HPWL / optimal HPWL (if all macros at centroid)
- **Assignment distribution**: L/R/B/A direction counts — is the assignment biased?
- **Dual variable distribution**: magnitude histogram, identify bottleneck macros
- **Active constraints**: count of tight (nonzero-dual) separation constraints vs total pairs

**Output:** Quality metrics table per benchmark.

---

## 9. Cross-Placer Comparison

**Effort: Low** — data exists for SDF and polyhedra.

For benchmarks where both SDF and polyhedra have --all data:
- Compare per-component costs (does SDF have better density? polyhedra better WL?)
- Identify hybridization opportunities (e.g., different init for different benchmarks)
- Check if OT variants from experiment_log have useful per-component insights

**Output:** Cross-placer component comparison table.

---

## 10. Visualization

**Effort: High** — new scripts needed.

| Visualization | What it shows |
|---------------|---------------|
| Density grid heatmap | Hot cells overlaid on placement |
| Congestion grid heatmap | Routing demand by grid cell |
| Convergence plot | Proxy cost vs time per benchmark |
| Gap waterfall | Component breakdown vs RePlAce per benchmark |
| Surrogate calibration scatter | Surrogate vs true cost for all candidates |
| Dual variable heatmap | Bottleneck macros colored by |dual| sum |
| Net visualization | Bounding boxes for worst-N nets |

**Output:** PNG/SVG files in `vis/profile/`.

---

## Execution Priority

Ranked by insight-to-effort ratio:

| Priority | Section | Why |
|----------|---------|-----|
| 1 | Cost component breakdown | Free — data exists, shows where proxy cost lives |
| 2 | Gap analysis | Free — shows which benchmarks to focus on |
| 3 | Surrogate accuracy | Critical — if surrogate ranks poorly, navigation is blind |
| 4 | Convergence curves | Light effort — reveals if time budget is used well |
| 5 | Computational profiling | Moderate — guides speed optimization |
| 6 | Sensitivity analysis | High effort, high value — finds best knobs |
| 7 | Benchmark characterization | Moderate — predicts difficulty per benchmark |
| 8 | Placement quality metrics | Supplementary — useful for debugging |
| 9 | Cross-placer comparison | Low effort — leverages existing data |
| 10 | Visualization | High effort — primarily for debugging/communication |

