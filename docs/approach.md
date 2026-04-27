# Approach

Last updated: 2026-04-27

The current approach is **full-proxy coordinate descent on an incremental evaluator with per-benchmark plateau detection**. It superseded the DPO and polyhedra-navigation approaches that came before. Both are documented further down — keep them in mind for the writeup, since the failures motivated the architecture that won.

---

## 1. Current approach: CD on incremental evaluator (CHAMPION)

**Result:** avg proxy **1.1055** on --all (17 IBM benchmarks), beats leaderboard 1.1172 by -1.05%, beats RePlAce 1.4578 by -24.2%. Zero overlaps. Total runtime 17480s (4.85 hr).

### 1.1 The pipeline

```
SDF init
  -> push-apart projection (clean residual overlaps)
  -> IncrementalProxyEvaluator (full-proxy: WL + density + congestion)
  -> coordinate descent sweeps (per-axis breakpoint enumeration)
       with per-benchmark plateau detection
  -> validate (zero overlaps, fixed macros not moved)
```

### 1.2 Components

**SDF init.** Same as DPO — analytical spreading via signed distance fields produces a non-overlapping starting placement that respects density. Reused from `submissions/polyhedra/init/sdf.py`.

**Incremental proxy evaluator (E1).** `macro_place/incremental_evaluator.py`. Caches per-net min/max trackers, per-cell density, per-cell macro routing, per-net WL bbox; updates only what a single-macro move touches. Re-uses bit-for-bit the smoothing pass from `compute_proxy_cost` for parity. **4657× speedup per move on ibm10**, parity at 1e-15 absolute, revert tested. This is load-bearing — without it, 600s/bench wouldn't be enough sweeps to converge.

**Coordinate descent.** For each non-fixed macro, search both x and y axes via breakpoint enumeration:
- *Breakpoints* = net endpoints (HPWL change-points) + grid bin lines (density change-points) + congestion change-points within the legal range.
- *Legal range* per axis = `legal_axis_range(macro, axis)` — clamps to canvas + non-overlap with all other macros at the current other-axis value.
- *Best position* on the axis = the breakpoint that minimizes proxy under the incremental evaluator. Closed-form when breakpoint count is tractable; falls back to golden section if too dense.
- *Acceptance* = monotone, only commit if proxy strictly drops by > 1e-9.

**Per-benchmark plateau detection (E9 — the leaderboard-beating change).** Each benchmark exits CD when:
- 3 consecutive sweep-deltas drop below `plateau_threshold = 0.005`, AND
- `elapsed >= min_time_s = 300` (don't exit too early on a noisy first few sweeps).

Or hard-caps at `hard_cap_s = 3600` (matches competition 1hr/bench rule). All 17 IBM benchmarks exit via plateau under defaults; none hit the cap. Easy benchmarks exit at ~5-10 min; hard benchmarks (ibm14/15/16/17) get 25-37 min when they're still descending.

### 1.3 Why this works

The 2026-04-26 LP-HPWL diagnostic decomposed proxy as **6% WL, 20% density, 74% congestion**. That falsified two prior framings:

1. *Pure HPWL CD* (Miftari weighted-median) caps at ~5% gain. WL alone is a small fraction.
2. *DPO via gradient on smooth proxy* hits a basin ceiling at ~1.38; gradients can't cross discrete topology barriers (proven by 4-seed verification on `best_of_v2`: ibm02/ibm12 byte-identical).

Full-proxy CD with breakpoint enumeration is **basin-changing**:
- Each per-axis search considers all O(degree) net endpoints + O(grid_col) bin lines as candidates. That's typically 30-300 candidates per axis per macro, not a continuous gradient step.
- Across a sweep, every macro tries to relocate to a globally cost-minimizing position given the current state of all others. Topology can flip without going through "infeasible" intermediates.
- ibm02 (DPO basin-locked at 1.6888 across 4 seeds) drops to 1.1534 in 600s of CD — proves CD reaches a different fixed point.

### 1.4 The plateau-detection win

CDOnly's fixed 600s/bench was one-size-fits-all. Easy benchmarks (ibm09 ~3 min) plateaued early and wasted the rest. Hard benchmarks (ibm17/18 sweep deltas at 0.001-0.003 at the 600s mark) ran out mid-descent. CDAdaptive lets hard benchmarks use the time the easy ones save:

| Bench | E9 wall (s) | CDOnly wall (s) | E9 vs CDOnly |
|---|---|---|---|
| ibm09 (easy) | 449 | 600 | +0.06% (tied; saved 151s) |
| ibm14 (hard) | 1572 | 600 | -1.96% |
| ibm17 (hardest) | 2238 | 600 | **-3.64%** |

Total wall 17480s (E9) vs 10316s (CDOnly) — 70% more compute, but every minute spent on a hard benchmark.

### 1.5 Why this transfers to the hidden NG45 test

The plateau-detection policy is **per-benchmark, per-run**. It adapts to the actual descent trajectory rather than to priors set on IBM. On a benchmark we've never seen, this is the only sane policy.

The 1-hour hard cap matches the competition rule, so we never overshoot the per-benchmark compute limit even on the hardest case.

---

## 2. Prior approach: DPO (superseded — useful for writeup)

**Best result:** avg proxy **1.3834** on --all (`best_of_v2`). Replaced by CDOnly on 2026-04-27.

DPO differentiates through the actual proxy cost formula `f(p) = WL + 0.5*D + 0.5*C`:

1. **SDF init** (~3s) — non-overlapping starting placement.
2. **LSE-HPWL** — log-sum-exp smooth HPWL with annealed `gamma`.
3. **Differentiable grid density** — exact grid overlap, top-10% via `torch.topk`.
4. **Differentiable RUDY congestion** — smooth bbox → grid overlap → ABU-5%.
5. **Overlap penalty** — pairwise ReLU, annealed across 3 phases.
6. **Adam optimizer** — 3-phase penalty continuation (exploration → refinement → sharpening).
7. **Legalization** — iterative overlap repair for zero hard-macro overlaps.
8. **Best-of selection** — evaluate both SDF init and DPO output, return better per benchmark.

### 2.1 Why DPO hit a ceiling

Multi-seed verification (E5, 4 seeds: 43, 44, 45, 46) showed `best_of_v2` was within 1% across seeds on --all, and **ibm02/ibm12 were byte-identical** across all 4 seeds. DPO converges to a deterministic local minimum that gradient steps cannot escape. Within-DPO refinements consistently capped at 1-2%:

| Falsified DPO extension | Best (--all) | Lesson |
|---|---|---|
| E5: batched seeds, sigma=0.04·canvas | 1.3790 | All seeds collapse to same basin |
| E10: congestion-only refinement | 1.3788 | Marginal -0.33%; ibm02 *worse* +2.3% |
| E11: diverse priors (SDF + Will + greedy + random) | 1.3839 | Fast set -0.8%; --all FLAT +0.04%; ibm02/12 worse |

The diagnosis: DPO's smooth-proxy gradient cannot make discrete topology jumps. CD's breakpoint enumeration can.

---

## 3. Pre-DPO approach: Polyhedra navigation (superseded — kept for writeup)

**Best result:** avg proxy **1.4867** on --all (300s nav). Hit a ceiling at ~1.49.

The feasible region of non-overlapping placements is a **union of convex polyhedra**. Each polyhedron = a fixed L/R/A/B pairwise assignment. Within any polyhedron, the problem is a linear program. The framework decomposes:
1. **Which polyhedron** (discrete, NP-hard) — the topology
2. **Where within it** (continuous, polynomial) — solved exactly via LP

Implementation: SDF v5 init → assignment extraction + HPWL LP solve (HiGHS) + dual extraction → GridSurrogate (~0.1ms) for fast candidate evaluation → surrogate-guided navigation with SA acceptance → cluster moves (net-correlated + macro-centered multi-pair flips) → ClusterScreener (multi-tier pre-projection pruning) → robust projection (cascade repair + overlap repair).

### 3.1 Why polyhedra navigation hit a ceiling

Phase 5 (overnight Apr 14-15, 22 automated experiments across 3 subproblems) systematically falsified the levers:

- **SP3 (Surrogate fixes):** Surrogate is already well-calibrated. Top-k verify=20 helped marginally (1.4919, noise).
- **SP1 (Alternative inits):** SDF is genuinely hard to beat. Spectral (1.78), HMetis (1.91), greedy (1.75) all far worse.
- **SP4 (LP modifications):** Washed out by 50s of navigation. McCormick area penalty best at 1.4918 (noise).
- **Phase 4 (Tunneling):** Soft-pair flips, group cascade, sequence pairs all failed to cross the congestion barrier. Barrier is hundreds of flips wide.

### 3.2 The congestion barrier (the deep lesson)

Six experiments established that **local navigation cannot reach better congestion**:

| Experiment | Cong delta | Density delta | Conclusion |
|---|---|---|---|
| Real-proxy pair flips (300s) | <1% | -2.8% to -5.6% | Local flips only move density |
| Swap + LP | **-44%** | +180% | Different topologies CAN have better congestion |
| Swap only (no LP) | <0.5% | ~0% | Single swaps don't change routing structure |
| Group topology cascade | LP infeasible | N/A | Changing 200+ pairs creates constraint cycles |
| Sequence pair (1-500 steps) | +1-2% (worse) | +2-5% (worse) | Random distant topologies are worse |
| Position blending (SDF↔LP) | Monotonically worse until pure LP | -- | No smooth path in position space |

The barrier was **structural**, not surrogate quality. SDF's basin was locally optimal; nearby topologies were worse.

### 3.3 The LP-proxy disconnect

The LP optimizes HPWL, but **HPWL has zero correlation with proxy cost** (rho=-0.001, Miftari experiments). HPWL and density anti-correlate physically (tighter WL = denser); congestion is uncorrelated with HPWL (rho=0.072). This invalidated:
- Branch-and-bound with LP relaxation (bounds the wrong objective)
- Lagrangian bounds from LP duals (predict LP-HPWL not proxy)
- MCMC/SA weighted by LP cost (samples wrong distribution)
- Partial-commitment LP (bounds wrong objective)

**This is the unblock that motivated the incremental real-proxy evaluator** (E1) and ultimately CD on full proxy. See `docs/lp_hpwl_diagnostic.md` for the formal decomposition (6% WL, 20% density, 74% congestion).

---

## 4. Comparison vs traditional approaches

| Approach | Paradigm | Overlap handling | Congestion strategy | Our verdict |
|---|---|---|---|---|
| RePlAce | Nesterov on smooth relaxation | Continuous penalty → one-shot legalize | Implicit via density spreading | Beat by -24.2% |
| ePlace / DREAMPlace | Same as RePlAce, GPU-accelerated | Same | Same | Same paradigm |
| SA on B*-tree / Seq. Pair | Stochastic combinatorial search | Always feasible (compact repr.) | Evaluated but not optimized | Slow at N=500 |
| Partitioning (Capo) | Recursive bisection | Top-down slot assignment | Coarse: only at partition boundaries | Greedy, loses global structure |
| MaskPlace (RL) | Learned policy, sequential | Sequential feasible placement | Reward shaping | Generalization is poor |
| **Polyhedra navigation (ours, 1.49)** | LP inside combinatorial navigation | Always feasible by construction | Surrogate (RUDY) guides search | Hit congestion barrier |
| **DPO (ours, 1.38)** | Differentiable proxy + gradient steps | Iterative legalization | Smooth congestion gradient | Basin lock — within-basin only |
| **CD on incremental evaluator (ours, 1.10)** | Per-axis breakpoint enumeration on full proxy | Strict per-axis legality | Direct in evaluator | **CHAMPION** |
| **CD-adaptive (ours, 1.1055)** | CD + per-benchmark plateau detection | Same | Same | **BEATS leaderboard** |

---

## 5. Subproblem decomposition (revised post-CD)

The proxy `f(p) = WL + 0.5*D + 0.5*C`:

### 5.1 Wirelength (6%) — solved
LP-optimal within any topology; CD breakpoint enumeration captures the per-axis median exactly.

### 5.2 Density (20%) — solved
Top-10% is a non-convex order statistic, but the incremental evaluator updates per-cell density in O(touched cells) per move. CD descends density without surrogate error.

### 5.3 Congestion (74%) — handled by full-proxy CD
RUDY congestion *is* decomposable per single-macro move (per-net congestion contributions, per-cell macro routing, per-cell density). The smoothing pass on `current_cost()` is the dominant per-call cost. CD's breakpoint enumeration finds positions that minimize all three components simultaneously.

### 5.4 Feasibility — solved
`legal_axis_range` enforces strict per-axis non-overlap. Every CD move is legal by construction; final placement validated against `compute_overlap_metrics`. Zero overlaps on all 17 IBM benchmarks.

---

## 6. Innovation contributions (for writeup)

1. **Proxy decomposition diagnostic (E8).** First (in our exploration) to formally decompose Partcl proxy as 6%/20%/74% via LP-HPWL lower bound. Reframes the problem from "WL optimization with constraints" to "congestion-dominated multi-component objective". `scripts/lp_hpwl_lower_bound.py`.

2. **Incremental evaluator with full-proxy parity (E1).** First (in our exploration) to update RUDY congestion deltas per single-macro move at bit-for-bit parity with `compute_proxy_cost`. 4657× speedup unlocks coordinate-descent inside the 1hr/bench compute budget. `macro_place/incremental_evaluator.py`.

3. **Per-benchmark plateau detection (E9).** Adapts compute budget to each benchmark's actual descent trajectory rather than fixed budget or per-benchmark prior. Transfers to hidden NG45 without manual tuning. `submissions/cd/cd_adaptive_placer.py`.

4. **Falsification record.** 60+ experiments documented in `docs/experiment_index.md`. The LP-proxy disconnect (LP-HPWL doesn't correlate with proxy) and the basin lock (DPO byte-identical across seeds) are the deep findings that motivated the working approach.

---

## See also

- [problem.md](problem.md) — formal mathematical formulation
- [theory.md](theory.md) — tunneling frameworks and theoretical backing
- [results.md](results.md) — per-benchmark champion tables
- [roadmap.md](roadmap.md) — submission plan
- [experiment_index.md](experiment_index.md) — full catalog including falsified hypotheses
- [lp_hpwl_diagnostic.md](lp_hpwl_diagnostic.md) — E8 (the unblock)
- [cd_ibm10_results.md](cd_ibm10_results.md) — E2 single-bench breakthrough
- [closing_the_gap.md](closing_the_gap.md) — leaderboard-beating run narrative
- [evaluation.md](evaluation.md) — earlier surrogate-eval pipeline (now superseded)
