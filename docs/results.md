# Results

Last updated: 2026-04-28

Operational doc — current champion only. Historical per-benchmark
tables for DPO / Polyhedra / Overnight Sweep / Miftari live in
`writeup/historical_results.md`.

## Baselines

| Method | Avg Proxy (--all) | Overlaps | Notes |
|--------|-------------------|----------|-------|
| RePlAce | 1.4578 | 0 | Target to beat for champion |
| SA | 2.1362 | 0 | Weak baseline |
| Will's seed (SA 3000) | 1.5338 | 0 | Pre-fork submission |
| Greedy row | ~2.20 | 0 | Demo placer |
| Leaderboard target (vmallela) | 1.1172 | 0 | Unverified; CD+LNS, ~40 min/bench |

## Champion lineage

| Hypothesis | Status | Best Avg Proxy | Notes |
|------------|--------|----------------|-------|
| **CDLNSGridBinPlacer (E12)** | **CHAMPION** | **1.0990** | **Beats leaderboard 1.1172 by -1.63%; -24.6% vs RePlAce; CD plateau + grid-bin LNS overlay (ADR-007). Promoted 2026-04-28.** |
| CDAdaptivePlacer (E9) | superseded | 1.1055 | Was champion 2026-04-27; -0.59% lift from E12 LNS overlay. Plateau-bound: every bench exited via plateau, none hit cap. |
| CDOnlyPlacer | superseded | 1.1193 | Was champion 2026-04-27 (am); -1.23% lift from E9 adaptive budget |
| DPO best-of-v2 | superseded | 1.3834 | Was champion 2026-04-26; -19.1% vs CDOnly. Details in `writeup/historical_results.md`. |
| Polyhedra Navigation | superseded | 1.4867 | At ceiling; replaced by DPO. Details in `writeup/historical_results.md`. |
| SDF Density | init only | 1.5002 | Now used as init for both CD placers |

## CDLNSGridBinPlacer --- Champion Result (2026-04-28)

**Status: CHAMPION** --- avg proxy **1.0990** on --all (**24.6% better than RePlAce**, **beats leaderboard 1.1172 by -1.63%**, **-0.59% better than prior CDAdaptive champion**, zero overlaps everywhere).

Configuration: `submissions/cd_lns_gridbin/placer.py` (full-proxy CD on `IncrementalProxyEvaluator` with per-benchmark plateau detection, then a grid-bin LNS escape phase, SDF init). Total runtime 28 256 s = 7.85 hr.

Time-budget split per benchmark: CD phase ≤ 3 000 s + LNS phase ≤ 600 s = 3 600 s total (matches the contest 1-hour-per-bench cap). Plateau-detection params for the CD phase: `min_time_s=300, hard_cap_s=3000, patience=3, plateau_threshold=0.001`. LNS params: `destroy_frac=0.05, destroy_cap=30, destroy_strategy='cost_aware', lns_budget_s=600`.

### How it differs from CDAdaptive

CDAdaptive (E9) was *plateau-bound*, not budget-bound: every benchmark exited via plateau detection, none hit the 1-hour cap. The plateau is a per-axis fixed point of CD's breakpoint enumeration. Three candidate escape mechanisms were tested and falsified before E12 (E3 LNS, SDF jitter, subset-CD destroy — all reused CD's per-axis move type). E12's grid-bin LNS uses a **different move type**: for each destroyed macro it enumerates all `(grid_col, grid_row)` cell centers (~50–2 500 candidates per benchmark) and accepts the globally proxy-minimizing legal reinsertion. That candidate set is outside CD's per-axis breakpoint enumeration, so it can find escapes CD cannot.

### Per-benchmark (--all, zero overlaps everywhere)

| Bench | E12 (champion) | E9 CDAdaptive | RePlAce | Δ vs E9 | Wall (s) |
|-------|---------------:|--------------:|--------:|---------:|---------:|
| ibm01 | **0.9045** | 0.9159 | 0.9976 | **-1.24%** | 524 |
| ibm02 | **1.1340** | 1.1538 | 1.8370 | **-1.72%** | 1049 |
| ibm03 | **0.9886** | 0.9950 | 1.3222 | **-0.64%** | 791 |
| ibm04 | **1.0150** | 1.0226 | 1.3024 | **-0.74%** | 760 |
| ibm06 | **1.1583** | 1.1592 | 1.6187 | **-0.08%** | 1054 |
| ibm07 | **1.1021** | 1.1103 | 1.4633 | **-0.74%** | 1167 |
| ibm08 | **1.1190** | 1.1254 | 1.4285 | **-0.57%** | 1293 |
| ibm09 | **0.8591** | 0.8611 | 1.1194 | **-0.23%** | 660 |
| ibm10 | **1.0562** | 1.0749 | 1.5009 | **-1.74%** | 2433 |
| ibm11 | **0.9136** | 0.9223 | 1.1774 | **-0.94%** | 1179 |
| ibm12 | **1.2076** | 1.2153 | 1.7261 | **-0.63%** | 2771 |
| ibm13 | **0.9766** | 0.9772 | 1.3355 | **-0.06%** | 1874 |
| ibm14 | **1.2205** | 1.2234 | 1.5436 | **-0.24%** | 2390 |
| ibm15 | **1.1797** | 1.1809 | 1.5159 | **-0.10%** | 2098 |
| ibm16 | **1.1573** | 1.1610 | 1.4780 | **-0.32%** | 2426 |
| ibm17 | **1.3299** | 1.3326 | 1.6446 | **-0.20%** | 3487 |
| ibm18 | **1.3603** | 1.3633 | 1.7722 | **-0.22%** | 2301 |
| **AVG** | **1.0990** | **1.1055** | **1.4578** | **-0.59%** | **28 256 (7.85 hr)** |

All 17 benchmarks improved over E9 CDAdaptive. No regressions. Biggest wins on basin-locked / high-density benchmarks: ibm10 (-1.74%), ibm02 (-1.72%), ibm01 (-1.24%), ibm11 (-0.94%). Source JSON: `results/CDLNSGridBinPlacer_20260428_155739.json`.

### Random-destroy ablation (cost-aware ranking is NOT load-bearing)

The cost-aware destroy step in the LNS phase scores each candidate macro by its proxy delta when temporarily moved to canvas center, then picks the K most costly. An ablation `cd_lns_gridbin_random.py` (kept at `experiments/E12_grid_bin_lns/code/`) replaces this with uniform random destroy. On `--fast` (ibm01/04/09/13), random destroy averaged 0.9372 — matching cost-aware within noise. On ibm09, random destroy (0.8541) actually *beat* cost-aware (0.8591). Cost ranking adds ~10% wall per LNS sample but does not change quality. Future simplification: drop the ranking, use random. Stays cost-aware in the production placer for now to avoid mid-deadline changes.

---

## CDAdaptivePlacer --- Prior Champion (superseded 2026-04-28 by E12)

**Status: SUPERSEDED 2026-04-28 by CDLNSGridBin (E12)** --- avg proxy **1.1055** on --all (24.2% better than RePlAce, beat leaderboard 1.1172 by -1.05%, 1.23% better than prior CDOnly champion, zero overlaps everywhere).

Configuration: `submissions/cd_adaptive/placer.py` (full-proxy CD on `IncrementalProxyEvaluator`, **per-benchmark plateau detection** with 1hr hard cap, SDF init). Total runtime 17480 s = 4.85 hr.

Plateau-detection params: `min_time_s=300, hard_cap_s=3600, patience=3, plateau_threshold=0.005`. Each benchmark exits when 3 consecutive sweep-deltas fall below 0.005 (and 5 min minimum elapsed), or hits the 1-hour cap.

### How it differs from CDOnly

CDOnly used a fixed 600 s/benchmark — one-size-fits-all. Easy benchmarks plateaued at ~3 min and wasted the rest; hard ones (ibm17/18) ran out mid-descent. CDAdaptive lets each benchmark **exit early when converged** and **run longer when still descending** (up to 1 hr). Net: hard benchmarks get the extra time the easy ones save; nobody is forced to stop mid-improvement.

### Per-benchmark (--all, zero overlaps everywhere)

| Bench | E9 Adaptive | CDOnly | RePlAce | Δ vs CDOnly | Wall (s) | Plateau? |
|-------|-------------|--------|---------|-------------|----------|----------|
| ibm01 | 0.9159 | 0.9133 | 0.9976 | +0.28% | 322 | plateau |
| ibm02 | 1.1538 | 1.1534 | 1.8370 | +0.04% | 524 | plateau |
| ibm03 | 0.9950 | 0.9942 | 1.3222 | +0.08% | 534 | plateau |
| ibm04 | 1.0226 | 1.0193 | 1.3024 | +0.32% | 382 | plateau |
| ibm06 | **1.1592** | 1.1656 | 1.6187 | **-0.55%** | 792 | plateau |
| ibm07 | 1.1103 | 1.1105 | 1.4633 | -0.02% | 607 | plateau |
| ibm08 | **1.1254** | 1.1307 | 1.4285 | **-0.47%** | 833 | plateau |
| ibm09 | 0.8611 | 0.8606 | 1.1194 | +0.06% | 449 | plateau |
| ibm10 | **1.0749** | 1.1000 | 1.5009 | **-2.28%** | 1332 | plateau |
| ibm11 | 0.9223 | 0.9248 | 1.1774 | -0.27% | 614 | plateau |
| ibm12 | **1.2153** | 1.2418 | 1.7261 | **-2.13%** | 1424 | plateau |
| ibm13 | **0.9772** | 0.9939 | 1.3355 | **-1.68%** | 1143 | plateau |
| ibm14 | **1.2234** | 1.2478 | 1.5436 | **-1.96%** | 1572 | plateau |
| ibm15 | **1.1809** | 1.2109 | 1.5159 | **-2.48%** | 1609 | plateau |
| ibm16 | **1.1610** | 1.1919 | 1.4780 | **-2.59%** | 1518 | plateau |
| ibm17 | **1.3326** | 1.3830 | 1.6446 | **-3.64%** | 2238 | plateau |
| ibm18 | **1.3633** | 1.3865 | 1.7722 | **-1.68%** | 1589 | plateau |
| **AVG** | **1.1055** | **1.1193** | **1.4578** | **-1.23%** | **17 480** | **17/17 plateau** |

Every hard benchmark (ibm10/12/13/14/15/16/17/18) improved 1.7-3.6% via adaptive budget. Easy benchmarks (ibm01-04, 09, 11) tied within ±0.4%. ALL 17 benchmarks exited via plateau detection — none hit the 1hr cap. Plateau detection is a real win, not a budget-up-the-wall trick.

### What unblocked this

The CDOnly result (1.1193, matching leaderboard within 0.18%) showed that hard benchmarks ibm17/18/14/12 still had room — their last-CD-sweep deltas were 0.001-0.003, but the fixed 600s budget cut them off. E9's per-benchmark plateau detection gave them the time they needed (1500-2200s on hard, 300-600s on easy) without manual per-bench tuning. The 1-hour hard cap matches the competition rule and keeps total runtime bounded.

---

## CDOnlyPlacer --- Prior-prior Champion (superseded 2026-04-27 by CDAdaptive)

**Status: SUPERSEDED** --- avg proxy **1.1193** on --all (**23.2% better than RePlAce**, **matches leaderboard 1.1172 within 0.18%**, **19.1% better than prior DPO champion**).

Configuration: `submissions/cd_only/placer.py` (full-proxy coordinate descent on `IncrementalProxyEvaluator`, 600s budget per benchmark, SDF init). Runtime 10316s = 172 min total. Zero overlaps on all 17 benchmarks.

### How it works

1. **SDF init** --- non-overlapping starting placement
2. **Incremental evaluator** (`macro_place/incremental_evaluator.py`) --- bit-for-bit parity with `compute_proxy_cost`, **4657x speedup** on per-move cost queries via per-net min/max trackers + bin-density grid + RUDY congestion deltas
3. **Full-proxy coordinate descent** --- for each non-fixed macro, search both x and y axes; enumerate breakpoints (net endpoints + bin grid lines) and pick the proxy-minimizing position via incremental cost queries.
4. **Time budget** --- 600s wall clock per benchmark; sweeps continue until budget expires. Typical: 3-13 sweeps depending on macro count.

### Per-benchmark (zero overlaps everywhere)

| Bench | CDOnly | DPO BoV2 | RePlAce | Delta vs DPO |
|---|---|---|---|---|
| ibm01 | 0.9133 | 1.1285 | 0.9976 | -19.1% |
| ibm02 | 1.1534 | 1.6888 | 1.8370 | -31.7% |
| ibm03 | 0.9942 | 1.2508 | 1.3222 | -20.5% |
| ibm04 | 1.0193 | 1.3239 | 1.3024 | -23.0% |
| ibm06 | 1.1656 | 1.6434 | 1.6187 | -29.1% |
| ibm07 | 1.1105 | 1.3846 | 1.4633 | -19.8% |
| ibm08 | 1.1307 | 1.3816 | 1.4285 | -18.2% |
| ibm09 | 0.8606 | 1.0130 | 1.1194 | -15.1% |
| ibm10 | 1.1000 | 1.2540 | 1.5009 | -12.3% |
| ibm11 | 0.9248 | 1.0657 | 1.1774 | -13.2% |
| ibm12 | 1.2418 | 1.6497 | 1.7261 | -24.7% |
| ibm13 | 0.9939 | 1.2089 | 1.3355 | -17.8% |
| ibm14 | 1.2478 | 1.4725 | 1.5436 | -15.3% |
| ibm15 | 1.2109 | 1.3802 | 1.5159 | -12.3% |
| ibm16 | 1.1919 | 1.3594 | 1.4780 | -12.3% |
| ibm17 | 1.3830 | 1.5888 | 1.6446 | -13.0% |
| ibm18 | 1.3865 | 1.6352 | 1.7722 | -15.2% |
| **AVG** | **1.1193** | **1.3134** | **1.4578** | **-14.8%** |

Every single benchmark improves over DPO. No regressions.

### What unblocked this

The 2026-04-26 strategic update decomposed proxy cost as **WL 6%, density 20%, congestion 74%** (`analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md`). Pure HPWL coordinate descent (closed-form weighted-median) caps at ~5% improvement. **Full-proxy CD on a fast incremental evaluator** captures all three components at once. The infrastructure prerequisite (4657x speedup) gated the entire approach.

---

## See Also

- [approach.md](approach.md) --- methodology and architecture
- [roadmap.md](roadmap.md) --- next steps and submission plan
- [experiment_index.md](experiment_index.md) --- rigorous catalog of every experiment (live + falsified)
- [decisions/007_cd_lns_gridbin_promotion.md](decisions/007_cd_lns_gridbin_promotion.md) --- ADR for the E12 promotion
- `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` --- E8 proxy decomposition diagnostic
- `writeup/historical_results.md` --- DPO/Polyhedra/Overnight/Miftari per-benchmark data
- `writeup/closing_the_gap.md` --- leaderboard-beat narrative + design sketches
