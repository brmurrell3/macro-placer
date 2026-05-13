---
id: E96
name: multi_config_dp
status: in_progress
parent: PATH B B-R0' improvement — does DP basin quality vary with config?
created: 2026-05-13
decided: null
champion_at_time: 1.122 (finegrain_adaptive submission floor)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E96: DP config sensitivity probe + multi-config basin selector

## Hypothesis

The B-R0' DP+full-polish hybrid loses on **ibm10** (+1.6% vs cascade-capped)
but wins big on ibm12/14/17. Inspection of the per-bench delta from
`DP basin → greedy_legalize` shows a striking divergence:

| Bench | basin_proxy | legal_proxy | Δ legal | DP basin ovl |
|---|---:|---:|---:|---:|
| ibm09 | 0.954 | 0.955 | **+0.001** | 52 |
| ibm10 | 1.215 | **1.444** | **+0.229** | 310 |
| ibm12 | 1.442 | 1.477 | +0.035 | 98 |
| ibm14 | 1.494 | 1.523 | +0.028 | 113 |
| ibm17 | 1.543 | 1.572 | +0.029 | 138 |

**ibm10's legalize cost is 8× the others.** The DP basin on ibm10 is brittle:
the auto-config target_density (0.85 from macro_density × 1.5 clamp) packs
macros tightly. When legalize disperses the 310 overlapping macros, polish
can't fully recover (final 1.095 vs cascade-capped 1.0775).

**Question**: does varying DP config (target_density, density_weight, lr,
iter, gp_noise_ratio, random_seed) produce meaningfully different basins
on ibm10? If a different config yields lower legalize_proxy on ibm10,
multi-config basin selection (pick best legalize_proxy per bench) is a
high-EV layer on top of B-R0'.

## Method

8-config grid per bench, varying along salient axes:

1. **baseline_auto** — current B-R0' default
2. **td_low** — target_density - 0.20 (looser packing)
3. **td_high** — target_density + 0.10 (tighter packing)
4. **dw_low** — density_weight 2e-5 (weaker overlap penalty)
5. **dw_high** — density_weight 4e-4 (stronger overlap penalty)
6. **long_slow** — iter 4000, lr 0.002 (more careful convergence)
7. **high_noise** — gp_noise_ratio 0.10 (4× more init noise)
8. **seed_2024** — random_seed 2024 (different sample path)

For each config × bench, measure:
- basin_proxy
- basin_overlap_count
- legalize_proxy (after greedy_macro_legalize)
- legalize_overlap_count
- legalize_delta = legalize_proxy - basin_proxy

Phase 1 benches: ibm10 + ibm12 (one strong loser, one strong winner). If
the spread on ibm10 legalize_proxy is meaningful (>3%), extend to
ibm14/ibm17 + integrate into hybrid.

## Kill gate

- If max-min legalize_proxy spread on ibm10 < 2%: kill multi-config DP
  layer. Investigate **smarter legalizer** instead (E97).
- If max-min spread > 5% AND a non-baseline config beats baseline on
  legalize_proxy by >3%: implement multi-config selector in hybrid placer.
- If spread is 2-5%: marginal; consider as one component of a wider
  basin-diversity strategy.

## Generalization check

If implementing the selector: full polish on top 3 configs per bench
(top-3 lowest legalize_proxy), pick best polished. Validate on hard 4
(ibm10/12/14/17) before integrating into hybrid `--all`.

## Cost analysis

Per-bench probe: 8 DP runs × ~20-30s CPU + 8 legalize × ~5s = ~3-5 min.
Total Phase 1 (ibm10 + ibm12): ~10 min on cloud lambda.ai box. Far cheaper
than running full polish per config (~50 min each).

Multi-config DP layer for production: K=4 configs × ~25s = 100s extra DP
wall + 4 × 5s legalize = 120s extra per bench. With 3300s budget, 3.6%
overhead. If yields 3%+ basin improvement, net win.

## Pointers

- Driver: `code/dp_config_probe.py`
- Builds on B-R0' driver: `experiments/E91_dp_full_polish/code/dp_full_polish.py`
- Reuses DP bookshelf I/O: `experiments/E76_dreamplace_integration/code/`
- DP supports config knobs documented in: `~/DREAMPlace_cpu/install/dreamplace/params.json`
  (cloud); `random_seed`, `gp_noise_ratio`, `target_density`, `density_weight`,
  `iteration`, `learning_rate`, `stop_overflow` all settable per run.

## Outcome (filled when decided)
[Pending probe results from cloud lambda.ai.]
