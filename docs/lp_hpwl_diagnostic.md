# LP HPWL Lower Bound Diagnostic (E8)

Per-benchmark LP relaxation of HPWL (drop overlap constraints, fix initial-fixed macros and ports). LP-HPWL is an unattainable lower bound. Our HPWL is from BestOfV2Placer `--all` (BestOfV2Placer_20260426_220600.json). All HPWL values are normalized as in `plc.get_cost()`: `hpwl / ((W+H) * net_cnt)`.

| Benchmark | LP-HPWL  | Our HPWL | Gap %  | Our Density | Our Congestion | LP time (s) |
|-----------|----------|----------|--------|-------------|----------------|-------------|
| ibm01 | 0.016128 | 0.109791 | 580.76% | 0.5488 | 1.4887 | 2.65 |
| ibm02 | 0.012102 | 0.077072 | 536.83% | 0.8432 | 2.3802 | 6.26 |
| ibm03 | 0.016039 | 0.106957 | 566.86% | 0.5129 | 1.7749 | 4.17 |
| ibm04 | 0.012838 | 0.093274 | 626.56% | 0.5073 | 1.9540 | 5.94 |
| ibm06 | 0.006987 | 0.085023 | 1116.79% | 0.5093 | 2.6074 | 3.05 |
| ibm07 | 0.007862 | 0.087418 | 1011.87% | 0.5170 | 2.0774 | 6.41 |
| ibm08 | 0.012774 | 0.087685 | 586.41% | 0.5090 | 2.0789 | 15.21 |
| ibm09 | 0.008821 | 0.070151 | 695.28% | 0.5177 | 1.3679 | 7.24 |
| ibm10 | 0.010984 | 0.080174 | 629.90% | 0.5375 | 1.8101 | 40.60 |
| ibm11 | 0.008590 | 0.068487 | 697.24% | 0.5077 | 1.4867 | 8.53 |
| ibm12 | 0.011689 | 0.060013 | 413.43% | 0.8067 | 2.3728 | 58.88 |
| ibm13 | 0.008639 | 0.067693 | 683.58% | 0.5058 | 1.8243 | 11.05 |
| ibm14 | 0.004574 | 0.067018 | 1365.29% | 0.5056 | 2.3043 | 32.78 |
| ibm15 | 0.005010 | 0.069519 | 1287.59% | 0.5176 | 2.0919 | 17.44 |
| ibm16 | 0.005318 | 0.061282 | 1052.30% | 0.5253 | 2.1160 | 79.99 |
| ibm17 | 0.003754 | 0.068022 | 1712.10% | 0.4996 | 2.5885 | 63.06 |
| ibm18 | 0.002152 | 0.068568 | 3086.72% | 0.6198 | 2.5658 | 17.33 |

## Interpretation

### Proxy decomposition (verified)

Our `BestOfV2Placer` average proxy of **1.3834** decomposes exactly as:

| Component | Value | Coefficient | Contribution to proxy | % of proxy |
|---|---|---|---|---|
| Wirelength | 0.0781 | 1.0 | 0.0781 | **5.6%** |
| Density    | 0.5583 | 0.5 | 0.2792 | **20.2%** |
| Congestion | 2.0523 | 0.5 | 1.0262 | **74.2%** |
| **Total**  |        |     | **1.3834** | 100% |

(Reconstruction matches `avg_proxy_cost` in the BestOfV2Placer JSON to 4 decimal places.)

### What this means for the gap to 1.117

The gap from us (1.383) to the leaderboard (1.117) is **−0.266**. The maximum possible recovery from each component, holding others fixed, is:

| Component | Our value | Floor / lower bound | Max coefficient·gap |
|---|---|---|---|
| Wirelength | 0.0781 | 0.0091 (LP) | **0.069** |
| Density    | 0.5583 | ~0.50 (achievable on most ICCAD) | ~0.029 |
| Congestion | 2.0523 | unknown; RUDY proxies suggest ≥ 1.0 | **≥ 0.526** |

**The math forces the conclusion: if vmallela is at 1.117, congestion must be the dominant driver of their improvement.** Even closing 100% of the WL gap and meeting the density floor everywhere only buys ~0.10. Their remaining 0.166 reduction has to come from congestion. Our average congestion of 2.05 must be drivable down to roughly 1.0–1.3 to land near 1.11.

### Per-benchmark targets

The benchmarks with the **largest absolute proxy room** (high cost AND high congestion fraction) are where attack matters most:

- **ibm02 (1.689, congestion 2.38, density 0.84):** both density and congestion high — needs structural moves
- **ibm12 (1.650, congestion 2.37, density 0.81):** same profile
- **ibm06 (1.643, congestion 2.61):** congestion-dominant
- **ibm17 (1.612, congestion 2.59):** congestion-dominant
- **ibm18 (1.661, congestion 2.57, density 0.62):** all three

ibm09 (1.013) and ibm11 (1.066) are already low; little room.

### Strategy implications

1. **HPWL-only coordinate descent (E2 as originally framed) cannot reach 1.11.** Closing the entire WL gap saves ≤0.07. E2 should be redefined to do CD on the **full proxy** (numerical 1D line search per coordinate, since WL gives weighted-median but density+congestion don't have closed forms).
2. **Congestion-aware moves are the load-bearing piece.** The incremental evaluator (E1) MUST support fast congestion deltas, not just HPWL. If RUDY isn't decomposable per macro-move, this is the binding constraint on the whole program.
3. **LNS (E3) becomes higher priority than E2.** Rip-up-and-reinsert lets us re-route around congested regions in ways CD can't.
4. **DPO already had a congestion gradient.** The leaderboard winner achieving this in pure Python+numpy means there's a more efficient *exact* congestion descent we're missing — not just gradient methods.

### Footnote: the WL gap %

The WL gap to LP looks huge in percent terms (avg 979%, max 3087% on ibm18). This is misleading: the LP-HPWL on ibm17/ibm18 is essentially zero because those benchmarks' nets mostly terminate on pre-fixed ports, so the LP places everything at port locations. The ratio is meaningless when the denominator is near-zero. Read the **absolute** gap column (Our − LP) — that's at most 0.07 anywhere.
