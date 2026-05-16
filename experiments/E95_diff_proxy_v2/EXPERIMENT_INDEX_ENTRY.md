# Draft entry for docs/experiment_index.md (paste when sweep + chain complete)

```
| E95 | C1 spike v2 — DPO smooth proxy with WL normalization fix + γ annealing + L-BFGS strong Wolfe / projected SGD; basin-preservation test on cached cascade outputs (PATH C1 revival after E88 misdiagnosis) | `experiments/E95_diff_proxy_v2/code/{diff_proxy_v2.py,spike_v2.py,calibrate_v2.py,sweep_benches.py,walltrunc_test.py,spike_aggressive.py,calibrate_all_ibm.py}`, manifest, MORNING_NOTES.md | ibm01 0.8453→0.8457 (-0.047%); ibm10 0.000%; ibm12 -0.018%; ibm14 0.000%; ibm17 0.000% (sweep complete on cached cascade outputs); walltrunc + aggressive + 17-bench calibration pending | spike→sweep→walltrunc→aggressive→calibrate-all | **FALSIFIED 2026-05-13** — E88's WL norm bug (smooth used `len(plc.nets)` where canonical uses `plc.net_cnt`) was load-bearing for E88's misdiagnosis. Fix dropped ibm01 smooth-canonical gap from +2.1 % to +0.05 %. Deeper bug discovered: smooth `_rudy_congestion` (bbox-uniform-spread) diverges 3-4× from canonical `get_routing` (per-net trace-route) on ibm10/12/17 (matches well on ibm01 only). Cascade-uncapped sweep across 5 benches shows uniform basin preservation but zero lift — smooth gradient direction misaligned with canonical on hard benches. **Fix path: rewrite differentiable RUDY to match canonical trace-route algorithm (~2-3 days dev) = same primitive PATH B's E92/B-R3 needs.** Cascade saddle (E74/E84) is robust to this bug because eigenvector direction is scale-invariant under uniform bias; C1's gradient-descent isn't. Until diff RUDY is fixed, ANY smooth-proxy-gradient placer (DPO, C1, B-R3 with current ops) has miscalibrated congestion on hard benches. Two memory entries record this: `diff_proxy_wl_norm_gotcha.md`, `diff_proxy_rudy_mismatch.md`. |
```

Sweep numbers final 2026-05-13 ~02:15 UTC:
- ibm01 lift: -0.047 %
- ibm10 lift: +0.000 %
- ibm12 lift: -0.018 %
- ibm14 lift: +0.000 %
- ibm17 lift: +0.000 %

Walltrunc numbers (E25 with 300s budget → C1 polish):
- ibm10: e25=1.0695 → c1=1.0703 → **−0.07 %** (worse)
- ibm12: e25=1.2247 → c1=1.2247 → **+0.00 %** (no change)

Aggressive numbers (8 stages × 50 iters L-BFGS + restart):
- ibm10: 0.9894 → 0.9894 → +0.000 % (L-BFGS terminated early; smooth gradient ≈ 0)
- ibm12: 1.1977 → 1.1977 → +0.000 %

calibrate-all-ibm pending (chain2 in progress). Will update when complete.

Falsification class: **higher confidence than E88** because the root cause is now identified (RUDY mismatch on hard benches, not "structural objective mismatch"). E88's verdict "the differentiable approach has the same structural mismatch as DP" is **partly wrong**: it's the same mismatch *as DP's stock RUDY*, but the mismatch is fixable (with effort).

Related memory entries:
- `diff_proxy_wl_norm_gotcha.md` — the normalization bug
- `diff_proxy_rudy_mismatch.md` — the RUDY divergence finding
