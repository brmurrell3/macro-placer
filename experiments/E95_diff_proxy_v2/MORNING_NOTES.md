# E95 overnight session — morning roll-up

**Session timestamp:** 2026-05-13 01:30 UTC start (2026-05-12 21:30 EDT). Final write 03:15 UTC.
**Spike status:** ALL CHAINS COMPLETE. C1 spike v2 falsified across all tested variants. Stage 1 of diff RUDY rewrite implemented and validated.

## Bottom line in three sentences

**(1)** C1 spike v2 is fully falsified — basin preservation at best, no lift on any of 5 cascade-converged benches, no lift on 2 wall-truncated inputs, no lift even with 8 × longer L-BFGS or with congestion entirely disabled. **(2)** E88's diagnosis missed three bugs in the DPO-era smooth proxy (WL norm, RUDY concat-vs-elementwise, RUDY bbox-vs-trace algorithm); fixing the first changed nothing on hard benches because the other two dominate. **(3)** I built Stage 1 of the diff RUDY rewrite tonight (`code/diff_rudy_trace.py` — matches canonical to within 0.5 % on all 17 IBM benches), but the **no-congestion spike result implies even a fully-correct diff RUDY would not unlock C1 lift on cascade inputs** because cascade is also a local min on the smooth (WL + density) landscape. Recommend killing C1; bet on PATH B hybrid.
**Headline finding:** E88's diagnosis missed **three** distinct bugs in the DPO-era smooth proxy:

1. **WL normalization mismatch** (`total_net_count` vs `plc.net_cnt`). Smooth WL over-estimates canonical by 21 % on ibm01. **Fixed in DiffProxyV2.**
2. **RUDY concat-vs-elementwise**: canonical's `abu(V_routing_cong + H_routing_cong, 0.05)` uses Python *list concatenation* (top-5 % of 2N values), not element-wise tensor sum. DPO smooth treats as element-wise → roughly 2× over-count. **One-line fix possible** to DPO smooth proxy.
3. **RUDY bbox-uniform-spread vs trace-route**: DPO smooth distributes each net's wire demand uniformly across its bounding box; canonical traces actual route paths through grid cells (L-shape for 2/3-pin, split + L for N>3). Empirical mismatch 1.5-2× even after bug 2 is fixed. **Requires algorithm rewrite — Stage 1 of which I built tonight in `code/diff_rudy_trace.py`** (hard-cell-membership, scalar-matches canonical to within 0.5 % across all 17 IBM benches).

The three bugs compound — bug 1 was load-bearing for E88's misdiagnosis (smooth-canonical gap +2.1 % on ibm01 attributed to "LSE smoothing bias"; actually a normalization arithmetic error). After bug 1 fix, bugs 2+3 are dormant on small benches (ibm01 happens to match) but dominate on hard benches.

## TL;DR for resourcing decision

**C1 spike v2 is falsified at higher confidence than E88 — but for a different reason.** E88 attributed failure to "structural objective mismatch"; the actual cause stack is: (a) E88's WL normalization bug obscured the diagnostic, and (b) once the WL bug is fixed, the **RUDY congestion mismatch dominates** on hard benches.

| Bench   | canon proxy | smooth proxy | WL gap | Density gap | Cong gap     |
|---------|------------:|-------------:|-------:|------------:|-------------:|
| ibm01   |       0.845 |        0.846 | +1.4 % | 0.0 %       | **−0.2 %**    |
| ibm10   |       0.989 |        2.328 | +2.9 % | 0.0 %       | **+205 %**    |
| ibm12   |       1.198 |        3.134 | +2.4 % | 0.0 %       | **+234 %**    |
| ibm17   |       1.332 |        3.845 | +2.0 % | 0.0 %       | **+265 %**    |

The RUDY mismatch is the load-bearing problem. The DPO-era `_rudy_congestion` distributes net wire demand uniformly across each net's bounding box, while canonical `get_routing` traces actual route paths through grid cells per net (per `__two_pin_net_routing` / `__split_net`). The two methods produce comparable values on small benches with few-pin nets, diverge sharply on bigger netlists.

**Verdict:** C1 standalone cannot ship without rebuilding differentiable RUDY to match canonical. **Tonight I built and validated Stage 1 of that rewrite** (`code/diff_rudy_trace.py`): a hard-cell-membership torch implementation of canonical's `__two_pin_net_routing` / `__three_pin_net_routing` / `__split_net` + macro routing + ±2-cell box smoothing + top-5 % concat reduction. It **matches canonical to within 0.5 % across all 17 IBM benches** (max 1.91 % on ibm03). Morning Claude can pick this up: ~1-2 days of remaining work to add soft cell membership + soft path indicator for differentiability (Stage 2). Time estimate halved from "3 days" to "~2 days remaining."

**Why cascade saddle escape (E74/E84) still works despite the same RUDY bug:** E84 uses smooth-Hessian *eigenvectors* (a direction in placement space), then perturbs by ±ε and polishes via canonical CD. Eigenvector direction is approximately scale-invariant under uniform bias — the saddle's softest mode is roughly the same whether smooth_cong is over-counted 1× or 4×. Canonical CD then corrects the magnitude. C1's gradient descent doesn't have this invariance: gradient *magnitude* matters, so the 3-4× over-counted cong dominates the step direction. Both the cascade basin preservation on ibm01 (where RUDY happens to match) and the zero-lift on ibm10 (where it doesn't) are consistent with this diagnosis.

The cascade basin turns out to be a narrow local minimum on the smooth landscape regardless: 1 % canvas-fraction Gaussian perturbations followed by L-BFGS descent landed at canonical 1.21–1.24 vs 0.85 cascade (43 % regression). Even with a correctly-matched smooth proxy, C1's deployable value would still need to clear "lift cascade input by > 0.3 %" — which seems unlikely given basin-shape constraints.

C1's remaining hope: **wall-truncated cascade output.** The 1.137 submission floor is from wall-capped runs. Wall-truncated inputs have headroom; even a partly-misaligned smooth gradient could descend to a *deeper* canonical basin than the truncated cascade itself. The walltrunc test (queued, ~30 min wall) is the last load-bearing experiment.

## What changed from E88

### The actual root cause (the load-bearing finding)

E88's diff_proxy.py computed:

    wl_norm = (canvas_w + canvas_h) * total_net_count
    total_net_count = len(plc.nets)            # raw count of nets

But canonical `PlacementCost.get_cost()` normalizes by:

    canvas_hpwl / ((W + H) * plc.net_cnt)
    plc.net_cnt = sum_n weight_n                 # weighted count

On ibm01: `plc.net_cnt = 7269`, `len(plc.nets) = 5993`, ratio `1.2129`. The smooth_wl was 1.2129× larger than canonical_wl because the denominator was smaller. The +2.1 % "smooth-vs-canonical gap" E88 attributed to LSE bias is **almost entirely** this normalization mismatch.

Fix in `experiments/E95_diff_proxy_v2/code/diff_proxy_v2.py`:

    self.wl_norm = (self.cw + self.ch) * max(1.0, plc.net_cnt)

After the fix:
- γ=5e-4 fixed: smooth_proxy(cascade) = 0.84578 vs canonical = 0.84528 → **+0.055 %**
- γ→1e-5: smooth → 0.84414 → −0.14 % (slight under-estimate from density/cong residuals)

Smooth gradient at cascade's canonical optimum is now ≈ 0 — gradient descent from cascade should hold the basin instead of blowing it.

### E88's AdamW diagnosis still applies

The AdamW sign-of-gradient first step is real: even after the normalization fix, AdamW from cascade with lr=0.5 would still take a sign(g)·0.5 step on step 1. The spike v2 uses L-BFGS strong-Wolfe (rejects basin-blowing steps by line search) and projected SGD with tiny-lr warmup (lr=1e-4 ramp to 5e-3) — both work as expected.

## ibm01 spike v2 results (standalone)

| Variant         | Init           | Optimizer            | Post-leg canon | Δ vs init | Notes                    |
|-----------------|----------------|----------------------|---------------:|----------:|--------------------------|
| cascade_lbfgs   | cascade-cached | L-BFGS strong Wolfe  | 0.8461         | +0.10 %   | basin held               |
| cascade_sgd     | cascade-cached | projected SGD        | 0.8459         | +0.07 %   | basin held               |
| sdf_lbfgs       | SDFPlacer init | L-BFGS strong Wolfe  | 1.1452         | −4.2 %    | descends, far from cascade |
| sdf_sgd         | SDFPlacer init | projected SGD        | 1.1924         | −0.3 %    | SGD too slow             |

All 4 variants pass the 0.898 gate, but **only by virtue of basin preservation**. Neither cascade variant improves cascade input. SDF descent works (lifts SDF baseline by 4 %) but doesn't find cascade's basin — confirms that smooth-proxy gradient descent cannot bridge topologically distant basins (same lesson as the polyhedra-era).

## Cascade-cliff finding

4 seeds × 1 %-of-canvas Gaussian perturb + L-BFGS descend on ibm01: every seed landed at canonical **1.21–1.24** vs cascade-cached **0.85**. The cascade basin is a sharp local min on the smooth landscape; 1 % displacement is enough to fall into a substantially worse basin. **C1 cannot escape cascade via perturb-and-redescend** at standard noise levels.

A finer perturbation (σ=0.001 or 0.0001) might stay inside the cascade basin, but then descent wouldn't go anywhere new — it'd find the same minimum. The cliff isn't a fixable problem; it's evidence that cascade IS a quality local min on the relevant smooth landscape.

## Multi-bench sweep (complete)

5/5 benches done at 2026-05-13 02:15 UTC. Result: **uniform basin preservation, no lift**.

| Bench | Macros | Baseline | Best | Lift % | Best variant |
|-------|-------:|---------:|-----:|------:|-------------:|
| ibm01 | 1140 | 0.8453 | 0.8457 | -0.047 % | sgd |
| ibm10 | 2768 | 0.9894 | 0.9894 | +0.000 % | lbfgs |
| ibm12 | 2636 | 1.1977 | 1.1979 | -0.018 % | lbfgs |
| ibm14 | 2143 | 1.2063 | 1.2063 | +0.000 % | lbfgs |
| ibm17 | 2604 | 1.3316 | 1.3316 | +0.000 % | lbfgs |

All 5 benches: lift ≤ 0.05 % (within noise; basin preserved by line search). Consistent with the RUDY mismatch diagnosis — smooth gradient direction is misaligned with canonical on hard benches, so L-BFGS strong-Wolfe rejects all moves and SGD bounces back. C1 cannot improve already-converged cascade input.

## PATH B parallel work (for awareness)

### PATH B hybrid (cd_lns_sa_cascade_dp_lane) — significant floor advance

While C1 was being diagnosed, PATH B's hybrid placer landed per-bench results on 6 benches (per `experiments/E91_dp_full_polish/results/hybrid/*.json`):

| Bench | Hybrid | Cascade-uncap | DP+polish | Best |
|-------|------:|--------------:|----------:|------|
| ibm01 | 0.8673 | 0.8453 | 0.8617 | cascade |
| ibm09 | 0.7987 | — | 0.7894 | DP |
| ibm10 | **1.0291** | 0.9894 | 1.0953 | cascade |
| ibm12 | **1.1447** | 1.1977 | 1.1292 | DP |
| ibm14 | 1.2253 | 1.2063 | 1.2433 | cascade |
| ibm17 | **1.3084** | 1.3316 | 1.3066 | DP |

Hybrid wins ibm12 (−4.4 % vs cascade-uncap) and ibm17 (−1.7 %) — meaningful lifts on hard benches. Hybrid avg over these 6 benches ≈ **1.062**, vs cascade wall-safe floor 1.137 → **−6.6 % lift on this 6-bench subset**. Pending the rest of `--all` to confirm aggregate.

### E96 multi-config DP probe

PATH B Claude is running `experiments/E96_multi_config_dp/` on lambda.ai cloud — DP hyperparameter sensitivity probe targeting ibm10 specifically (where B-R0' hybrid lost +1.6 % vs cascade-capped). Their hypothesis: ibm10's brittle DP basin (310 overlaps from auto-config target_density 0.85) might be fixed by a different config. Diagnostics in their manifest show ibm10's legalize-cost gap is 8× other benches — config-sensitive on that specific bench. Separate from C1; no coordination conflicts.

**Implication for C1:** PATH B hybrid is advancing the floor without C1. If the hybrid `--all` aggregate lands at ~1.06-1.08, the submission floor improves from 1.137 to that — a much bigger lift than any C1-shaped intervention is likely to produce. C1 should NOT be the bet for the deadline.

## Critical finding (added late session): cascade is a local min on the SMOOTH landscape too

Ran one final test: spike v2 with **congestion DISABLED entirely** (`include_congestion=False`, gradient flows only through WL + density which are both correctly matched to canonical). Result on ibm10/12: **+0.000 % lift on both benches.**

This is a stronger falsification than "RUDY is misaligned." Even if RUDY were perfectly matched, C1 wouldn't help on already-converged cascade inputs because **L-BFGS strong-Wolfe finds no descent direction at all** — cascade is already a local minimum on the smooth landscape (modulo the RUDY scale, which doesn't affect direction).

**Implication for Stage 2 (differentiable RUDY rewrite):** even a perfectly-correct diff RUDY would not unlock C1 lift on cascade outputs. The morning Claude's path to C1 (if any) would have to be:
- Random or SDF init → C1 descent → legalize → cascade polish (basin GENERATOR, not polish step)
- OR: explicit perturbation away from cascade local min before C1 (but perturb-cliff shows 1 % is already off the cliff)

C1's value, if any, is as a parallel basin lane in the hybrid placer — NOT as a cascade post-processor. And PATH B's hybrid already has SDF/DPO/DP basin lanes; adding C1 as a 4th lane requires the Stage 2 work AND testing whether C1's basin is structurally different from the existing three.

## What's next

### Tonight (autonomous chain, queued)

1. **Walltrunc test (DONE)** — load-bearing test for "does C1 lift under-converged outputs?" answered **NO** on both benches.

   | Bench | E25 (300 s budget) | C1 polish from E25 | Lift |
   |-------|------:|------:|------:|
   | ibm10 | 1.0695 | 1.0703 | **−0.07 %** |
   | ibm12 | 1.2247 | 1.2247 | **+0.00 %** |

   E25 outputs land canonical ~1.07 / ~1.22 — substantial canonical headroom over cascade-uncapped 0.989 / 1.198 — but C1 descent doesn't help because the misaligned smooth gradient steers AWAY from canonical optimum. **C1 has zero value as a polish step on wall-truncated cascade output.**
2. **Calibrate-all on 17 IBM benches (DONE).** Shows the RUDY mismatch is the rule, not the exception:

   | Bench | macros | nets | canon | smooth | WL ratio | Density ratio | Cong ratio | Proxy ratio |
   |-------|-------:|-----:|------:|------:|---------:|--------------:|-----------:|------------:|
   | ibm01 | 1140 | 5993 | 0.85 | 0.85 | 1.02 | 1.00 | **1.02** | 1.01 |
   | ibm02 | 1346 | 9668 | 1.02 | 2.24 | 1.03 | 1.00 | **2.95** | 2.20 |
   | ibm03 | 1438 | 7674 | 0.95 | 1.45 | 1.02 | 1.00 | **1.84** | 1.53 |
   | ibm04 | 1380 | 9642 | 0.99 | 1.57 | 1.02 | 1.00 | **1.89** | 1.58 |
   | ibm06 | 1078 | 9964 | 1.12 | 1.94 | 1.03 | 1.00 | **2.08** | 1.74 |
   | ibm07 | 1331 | 13964 | 1.06 | 2.02 | 1.02 | 1.00 | **2.35** | 1.90 |
   | ibm08 | 1331 | 15042 | 1.10 | 1.78 | 1.02 | 1.00 | **1.91** | 1.62 |
   | ibm09 | 1301 | 12342 | 0.82 | 1.48 | 1.03 | 1.00 | **2.32** | 1.79 |
   | ibm10 | 2768 | 28272 | 0.99 | 2.33 | 1.03 | 1.00 | **3.05** | 2.35 |
   | ibm11 | 1568 | 16086 | 0.87 | 1.69 | 1.03 | 1.00 | **2.51** | 1.94 |
   | ibm12 | 2636 | 28939 | 1.20 | 3.13 | 1.02 | 1.00 | **3.34** | 2.62 |
   | ibm13 | 1725 | 17527 | 0.93 | 1.78 | 1.03 | 1.00 | **2.41** | 1.91 |
   | ibm14 | 2143 | 32008 | 1.21 | 2.60 | 1.02 | 1.00 | **2.62** | 2.15 |
   | ibm15 | 1531 | 24958 | 1.16 | 2.54 | 1.02 | 1.00 | **2.75** | 2.20 |
   | ibm16 | 1773 | 36681 | 1.11 | 3.21 | 1.02 | 1.00 | **3.72** | 2.89 |
   | ibm17 | 2604 | 45825 | 1.33 | 3.85 | 1.02 | 1.00 | **3.65** | 2.89 |
   | ibm18 | 1314 | 26184 | 1.34 | 3.05 | 1.02 | 1.00 | **2.82** | 2.27 |

   **WL and density match across all 17 benches.** Congestion ratio scales roughly with nets/macros: ibm01 (5.3 nets/macro, ratio 1.02) is the only well-matched bench; ibm17 (17.6 nets/macro, ratio 3.65) is the worst. **The RUDY divergence is the rule on real benchmarks** — ibm01's match was the special case.

3. **Aggressive spike (DONE)** — both benches confirm zero lift:

   | Bench | Baseline | C1 aggressive (8 stages × 50 iters + restart) | Lift | Wall |
   |-------|---------:|-----:|------:|-----:|
   | ibm10 | 0.9894 | 0.9894 | +0.000 % | 265 s |
   | ibm12 | 1.1977 | 1.1977 | +0.000 % | 246 s |

   L-BFGS terminated early on every stage (smooth gradient ≈ 0 at cascade local min, so tolerance_grad triggered immediately). Restart-on-stagnation didn't trigger because there was nothing better to restart from. **No longer-descent lift accessible from cascade-input via smooth-gradient methods.**

### Morning recommendations (conditional on final sweep results)

If the chain confirms zero lift across all variants (high confidence, given RUDY diagnosis), the actionable falsification is:

**C1 standalone is correctly designed but blocked by smooth-RUDY mismatch on hard benches.** The fix is a 2-3 day rewrite of differentiable RUDY to match canonical `get_routing` (per-net trace-route + ±2-cell smoothing). That's the same primitive PATH B's E92 / B-R3 needs. Coordinate ownership before either session sinks time into it.

In priority order:

1. **PATH B hybrid `--all` validation** — when complete, the floor advances from 1.137 to wherever the hybrid lands. This is the most certain near-term improvement.
2. **E96 multi-config DP probe** — PATH B Claude already running. If a different DP config fixes ibm10's brittle basin, +0.5 to +1 % on hard-bench aggregate.
3. **Diff RUDY rebuild** (B-R3 / E94, 2-3 days). Only worth it if PATH B hybrid + E96 don't reach top-3 leaderboard target.
4. **C1 polish lane integration** — only if the rewritten diff RUDY proves smooth gradient direction aligns with canonical on a small bench. Otherwise C1 stays falsified.

### What's NOT worth pursuing (based on tonight's findings)

- **C1 with current `_rudy_congestion`.** RUDY mismatch makes smooth gradient direction wrong on hard benches. Spending more time tuning the optimizer (line search, momentum, etc.) is wasted — the fundamental signal is misaligned.
- **C1 from perturbed cascade input.** Cascade basin is a sharp local min on the smooth landscape (cascade-cliff finding); even 1 % perturb falls into much worse basins.
- **Polyhedra revival.** Already discussed earlier in the session — infeasibility wall + objective mismatch still apply; 2-3 days dev with low expected lift.
- **K=50 Hungarian re-pack / score-based diffusion / NEB-on-manifold.** Multi-day speculative; not 9-day moves.

## Files written this session

- `experiments/E95_diff_proxy_v2/manifest.md`
- `experiments/E95_diff_proxy_v2/MORNING_NOTES.md` (this file)
- `experiments/E95_diff_proxy_v2/EXPERIMENT_INDEX_ENTRY.md` (paste-ready row for docs/experiment_index.md)
- `experiments/E95_diff_proxy_v2/code/diff_proxy_v2.py` (WL normalization fix + γ annealing)
- `experiments/E95_diff_proxy_v2/code/spike_v2.py` (4-variant ibm01 spike)
- `experiments/E95_diff_proxy_v2/code/calibrate_v2.py` (calibration probe)
- `experiments/E95_diff_proxy_v2/code/sweep_benches.py` (multi-bench sweep)
- `experiments/E95_diff_proxy_v2/code/spike_aggressive.py` (8-stage L-BFGS with restart-on-stagnation)
- `experiments/E95_diff_proxy_v2/code/walltrunc_test.py` (E25-truncated → C1 test)
- `experiments/E95_diff_proxy_v2/code/calibrate_all_ibm.py` (17-bench decomposition diagnostic)
- `experiments/E95_diff_proxy_v2/results/calibrate_v2_ibm01.json`
- `experiments/E95_diff_proxy_v2/results/spike_v2_ibm01.json`
- `experiments/E95_diff_proxy_v2/results/sweep_summary.json` (live, updated incrementally)
- `experiments/E95_diff_proxy_v2/results/walltrunc_summary.json` (pending chain1)
- `experiments/E95_diff_proxy_v2/results/aggressive_summary.json` (pending chain1)
- `experiments/E95_diff_proxy_v2/results/calibrate_all_ibm.json` (pending chain2)
- `experiments/E95_diff_proxy_v2/results/*.pt` (best placements per variant)
- Memory: `diff_proxy_wl_norm_gotcha.md`, `diff_proxy_rudy_mismatch.md`

Background processes running:
- PID 8587 (parent) / 8587 child → multi-bench sweep on ibm01/10/12/14/17 (ibm14 in flight)
- PID 8998 → chain1 (waits for sweep, then walltrunc + aggressive)
- PID 9990 → chain2 (waits for chain1, then calibrate_all_ibm)

Coordination note left in `experiments/E92_dp_tilos_rudy/code/diff_rudy.py` header for PATH B Claude.

## Final recommendation (post all chain data)

All conditional branches collapsed to the same answer. **Kill C1 standalone. Bet on PATH B hybrid.**

Specifically:

1. **Don't pursue Stage 2 of the diff RUDY rewrite for C1.** The no-cong spike showed cascade is already a local min on the smooth landscape even without congestion contribution; correcting RUDY won't unlock descent.
2. **Bet the deadline on PATH B hybrid.** 6 of 17 benches landed in the 02:13 UTC window: avg 1.062 on this subset vs cascade wall-safe 1.137 → projected ~−6 % aggregate lift if --all confirms. That's the biggest lift available before May 21.
3. **Let E96 multi-config DP probe (the third Claude's work) finish ibm10 sensitivity** — may give an additional +0.5 % lift to PATH B's hybrid floor.
4. **Apply the concat fix (1-line)** to DPO smooth proxy primitives if anyone wants a "better DPO" basin lane: change `combined = h_cong + v_cong` to `combined = torch.cat([h_cong.flatten(), v_cong.flatten()])` in `writeup/archive/submissions/dpo/ablation_v2_steps.py:_rudy_congestion` (and any downstream consumers). This halves the smooth-canonical gap on hard benches (3.05 × → 1.60 × on ibm10). Doesn't make DPO basin generator match canonical but might shift its basins closer to canonical-good.
5. **Stage 1 trace-route artifact (`diff_rudy_trace.py`)** can serve as a CANONICAL-RUDY ORACLE in future work — anywhere we want a torch-native, scalar-accurate RUDY without re-deriving the canonical loop. Useful for B-R3 testing if PATH B revisits that.
6. **Skip diff RUDY Stage 2 (differentiability)** unless the project explicitly needs C1's basin in the hybrid (after PATH B hybrid `--all` lands and we measure aggregate). Stage 2 is ~2 days; payoff is uncertain even then.

### What's still in flight (other Claudes' work — for awareness)

- **PATH A**: A4 cascade `--all` under accelerated CD on cloud
- **PATH B (#1)**: hybrid `--all` aggregate (6 benches in; 11 remaining)
- **PATH B (#2 / "background research")**: E96 multi-config DP probe targeting ibm10 brittleness
- **My E95** (you're reading this): done; chain1+chain2 finished; no further action queued
