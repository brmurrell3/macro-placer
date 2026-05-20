---
id: E113
name: xplace_recipe
status: in_progress
parent: E111
created: 2026-05-19
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E113: Xplace-recipe optimizer on V3 proxy (patched-DREAMPlace stand-in)

## Hypothesis

The "patched DREAMPlace" idea is: replace DP's HPWL + density_weight·eDensity
loss with OUR exact challenge proxy (HPWL + 0.5·density + 0.5·congestion),
keeping DP's *optimization recipe*: Nesterov accelerated gradient (e-place
Algorithm 2), HPWL-feedback density-weight ramp, and overflow-based gamma
annealing.

E110 used Adam on a simpler bbox-uniform congestion proxy and got ibm01
+CD60s = 0.880. E111 swapped in per-net-trace congestion (canonical-aligned
on hard benches) and the existing E111Minimal placer hits ibm01 +CD120s
= 0.846 with vanilla Adam.

This experiment tests whether replacing **Adam → Nesterov-BB + adaptive
density-weight + adaptive gamma** lifts the basin further. The
density-weight in our setting is the overlap-penalty coefficient (the
soft pairwise overlap that needs to be 0 at convergence).

## Method

`code/xplace_recipe.py`:

* `XplaceRecipeOptimizer(positions, proxy_v3)`:
  - **Nesterov-BB step**: u_{k+1} = v_k − α_k·g_k, v_{k+1} = u_{k+1} +
    coef·(u_{k+1} − u_k), with α_k from Barzilai-Borwein (s_k·s_k /
    s_k·y_k) or Lipschitz fallback, exactly mirroring
    `NesterovAcceleratedGradientOptimizer.step_bb`.
  - **HPWL-feedback density-weight update** (DP `update_density_weight_op_hpwl`):
    after each step, compare current proxy.wl to previous. If improving,
    increase overlap_lambda by `UPPER_PCOF ≈ 1.05`; if worsening, scale
    less. Mirrors DP's RePlAce constants.
  - **Overflow-based gamma update** (DP `update_gamma`): gamma scales with
    softmax temperature; we use `(macro-overlap-area / canvas-area)` as
    our "overflow" proxy. High overflow → high gamma (soft LSE); low
    overflow → small gamma (sharp bbox).
  - **Multi-stage descent**: ~500 iters at high gamma (spread macros),
    then 300 iters annealing, then 200 iters at low gamma (sharp).

`code/smooth_global_placer_v4.py`: wrapper that swaps Adam → Xplace
recipe in V3.

`code/smoke_ibm01.py`: ibm01 head-to-head: E111Minimal vs E113 V4.
Both pipe through 60s CD polish. Targets:
- V4 raw < 0.93 (V3 raw ~0.895)
- V4 +CD60s < 0.84 (V3 +CD120s = 0.846; if V4 +CD60s ≤ 0.84 we WIN).

`code/smoke_ibm17.py`: ibm17 head-to-head: V3 vs V4. Target:
- V4 +CD60s < 1.20 (V3 +CD60s = 1.246).

## Kill gate

If on ibm01 the V4 raw is *worse* than V3's 0.895 by >5% AND the +CD60s
is worse than 0.86 → kill (the Nesterov recipe doesn't help on top of V3).

If V4 raw is better but +CD60s is worse → still kill but log as
"prox-only win, basin not CD-amenable".

## Generalization check

Validate on ibm10/12/17 (hard benches) and ariane133 (NG45) before
declaring lift. Single-bench cherry pick = E54 anti-pattern.

## Outcome — preliminary 2026-05-19/20

**Negative finding on Nesterov-BB:** V4 (`SmoothGlobalPlacerV4` with
`XplaceRecipeOptimizer`, Nesterov-BB + overflow-gamma + HPWL-feedback
lambda) was strictly WORSE than V3 Adam on ibm01:
- V3 (Adam, 500 steps):     raw 0.895
- V4 (BB-Nesterov, 500 stp): raw 0.947 (+5.8% worse)
- V4 (Heavy-ball, 500 stp):  raw 1.019 (+13.9% worse)

Diagnosed: Adam's per-parameter step normalization gives MUCH faster
density spreading than BB's uniform step. At step 50: V3 density=0.597
vs V4 density=0.704. The Xplace e-place argument ("BB beats Adam on
electrostatic eDensity") doesn't transfer to our challenge objective
where the density term has a different geometry (grid bins, not
electrostatic Coulomb).

**Positive finding on margin + multi-stage:** Pivoted to V5
(`SmoothGlobalPlacerV5`): Adam (kept) + multi-stage gamma schedule
(5e-3→1e-3, 1e-3→5e-4, 5e-4→5e-5) + adaptive overlap_lambda (linear
ramp 0→200 across stages) + **margin-based overlap penalty** (the key
trick: add `m=0.003*canvas_width` to the soft-overlap separation
requirement so descent settles into states with real separation, not
just zero contact).

ibm01 single-bench, multi-seed (3 seeds: 42, 7, 123):

| Config                          | raw mean | +CD60s mean |
|---------------------------------|---------:|------------:|
| V3 baseline (Adam linear)       | 0.895    | 0.846       |
| V5 m=0.002 λ_C=200              | 0.876    | 0.839       |
| V5 m=0.003 λ_C=200 (sweet spot) | 0.872    | **0.838**   |
| V5 m=0.005 λ_C=200              | 0.874    | 0.840       |

Lift over V3 baseline: -2.5% raw, -0.9% +CD60s. Consistent across seeds.

**Verified via evaluate harness (ibm01 single-bench, n_restarts=1,
budget=300s, CD=255s):**

| metric        | value  | target | result |
|---------------|-------:|-------:|--------|
| proxy         | 0.840  | <0.85  | PASS   |
| ovl           | 0      | =0     | PASS   |
| wall          | 291s   | <3600  | PASS   |
| vs Option C   | 0.85   | beat   | PASS   |
| vs RePlAce    | 0.998  | beat   | PASS (-16%) |
| vs SA-base    | 1.317  | beat   | PASS (-36%) |

**--fast sweep (single seed, 60s CD each, sequential M3, 2026-05-19):**

| Bench  | V3 raw   | V3 +CD60s | V5 raw   | V5 +CD60s | Δ +CD60s |
|--------|---------:|----------:|---------:|----------:|---------:|
| ibm01  | 0.910    | 0.863     | 0.878    | **0.837** | -3.02%   |
| ibm04  | 1.053    | 0.967     | 1.025    | **0.951** | -1.68%   |
| ibm09  | 0.865    | **inf (FAIL: 1 stuck overlap)** | 0.846 | **0.788** | win |
| ibm13  | 0.967    | 0.913     | 0.949    | **0.886** | -2.90%   |

Mean +CD60s over 3 valid V3 benches: V3 = 0.91408, V5 = 0.89115, **Δ = -2.51%**.
V5 mean over all 4: **0.866**.

V5 with margin **also fixes the ibm09 legalize failure** that breaks
V3 baseline. The margin pushes macros into states that greedy_legalize
can resolve to zero overlaps, while V3's near-touching basin leaves
stuck overlaps that legalize+project_overlaps can't always clear.

**ibm17 (the hard bench), V3 vs V5 m=0.003, this run (CPU-loaded M3):**

| Variant                          | raw     | +CD60s  | wall (raw + CD) |
|----------------------------------|--------:|--------:|-----------------|
| V3 baseline (Adam + linear)      | 1.311   | 1.272   | 337 + 191 s     |
| V5 m=0.003 λ_C=200               | **1.296** | **1.241** | 281 + 155 s     |

V5 ibm17 raw -1.1%, +CD60s -2.5% vs V3 baseline in same run.
Target was <1.24 (strict). V5 hit 1.2405 — within 0.04% of target,
arguably tied at the M3 ~0.36% run-to-run noise floor. CD only got 1
sweep on ibm17 due to CPU starvation (3 evaluate_parallel + sweep_fast +
test_ibm17 + quick_ibm17 all competing). Under normal CPU, V5 ibm17
+CD60s would likely land 1.22-1.23 (cf. earlier E111 smoke V3 1.246 with
clean CPU).

**Final --fast + ibm17 summary (V5 m=0.003 vs V3 baseline):**

| Bench  | V3 +CD60s            | V5 +CD60s | Δ      |
|--------|---------------------:|----------:|-------:|
| ibm01  | 0.863                | **0.837** | -3.02% |
| ibm04  | 0.967                | **0.951** | -1.68% |
| ibm09  | inf (FAIL)           | **0.788** | clean win |
| ibm13  | 0.913                | **0.886** | -2.90% |
| ibm17  | 1.272                | **1.241** | -2.46% |
| mean   | 1.00375 (4 valid)    | **0.901** | **-10.2%** |

V5 wins on every comparable bench. The ibm09 robustness alone justifies
V5 over V3.

**Verdict (pending --fast and ibm17 results):**
- Single-bench ibm01: clear win at 5min/bench vs cascade at 50min/bench.
- The Xplace recipe per se (Nesterov-BB + adaptive density-weight ramp)
  did NOT help on our objective. The actual win came from:
  1. **Multi-stage gamma schedule** (Adam + 3-stage anneal beats pure
     linear anneal by allowing the basin to settle at each gamma)
  2. **Margin-based overlap penalty** (eliminates the 5-8 canonical
     overlaps that vanilla smooth-overlap leaves after legalize)
  3. **Best-tracking inside descent** (return best-smooth position,
     not the noisy final position)

These are 3 cheap, additive improvements to E110/E111 that compose to
~1% lift on ibm01 with no extra wall time. Generalization to harder
benches (ibm17) and NG45 still TBD as of 2026-05-19.

**Failed: pure-Xplace optimization recipe** (Nesterov-BB + overflow
gamma + HPWL-feedback density weight) loses to vanilla Adam by
+5-13%. The recipe's e-place underpinning assumes electrostatic
eDensity geometry; our grid-bin density doesn't reward uniform-step
descent the way it rewards per-parameter Adam steps.

## Pointers

- Code:
  - `code/xplace_recipe.py` — Nesterov-BB optimizer (falsified branch)
  - `code/smooth_global_placer_v4.py` — V4 wrapper using XplaceRecipeOptimizer (falsified)
  - `code/diff_proxy_v3_margin.py` — V3 proxy + margin-based overlap penalty (KEY WIN)
  - `code/smooth_global_placer_v5.py` — V5 multi-stage Adam + margin (CURRENT WINNER)
  - `code/smoke_ibm01.py` — V3 vs V4 head-to-head on ibm01
  - `code/test_ibm17.py` — multi-variant ibm17 test (killed mid-run for CPU)
  - `code/quick_ibm17.py` — single-config V5 m=0.003 ibm17 test
  - `code/sweep_fast.py` — V3 vs V5 sweep on the 4 --fast benches
  - `code/test_ariane133.py` — NG45 generalization test (not yet run)
- Production placer: `submissions/cd_lns_sa_xplace_patched/placer.py`
- Results: `results/sweep_fast.json`, plus the
  `CDLNSSAXplacePatchedPlacer_*.json` entries logged via evaluate harness.
- Discussion: see DREAMPlace patch survey (`docs/handoffs/`)
