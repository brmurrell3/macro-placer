# TODO — strategic plan

**Deadline:** May 21, 2026 (~9 days)

## Submission status

**First submission ready.** Resubmission allowed by partcl, so:

1. Ship `submissions/cd_lns_sa_cascade/placer_adaptive.py` now as the floor
   (1.137 IBM / 0.6925 NG45 / max wall 57 min / beats RePlAce by 22 %).
2. Overwrite with each iteration that beats the floor.

Cleanup landed 2026-05-12 (see bottom of this doc).

---

## PATH A — speed up our champion CD pipeline  ★ HIGHEST EV

**Premise:** cascading saddle escape verified canonical **1.0612 on 17 IBM
uncapped** (`experiments/E84_cascading_saddle/results/`). Wall-safe
variant plateaus at 1.137 because **96 % of CD time is single-threaded
Python** (`IncrementalProxyEvaluator.move()` + `revert()`, confirmed by
cProfile on ibm04: 32 s of 33 s). A 10–30× CD speedup unlocks the
cached-quality basin under the 60-min cap.

### A1. Eliminate `revert()` via pure delta function — **DONE 2026-05-12**

Final speedup: **5.36× on ibm10 / 5.12× on ibm01** vs original
move+revert. End-to-end smoke (ibm01, 180 s budget): proxy 0.88148 vs
cleanup baseline 0.89195 (**−1.17 %** at same wall), zero overlaps.
Per-probe parity perfect to 2e-16 (machine epsilon) on 100 probes
each on both benches.

Three landed commits:

- **`59a7a8b`** *phase 1 — single-candidate `delta_cost`* (1.95× per probe).
  `IncrementalProxyEvaluator.delta_cost(macro_idx, new_xy)` returns
  the cost dict `move + current_cost + revert` would yield, without
  mutating state. Wired into `cd_core.search_axis`.
- **`53b4a26`** *phase 2 — batched `delta_cost_axis_batch`* (2.32× more,
  4.32× combined). Amortizes "subtract old" precompute once across K
  candidates; uses `index_put_(accumulate=True)` for [K, num_cells]
  scatter-adds; runs density + congestion cost as batched torch ops
  via `_density_cost_of_batched` / `_congestion_cost_of_batched` /
  `_smooth_batched`.
- **`af520c3`** *flat routing variant* (1.18× more, 5.36× combined).
  `_net_cong_contrib_flat(net_idx)` returns flat
  `(h_cells, h_vals, v_cells, v_vals)` lists instead of a dict.
  Vectorizes pin → gcell computation. Inlines 2-pin and multi-pin
  routing topology (the common cases). Duplicates tolerated downstream.

**Further A1 micro-optimizations stopped here** — the remaining
bottleneck is Python overhead inside `_net_cong_contrib_flat` (~10 µs
per call across 57 k calls per 200 axis-searches; cProfile-confirmed).
The remaining ~2× to hit the original "10×" target would need Numba/Cython
(adds a build/runtime dependency to the submission) or a structural GPU
port (A3 — multi-day refactor). Not worth it now — better-EV work waits
on PATH C.

### A2 / A3 — deferred (diminishing returns post-A1)

After A1, `move()` is called once per *accepted* candidate (1× per
macro per CD sweep) instead of K+1 times. Its absolute share of CD wall
dropped from 96 % to <20 %. Cython/Numba porting `move()` (A2) or
GPU-batching the routing (A3) would yield further speedup but the
remaining absolute time is small. **Both deferred** pending PATH C
results.

### A2. Cython/Numba port of `move()`  *(2–3 days)*
- Current: ~100 lines of Python dict updates over `affected_nets`,
  `macro_cong_contrib`, `H_net_cong` etc.
- Port to native; explicit memory layout for the dict-heavy paths.
- Expected speedup: 20–100× on move/revert.
- Composes with A1 (post-revert-elimination).

### A3. Batch candidate eval as tensor op  *(2–4 days, optional GPU port)*
- For macro M, axis A, K candidates → single `[K]` tensor op.
- Reuses `experiments/E87_gpu_cd/` scaffold but targets the **exact** proxy delta
  (not the smooth proxy approximation E87 currently uses).
- Blocker to fix first: `_extract_net_data` (lives in
  `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py:80`) has an O(N²)
  hot spot — must profile + rewrite before GPU port helps.

### A4. Validate cascade `--all` under accelerated pipeline  *(1 day)*
- Re-run cloud cascade `b=3000` with accelerated CD.
- Target: IBM aggregate ≤ 1.08 (matches uncapped 1.0612 ± wall-cap slop).
- NG45 aggregate ≤ 0.68.

### A5. Final submission packaging + ORFS Tier 2 scoping  *(2–3 days)*
- Form fill, evidence collation, freeze final placer.
- Test ORFS flow locally on one NG45 design to validate Tier 2 readiness.

**A active code:**
- `submissions/cd_lns_sa_cascade/{placer.py, placer_adaptive.py, README.md}`
- `submissions/cd_lns_sa/placer.py` (E25 component)
- `experiments/E41_dpo_kjoint/code/` (E41 component)
- `experiments/E74_hessian_saddle/code/` (saddle escape primitives — `hessian_saddle.py`, the actual `_extract_net_data` consumer)
- `experiments/E84_cascading_saddle/code/` (cascade logic) + `results/` (cached .pt files)
- `experiments/E87_gpu_cd/code/` (in-progress GPU CD scaffold — A1-A3 work happens here)
- `macro_place/incremental_evaluator.py`, `macro_place/cd_core.py` (hot paths to optimize)

---

## PATH C — three untried algorithmic directions (ranked by EV)

**Why these matter:** PATH A's ceiling is the uncapped cascade result on
cloud — likely ~1.06–1.08. That's a meaningful submission floor but
still **~5 % above the leaderboard top (~1.01)**. To beat 1.01 we need
basin or polish improvements PATH A can't deliver. The DP falsification
(PATH B) ruled out surrogate-objective placers — the next moves
optimize canonical directly or push the saddle escape further.

### C1. Differentiable canonical proxy + GPU gradient descent  ★ highest EV here

**The DP failure is the strongest argument for this.** If surrogates
(HPWL + Gaussian density + DP-RUDY) are structurally misaligned with
canonical (top-K density + TILOS-RUDY with TILOS weights), then optimize
canonical directly.

**Plan:**
- Re-implement `macro_place/objective.py` end-to-end in PyTorch:
  - HPWL via LogSumExp-smoothed max
  - Top-K density via softmax-weighted differentiable top-K
  - TILOS-RUDY in PyTorch (small extension of standard RUDY, weights already in `_plc.py`)
  - Pairwise overlap as Lagrangian penalty with annealed multiplier
- Optimize placement tensor on A100 with AdamW or L-BFGS.
- Snap to feasible (use the legalizer from PATH B's `macro_legalizer.py`).
- Cascade polish to recover canonical proxy.

**Spike gate:** 1-day local feasibility check — implement differentiable
HPWL + Gaussian density only, run AdamW on ibm01, must land within 5 %
of cascade's 0.85527 cached. If the spike clears, commit 4–5 more days.

**Engineering:** 3–4 days for full differentiable proxy +
canonical-agreement validation (must match `compute_proxy_cost` to <0.5 %
on a fixed placement); 1 day for the gradient-descent driver.

**Composes with A1/A3:** the batched-delta work in A3 is a stepping
stone — once delta becomes a tensor op, autograd gives the gradient for
free.

**Risks:** smoothing artifacts may drift far from canonical optima;
projection step may destroy lift. Mitigate with
smoothing-temperature annealing + periodic canonical evals + final
cascade polish.

### C2. ILP detailed legalization on cascade's output  *(low risk, stacks)*

Post-cascade, residual positions are near-optimal modulo small packing
decisions. HiGHS (`pip install highspy`) ILP on small candidate grids
+ WL minimization at fixed overlap=0. Likely **0.3–1 % lift per bench**,
~1–2 days to implement. Won't beat 1.01 alone; stacks cleanly with
everything else.

**Engineering:** mostly mechanical — solve N small ILPs (one per
overlap cluster after cascade), pick the position that minimizes WL
delta within the cluster.

### C3. Newton-CG / trust-region saddle escape  *(unfinished E74 territory)*

E74/E84 use only the smallest eigenvector + ε-step heuristic. With HVP
infrastructure already built in `experiments/E74_hessian_saddle/code/
hessian_saddle.py`, do actual second-order optimization: Newton-CG with
trust region, or BFGS in the soft-mode subspace. Each escape iteration
extracts more curvature info than a single eigvec step.

**Engineering:** ~2–3 days. Reuses `SmoothProxy` and
`find_softest_eigenvectors`; adds a trust-region / line-search wrapper
that uses the full HVP rather than just the leading eigenvector.

**Composes with cascade:** drop in as a replacement for the ε-step inner
loop in `cascading_saddle.py`.

---

## PATH B — DREAMPlace exploration  *(hybrid --all VALIDATED 2026-05-13)*

### TL;DR (final, 2026-05-13 10:52 UTC)

Overnight queue on lambda.ai completed all 17 IBM + 4 NG45 hybrid runs.
Auto-adaptive DP + two-stage retry + best-of-3 plateau pick + cascade
saddle. Reference: cascade `a4v2_postLNS` from PATH A.

| Set | Hybrid | Cascade A4v2 | Lift |
|---|---:|---:|---:|
| IBM 17 | **1.06687** | 1.07711 | **−0.95 %** |
| NG45 4 | **0.68086** | 0.68483 | **−0.58 %** |
| 21-bench combined | **0.9933** | 1.0024 | **−0.91 %** |

Big wins: ibm12 −5.35 %, ibm17 −3.84 %, ibm07 −3.13 %, ibm09 −2.52 %,
ibm03 −2.46 %, mempool_tile −2.28 %, ibm01 −2.11 %, ariane133 −2.05 %.

Regressions: ibm02 +2.99 %, ariane136 +1.76 %, ibm10 +1.28 %,
ibm18 +1.13 %, ibm08 +1.00 %.

ariane133 **0.66167 beats E74 hessian-saddle 0.6641 (−0.36 %)** — the
NG45 failure point that killed E42/43/44/54/62 is solved with auto-
adaptive DP. NG45 generalization holds.

**Cumulative session lift** (since start 2026-05-13):
- `placer_adaptive.py` (pre-A1): IBM 1.137
- cascade A4v2 (PATH A's A1+LNS-skip on `placer_adaptive`): IBM 1.077 (−5.3 %)
- hybrid (PATH B's DP lane added): IBM 1.067 (−0.95 % on top of A4v2)
- **Combined: −6.2 %** on IBM submission floor.

Gap to leaderboard top (~1.01): IBM 1.067 = **+5.7 %** above first
place. Still not there. PATH B added a real but modest lift; the bulk
of the session's gain came from PATH A.

**Submission floor candidate:** `submissions/cd_lns_sa_cascade_dp_lane/
placer.py`. Pending: resolve ibm17 wall-budget violation (60.9 min on
60-min cap — needs CD cap tightening from 0.14×B to 0.12×B, or
DP-skip-if-time-short rule). Until that's fixed, keep cascade
`placer_adaptive.py` as the official submission. See
`experiments/E91_dp_full_polish/SUMMARY.md` for the full per-bench
table.

**E96 multi-config DP variant** (other agent) is the natural follow-up,
projected to add −0.5 to −1.5 % on top of hybrid. Awaiting their `--all`
validation.

### Archive (full B-R0' analysis below — kept for traceability)

### 2026-05-12 PM: autopsy revised. Original falsification was basin-only.

### 2026-05-12 PM: autopsy revised. Original falsification was basin-only.

Verifying the PATH B falsification revealed that **stock DP basin +
full cascade-equivalent polish was never tested**. The autopsy table
below compared DP-after-greedy-legalize *basins* vs cascade
post-polish results — not DP basin under the same polish budget.

Evidence:
- `experiments/E76_dreamplace_integration/code/dp_sweep_b1.py:10-12`
  states: *"no CD polish at sweep time; cheap evaluation"*. The sweep
  numbers in the table are basin-only.
- `p1b_polish_test.py` only ran 15-min CD-adaptive (no LNS, no SA, no
  saddle escape) — far less than the ~55 min cascade uses.
- `submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py:192`
  gives DP only `min_time_s=30, hard_cap_s=60` brief CD cleanup while
  E25 and E41 each receive ~660s polish in the plateau pick.

Original autopsy table (kept as basin-only reference):

| Bench | Best DP **basin** | cascade b=3000 *post-polish* | Gap |
|-------|--------------:|---------------:|----:|
| ibm10 | 1.2553 | 1.0775 | **+16.5 %** |
| ibm12 | 1.3712 | 1.3031 | +5.2 % |
| ibm14 | 1.4176 | 1.2919 | +9.7 % |
| ibm17 | 1.5388 | 1.4546 | +5.8 % |

This table is *not* evidence of structural objective mismatch under fair
polish. SDF basin starts at ~1.50 and polishes through E25 to ~1.09
(−27 %); DPO basin at ~1.38 polishes through E41 to ~1.08 (−22 %). If
DP-legalized basins polish similarly, ibm10 (1.44 post-legalize) might
land at 1.04–1.12, around or below cascade-capped 1.0775.

What the original autopsy correctly falsified: **tuning stock DP as a
black-box basin generator with handicapped polish**. What it did NOT
falsify: stock DP + full polish, modifications to DP's loss ops, or
DP-as-perturber on a cascade init.

Code preserved at `experiments/E76_dreamplace_integration/` and
`submissions/_archive/falsified/cd_lns_sa_hessian_dp/`. The greedy macro
legalizer (`macro_legalizer.py`) is a reusable utility regardless.

### B-R0' (NEW) — Stock DP basin + full E25-equivalent polish + cascade saddle

**The proper falsification test that was never run.** Pipeline:
DP-stock → `greedy_macro_legalize` → full polish (CD-adaptive 660s +
LNS-gridbin 192s + SA-v2 192s) → cascading saddle escape (~1900s).
Same budgets as cascade gives its own SDF/DPO inits.

Driver: `experiments/E91_dp_full_polish/code/dp_full_polish.py`.
Cloud setup: lambda.ai A100 box at 129.213.18.245, DREAMPlace built
in Docker at `~/DREAMPlace_cpu/install` (CPU build — CUDA build
hit nvcc 11.0 vs compute_86 incompatibility; CPU DP runs ibm10 in 12s).

Status 2026-05-13 00:15 UTC: **all 6 B-R0' benches complete. Autopsy
empirically falsified on 3 of 4 hardest IBM benches.**

| Bench | DP+full-polish | Cascade-capped (autopsy) | Δ |
|---|---:|---:|---:|
| ibm01 | 0.862 | 0.85 (uncapped) | +1.4 % |
| ibm09 | 0.789 | (no autopsy ref) | — |
| ibm10 | 1.095 | 1.0775 | **+1.6 %** |
| ibm12 | **1.129** | 1.3031 | **−13.3 %** |
| ibm14 | **1.243** | 1.2919 | **−3.8 %** |
| ibm17 | **1.307** | 1.4546 | **−10.2 %** |

**Hard-bench aggregate** (ibm10/12/14/17): cascade-capped 1.2818 →
DP+full-polish 1.1936 = **−6.9 % lift**.

ibm12 at −13.3 % is striking — the autopsy reported only +5.2 % basin
gap there, then concluded the gap was unbridgeable. With fair polish
budget, DP basin polishes to a placement *substantially better* than
cascade's polished result. Same pattern on ibm17 (−10.2 %) and ibm14
(−3.8 %). Only ibm10 loses marginally (+1.6 %).

For a hybrid placer (E25 SDF + E41 DPO + **DP** lanes, all with fair
polish), see `submissions/cd_lns_sa_cascade_dp_lane/placer.py`. Drop-in
extension of cascade with DP as a third basin lane.

The original autopsy ("structural objective mismatch can't be bridged")
is wrong because basin-only quality doesn't predict polished quality
in this codebase. With fair polish budget, DP basins polish to a
different valley than SDF/DPO inits — sometimes *better* than the
cascade-capped result.

### Hybrid IBM results 2026-05-13 01:30 UTC (pre-rule-compliance config)

Hybrid placer on 6 IBM benches (3 lanes × polish + cascade saddle).
**These runs used the hardcoded `target_density=0.85` DP config** that
predated the rule-compliant patch. Phase 4 of the overnight queue
reruns hybrid with auto-adaptive config to get rule-compliant numbers.

| Bench | Hybrid | Cascade-capped | Δ | Winner lane |
|---|---:|---:|---:|---|
| ibm01 | 0.867 | 0.85 (uncapped) | +2.0 % | cascade-on-DP plateau |
| ibm09 | 0.799 | (no ref) | — | cascade-on-DP plateau |
| ibm10 | 1.029 | 1.0775 | **−4.5 %** | cascade-on-E41 plateau |
| ibm12 | 1.145 | 1.3031 | **−12.2 %** | cascade-on-DP plateau |
| ibm14 | 1.225 | 1.2919 | **−5.2 %** | cascade-on-E41 plateau |
| ibm17 | 1.308 | 1.4546 | **−10.1 %** | cascade-on-DP plateau |

**Hard-bench (ibm10/12/14/17) aggregate**: hybrid 1.177 vs cascade-capped
1.282 = **−8.2 % lift**. Plateau pick correctly picks DP on the wins
(ibm12, ibm17), E41 on the cases where DP doesn't help (ibm10, ibm14).

### Rule-compliance fix — auto-adaptive DP config

Original PATH B test used hardcoded `target_density=0.85`. That worked
for IBM but failed on ariane133 (3 residual overlaps, +11.6 % vs
cascade-uncapped). First fix attempted per-benchmark branching
(`if bench.name.startswith("ibm")`) — **against challenge rules**.

Replaced with a single algorithm derived from observable bench geometry:

```
macro_density = sum(macro_area) / canvas_area  # observable from input
target_density = clip(macro_density * 1.5, 0.40, 0.85)
stop_overflow = 0.02  # universally tighter than original 0.07
dp_iter = 2000
dp_lr = 0.005
```

Same formula applied to every input. Plus cleanup paths: stricter
cascade saddle budget check (rolling avg iter wall), `extended_legalize`
fallback in `_polish_dp_basin`, and "DP lane returns None if residuals
remain" filter on plateau pick.

### ariane133 NG45 generalization — VERIFIED with auto-adaptive

ariane133 with auto-adaptive config: macro_density=0.496 →
target_density=0.743 (auto). Final: **0.66993 with 0 overlaps**, total
wall 34 min. **vs cascade-uncapped 0.6641: only +0.88 % (tied)**.

NG45 generalization holds. PATH B with auto-adaptive config works on
both IBM and NG45.

### Overnight queue status (launched 2026-05-13 01:20 UTC)

Cloud chain on lambda.ai:
- Phase 1 (running): IBM auto-config validation on ibm10/12/14/17.
- Phase 2 (queued): NG45 set — ariane136, mempool_tile, nvdla.
- Phase 3 (queued): remaining 11 IBM (ibm02/03/04/06/07/08/11/13/15/16/18).
- Phase 4 (queued): full hybrid `--all` on 17 IBM + 4 NG45 with auto-adaptive.

Expected completion ~07:00 UTC = 03:00 EDT. Total ~10 hr cloud work.
Master log: `/tmp/overnight_master.log`. Phase logs: `/tmp/overnight/`.

### Live routes — modify DP's internals, not its hyperparameters

Ranked by information-per-day. R1 is the diagnostic that conditions
R2/R3.

#### B-R1. Add TILOS-RUDY as DP loss term — diagnostic spike  *(1–2 days, DEFERRED)*

**DEFERRED 2026-05-13** pending hybrid `--all` and ariane133 results.

Original intent: replace DP's RUDY op with TILOS-RUDY to fix routing-loss
mismatch. But B-R0' empirically shows stock DP + full polish already beats
cascade-capped by 4-13 % on 3 of 4 hardest IBM benches without any
custom loss work. The gap that B-R1 was supposed to bridge is **already
bridged by polish** on those benches.

**Reactivate B-R1 if (any of):**
- Hybrid `--all` aggregate stays above 1.10 (i.e. easy benches aren't
  lifting alongside hard ones, suggesting custom loss might help easy).
- ariane133 regresses (NG45 generalization fails) — custom loss might
  shift DP's basin toward NG45-friendly direction.
- Top-3 leaderboard reach requires another -2 to -3 % aggregate lift.

**Status of work:**
- E92 manifest written with implementation plan
- `experiments/E92_dp_tilos_rudy/code/diff_rudy.py` prototype written
  (torch-native differentiable RUDY, not yet vectorized or hooked into
  DP's PlaceObj.obj_fn)
- Engineering needed: vectorize the per-net loop, fork
  `~/DREAMPlace_cpu/install/dreamplace/PlaceObj.py` to add congestion
  term to obj_fn, rebuild DP, calibrate weights, re-test.

#### B-R2. Replace eDensity with differentiable top-K density  *(3–5 days)*

eDensity (electrostatic field via FFT-Poisson) is what makes DP fast on
millions of std cells but it's optimizing uniform-spread, not "no bin
exceeds threshold." Canonical top-K density is
`softmax(densities/τ) · densities` summed over highest-K bins —
trivially differentiable, tiny tensors at 200–537 macros. Forking DP's
density op is one well-isolated module. After this change DP's Nesterov
optimizer is descending the *correct* density surface.

#### B-R3. Full canonical losses on DP substrate — convergent with C1  *(7–10 days)*

B-R1 + B-R2 + WA-WL→smoothed-bbox-HPWL gives a fully canonical loss set
running on DP's machinery (Nesterov-with-noise, FFT acceleration, GPU
init strategies). This **is** C1, implemented on DP's substrate instead
of from scratch — the decision becomes "which vehicle gets canonical
losses bolted on." DP gives a 5-year-optimized GPU optimizer for free
but adds build-system and op-registration dependency.

Commit decision: only after B-R1 + B-R2 actually close the gap on a
hard bench. If they don't, write C1 standalone (no DP dependency).

#### B-R4. Inverted use — cascade-as-init, DP as basin-escape perturber  *(2–3 days)*

Untested mode: cascade→DP-gradient-steps→cascade-repolish. PATH B
tested only DP→cascade. Cascade lands at ~1.06 and stalls on local-move
plateau; DP's gradient noise could shake out non-local. Risk: still
pulls toward DP's wrong basin and *worsens* canonical. But cheap, cleanly
parallel to B-R1/R2/R3 (no shared code paths), and addresses a different
failure mode (escaping cascade plateau) than C1.

### Sequencing — revised after autopsy reverification

1. **B-R0' first** (NEW). Stock DP + full polish, the proper test that
   was never run. Cheap, in progress on lambda.ai. If it lands within
   3 % of cascade on ibm10, the autopsy was wrong and B-R1+ are
   downstream tunes on an already-competitive lane. If it lands +5%+
   above cascade, the autopsy holds at higher confidence and B-R1
   becomes the diagnostic that decides between routing- vs density-
   driven gap.
2. **B-R1** after B-R0'. Yes/no diagnostic for the *remaining* gap.
3. **B-R4 in parallel** — different code paths, different failure mode.
   Initial result (PERTURB_ITERS=100, lr=0.005): DP gradient on cascade
   init **diverged** (proxy 0.99 → 6.18, 308 k overlaps after 100 iter).
   Cascade-init lives at a canonical local minimum that's *not* a DP
   local minimum; DP's gradient drives away from it. Retry with
   iter=10 to see if a small perturb survives re-polish.
4. **B-R2** if B-R1 says routing alone doesn't close the gap.
5. **B-R3** only if B-R1+B-R2 evidence supports DP-as-vehicle for C1.

### Implementation status (2026-05-12 session)

- **B-R0'** (E91 `dp_full_polish`): driver + Docker DP build on lambda.ai
  done. ibm01 smoke at plateau 0.877 (within 3 % of cascade uncapped).
  ibm10 full run in progress at 3300 s budget. ibm14 + ibm12 + ibm17
  queued depending on ibm10 result.
- **B-R1** (E92 `dp_tilos_rudy`): manifest + `diff_rudy.py` prototype
  written. DP basic obj_fn doesn't include congestion at all — RUDY
  exists only for area-adjustment under `routability_opt_flag`. The
  "DP-RUDY vs TILOS-RUDY mismatch" framing is finer: stock DP doesn't
  optimize congestion. B-R1 must *add* a congestion term, not "swap"
  RUDY. Differentiable RUDY prototype scaffolded.
- **B-R2** (E93 `dp_topk_density`): manifest only. Scoped, not built.
- **B-R3** (E94 `dp_canonical_full`): manifest only. Heavier (7–10 d).
- **B-R4** (E91 `dp_cascade_perturb`): driver done. iter=100 result
  diverged. iter=10 retry running.

Cloud: lambda.ai A100 box `129.213.18.245`. DREAMPlace built CPU-only
inside `limbo018/dreamplace:cuda` Docker image (CUDA build failed on
compute_86 in nvcc 11.0; CPU build runs ibm10 in 12 s — not a
bottleneck). To resume work on cloud:

  cd ~/macro-place-challenge-2026
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    DP_USE_GPU=0 DP_DOCKER_IMAGE=dreamplace:custom \
    DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
    python3 experiments/E91_dp_full_polish/code/dp_full_polish.py <bench> <budget_s>

---

## Archived (do not work on; preserved for reference)

- `submissions/_archive/cd_lns_sa_hessian/` — E74 single-saddle champion
  (1.0666). Subsumed by cascade. Saddle primitives live in
  `experiments/E74_hessian_saddle/code/`, which is still load-bearing.
- `submissions/_archive/cd_lns_sa_hybrid/` — E48 per-bench best-of
  (1.08151). Prior champion.
- `submissions/_archive/cd_lns_gridbin/`, `cd_adaptive/`, `cd_only/`,
  `will_seed/` — older champion lineage.
- `submissions/_archive/cd_lns_sa_cascade_variants/` — 12 dead-end
  cascade tunings (placer_finegrain*, placer_widesaddle, placer_b3000,
  etc.) from the wall-safe tuning wave.
- `submissions/_archive/falsified/cd_lns_sa_hessian_dp/` — DREAMPlace
  lane (PATH B autopsy above).
- `experiments/_archive/` — 59 prior-generation experiments.
- `docs/archived/` — one-off handoff docs.

Anything in `_archive/` is **out of scope** until further notice. Don't
re-evaluate, don't try to revive without explicit reason. They're there
as historical record so we don't re-walk falsified paths.

---

## Hardware

- **Local M3 Max (16 cores)**: prototype / profile only. Per-core ~2×
  faster than partcl EPYC; results not predictive of submission
  performance.
- **Cloud OCI A100-SXM4-40GB** (`mpc-cloud` / 132.145.135.39, ssh alias):
  EPYC 9655P + 30 cores + A100 GPU + 216 GB. THIS is the partcl-class
  hardware. **All claims about "fits 60-min cap" or "beats X under cap"
  must be measured here**, never on M3.
- Required cloud env: `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
  MKL_NUM_THREADS=8` (otherwise numpy defaults to 1 thread → 9× slowdown;
  see `memory/cloud_openblas_gotcha.md`).
- A100 is the target hardware for C1 (differentiable canonical proxy).
  E87 GPU CD scaffold already builds on this.

## Verification budget

- Canonical proxy via `uv run evaluate <placer> --all --json` — this is
  what partcl uses; never trust hand-computed proxies.
- Wall budget claim must be measured under cloud `OPENBLAS=8` with
  `--jobs 1` (one bench at a time, full cores), not under parallel
  `--jobs N`.
- Local testing: M3 is fine for correctness checks (smoke + unit), not
  for wall-time claims.

---

## Cleanup landed 2026-05-12

First-submission polish pass:

1. Archived 12 dead cascade variants → `submissions/_archive/cd_lns_sa_cascade_variants/`
   (placer_finegrain*, placer_widesaddle, placer_morecascade, placer_b3000,
   placer_aggressive_kjoint, placer_dualbasin, placer_extended,
   placer_micrograin, placer_seed). Active folder is now just
   `placer.py` + `placer_adaptive.py`.
2. Archived `submissions/cd_lns_sa_hessian/` (E74 — subsumed by cascade)
   → `submissions/_archive/cd_lns_sa_hessian/`.
3. Archived `submissions/cd_lns_sa_hessian_dp/` (DP path falsified)
   → `submissions/_archive/falsified/cd_lns_sa_hessian_dp/`.
4. Replaced monkey-patching in `placer_adaptive.py` with a clean
   `contextmanager` (`_tune_cd_adaptive`) + module-level threshold
   constants. Same behavior, much more readable.
5. Refreshed top-level `submissions/README.md` and
   `submissions/cd_lns_sa_cascade/README.md` to clearly mark
   `placer_adaptive.py` as **THE submission entry** and document
   verified numbers + verification commands.
6. Simplified `run_cloud_walltight_all.sh` — removed dead dp/hessian
   variants, now takes the placer path as a positional arg defaulting
   to the cascade-adaptive entry.
7. Local smoke on ibm01 (180 s budget) → proxy 0.89195, zero overlaps,
   wall 250 s, property dispatch (IBM-class) fired correctly.

PATH A perf hunting deferred — the dominant wall cost is move/revert
inside `search_axis` (96 % of CD time per profile); micro-tweaks would
yield <1 %. The real speedup is A1, which is 1–2 days of engineering,
not a polish-pass quick win.
