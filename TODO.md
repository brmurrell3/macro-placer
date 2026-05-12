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

## PATH B — DREAMPlace exploration  *(KILLED 2026-05-11 PM)*

**Verdict: FALSIFIED.** Sweep ran 13–25 configs/bench across
ibm10/12/14/17 on cloud A100. Best DP basin proxy per bench (after
greedy_legalize):

| Bench | Best DP basin | cascade b=3000 | Gap |
|-------|--------------:|---------------:|----:|
| ibm10 | 1.2553 | 1.0775 | **+16.5 %** |
| ibm12 | 1.3712 | 1.3031 | +5.2 % |
| ibm14 | 1.4176 | 1.2919 | +9.7 % |
| ibm17 | 1.5388 | 1.4546 | +5.8 % |

**0/4 hardest benches** have a DP basin within even 5 % of cascade after
K=25. The 5D hyperparameter sweep can't bridge the structural objective
mismatch between DREAMPlace's loss (HPWL + Gaussian density +
RUDY-as-DP-formulates) and the canonical PlacementCost proxy (top-K
density + TILOS-smoothed RUDY with TILOS weights).

**Code preserved** at `experiments/E76_dreamplace_integration/` and
`submissions/_archive/falsified/cd_lns_sa_hessian_dp/`. The greedy macro
legalizer (`macro_legalizer.py`) is a reusable utility regardless — C1's
projection step will use it.

**Resources freed → all-in on PATH A + PATH C.**

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
