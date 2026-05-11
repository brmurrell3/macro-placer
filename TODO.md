# TODO — two-path work plan

**Deadline:** May 21, 2026 (~10 days)

## Current submission floor

`submissions/cd_lns_sa_cascade/placer_adaptive.py`
- IBM avg **1.137** (cloud cascade b=3000, max wall 57 min, safe under 60-min cap)
- NG45 avg **0.6925** (NG45 lane uses tuned min_time_s=180s)
- Beats RePlAce 1.4578 by **−22 %**
- Loses to leaderboard top (~1.01) by **+13 %** — closing this gap is the goal

---

## PATH A — speed up our champion CD pipeline ★ HIGHEST EV

**Premise:** Cascading saddle escape verified canonical **1.0612 on 17 IBM
uncapped** (`experiments/E84_cascading_saddle/results/`). Wall-safe variant
plateaus at 1.137 because **96% of CD time is single-threaded Python**
(`IncrementalProxyEvaluator.move()` + `revert()`, confirmed by cProfile on
ibm04: 32s of 33s). 10-30× CD speedup unlocks the cached-quality basin
under the 60-min cap.

### A1. Eliminate `revert()` via pure delta function *(1-2 days)*
- `search_axis` currently: `move() → cost → revert()` per candidate (96 % of CD)
- Replace with: `batch_delta_cost(macro_idx, K candidates, current_state) → [K] proxy deltas`
- Commit only the argmin candidate; no state mutation during probing
- Expected speedup: ~10× on CD inner loop

### A2. Cython/Numba port of `move()` *(2-3 days)*
- Current: ~100 lines of Python dict updates over `affected_nets`,
  `macro_cong_contrib`, `H_net_cong` etc.
- Port to native; explicit memory layout for the dict-heavy paths
- Expected speedup: 20-100× on move/revert
- Composes with A1 (post-revert-elimination)

### A3. Batch candidate eval as tensor op *(2-4 days, optional GPU port)*
- For macro M, axis A, K candidates → single `[K]` tensor op
- Reuses `experiments/E87_gpu_cd/` scaffold but targets the **exact** proxy delta
  (not the smooth proxy approximation E87 currently uses)
- Blocker to fix first: `_extract_net_data` has an O(N²) hot spot — must profile + rewrite before GPU port helps

### A4. Validate cascade --all under accelerated pipeline *(1 day)*
- Re-run cloud cascade b=3000 with accelerated CD
- Target: IBM aggregate ≤ 1.08 (matches uncapped 1.0612 ± wall-cap slop)
- NG45 aggregate ≤ 0.68

### A5. Final submission packaging + ORFS Tier 2 scoping *(2-3 days)*
- Form fill, evidence collation, freeze final placer
- Test ORFS flow locally on one NG45 design to validate Tier 2 readiness

**A active code:**
- `submissions/cd_lns_sa_cascade/{placer.py, placer_b3000.py, placer_adaptive.py, README.md}`
- `submissions/cd_lns_sa_hessian/placer.py` (E74 single saddle, imported by cascade)
- `submissions/cd_lns_sa/placer.py` (E25 component)
- `experiments/E41_dpo_kjoint/code/` (E41 component)
- `experiments/E74_hessian_saddle/code/` (saddle primitives)
- `experiments/E84_cascading_saddle/code/` (cascade logic) + `results/` (cached .pt files)
- `experiments/E87_gpu_cd/code/` (in-progress GPU CD scaffold — A1-A3 work happens here)
- `macro_place/incremental_evaluator.py`, `macro_place/cd_core.py` (hot paths to optimize)

---

## PATH B — DREAMPlace exploration *(KILLED 2026-05-11 PM)*

**Verdict: FALSIFIED.** Sweep ran 13-25 configs/bench across ibm10/12/14/17
on cloud A100. Best DP basin proxy per bench (after greedy_legalize):

| Bench | Best DP basin | cascade b=3000 | Gap |
|-------|--------------:|---------------:|----:|
| ibm10 | 1.2553 | 1.0775 | **+16.5%** |
| ibm12 | 1.3712 | 1.3031 | +5.2% |
| ibm14 | 1.4176 | 1.2919 | +9.7% |
| ibm17 | 1.5388 | 1.4546 | +5.8% |

**0/4 hardest benches** have a DP basin within even 5% of cascade after K=25.
The 5D hyperparameter sweep can't bridge the structural objective mismatch
between DREAMPlace's loss (HPWL + Gaussian density + RUDY-as-DP-formulates)
and the canonical PlacementCost proxy (top-K density + TILOS-smoothed RUDY
with TILOS weights).

**Code preserved** at `experiments/E76_dreamplace_integration/` and
`submissions/cd_lns_sa_hessian_dp/` for reference. The greedy macro
legalizer (`macro_legalizer.py`) is a reusable utility regardless.

**Resources freed → all-in on PATH A.**

---

## Archived (do not work on; preserved for reference)

- `submissions/_archive/` — old champions (cd_lns_gridbin E12, cd_adaptive E9, cd_only, will_seed), E48 hybrid (cd_lns_sa_hybrid), falsified wall-safe wrappers (placer_b2800, placer_b3000_e48, placer_maxiter10, placer_ng45)
- `experiments/_archive/` — 59 prior-generation experiments (E1-E73 minus the active ones, plus E75-E83, E85, E86)
- `docs/archived/` — one-off handoff docs (MORNING_REPORT_2026-05-11.md)

Anything in `_archive/` is **out of scope** until further notice. Don't
re-evaluate, don't try to revive without explicit reason. They're there as
historical record so we don't re-walk falsified paths.

---

## Hardware

- **Local M3 Max (16 cores)**: prototype/profile only. Per-core ~2× faster
  than partcl EPYC; results not predictive of submission performance.
- **Cloud OCI A100-SXM4-40GB** (`mpc-cloud` / 132.145.135.39, ssh alias):
  EPYC 9655P + 30 cores + A100 GPU + 216 GB. THIS is the partcl-class
  hardware. **All claims about "fits 60-min cap" or "beats X under cap"
  must be measured here**, never on M3.
- Required cloud env: `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`
  (otherwise numpy defaults to 1 thread → 9× slowdown; see
  `memory/cloud_openblas_gotcha.md`).
- DREAMPlace on cloud: `DREAMPLACE_ROOT=/opt/DREAMPlace/install` +
  `DREAMPLACE_PYTHON=/usr/bin/python3` (system Python has the shapely etc.
  runtime deps; uv's Python doesn't).

## Verification budget

- Canonical proxy via `uv run evaluate <placer> --all --json` — this is
  what partcl uses; never trust hand-computed proxies.
- Wall budget claim must be measured under cloud OPENBLAS=8 with
  `--jobs 1` (one bench at a time, full cores), not under parallel `--jobs N`.
