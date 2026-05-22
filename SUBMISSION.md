# thinkorplace submission

## TL;DR

```bash
git clone https://github.com/brmurrell3/macro-place-challenge-2026.git
cd macro-place-challenge-2026
./eval_docker/run_eval.sh thinkorplace placer.py
```

The first run will build the eval Docker image (PyTorch 2.5.1 + CUDA
12.4), then run our placer on all 17 IBM benchmarks. Results land in
`eval_docker/results/thinkorplace.log`.

## Entry point

`placer.py` at the repo root is a thin launcher. It locates the rest
of the repo on `sys.path` and dispatches to
[`submissions/thinkorplace-v2/placer.py`](submissions/thinkorplace-v2/placer.py)
(`Placer`), our **2026-05-22 submission**.

v2 is **self-contained**: every experiment module it imports lives in
`submissions/thinkorplace-v2/lib/`. Only `macro_place.*` (competition
infra) and standard deps (torch, numpy, scipy) are imported from
outside the submission directory.

## Two submissions in the repo

### `submissions/thinkorplace/` — original 2026-05-13 submission

Cascade pipeline composing SDF + DPO inits, K-joint LNS, simulated
annealing, cascade saddle escape, portfolio saddle escape, and a
periphery-bias post-pass. ~55 min/bench.

Public leaderboard rank #9 at IBM 1.0771 (partcl-verified on AMD EPYC).

| Mode | IBM avg | NG45 avg | Overlaps |
|---|---:|---:|---:|
| --all (M3 2026-05-17) | 1.0575 | 0.6893 | 0 |

### `submissions/thinkorplace-v2/` — 2026-05-22 submission

**3-lane parallel ensemble** built on the v2-extCD pipeline. Each bench
spawns three subprocesses via `multiprocessing.get_context("spawn")` so
each lane runs in an isolated process (no shared CUDA context, no
inter-lane interference). The master joins all three and returns the
canonical-best placement.

| Lane | Pipeline | Adds vs Lane A |
|---|---|---|
| **A — v2-extCD** | V4+Gaussian descent → greedy legalize → CD polish (~900s) | (baseline floor) |
| **B — +Hessian saddle** | A + bounded Hessian saddle escape (~120s) + trailing CD | Perturbs along softest eigenvector to escape CD plateau |
| **C — +Hungarian** | A + K=50 Hungarian joint permutation (~300s) + trailing CD | Re-shuffles K macros across slot grid; structurally different basin |

**Monotone safety property.** Lane A reproduces the prior v2-extCD
score; Lanes B/C cannot make it worse because the master picks min by
canonical proxy. Worst case = Lane A; expected case = strictly better.

#### Lane A core mechanics (inherited from v2-extCD)

1. **Per-net-trace congestion** (E111) — matches canonical PlacementCost
   within 15–25 % (vs the bbox-uniform ±200–260 %).
2. **Gaussian-smeared erf density** (E117) — replaces piecewise-linear
   `_grid_density`. C∞ smooth at cell boundaries.
3. **FastDiffProxy backbone** (E115) — drops per-net pair_chunk loop +
   uses `index_select` for advanced indexing → 3–16× faster forward +
   backward on GPU.
4. **Extended CD polish** — `cd_polish_s=900`, `budget_seconds=2700`
   per lane (was 600/720 in earlier config). CD on hard benches was
   budget-limited, not plateau-saturated.

#### Verified scores

| Mode | IBM `--all` | Overlaps | Hardware |
|---|---:|---:|---|
| Lane A floor (v2-extCD single pipeline) | **0.98387** | 0 | 2026-05-21 AWS EPYC c6a.4xlarge (2-way) |
| 3-lane oracle (per-bench MIN of A/B/C, offline) | **0.97745** | 0 | 2026-05-22 EPYC component runs |

The 3-lane number is the offline per-bench MIN of independently-run
Lane A / B / C across all 17 IBM. The on-box 3-lane ensemble run is in
flight (overnight); plateau-pick on the same trajectories should match
the oracle within noise.

**Projected public leaderboard rank #2:**
- vs Carrotato #1 (0.967): +1.1 % gap
- vs Shoom #2 (0.978): **−0.05 % (we beat at oracle)**
- vs vmallela (1.011): **−3.3 % (we beat)**

## Why the 3-lane wins

Each lane converges to a **structurally different** local minimum:
 - A reaches the CD local minimum from the V4+Gaussian descent basin.
 - B perturbs along the softest Hessian mode and re-polishes — escapes
   first-order CD plateaus that single-macro moves cannot.
 - C jointly re-permutes K=50 macros via Hungarian rectangular
   assignment on a slot grid — explores topology-level swaps that
   pair-swap LNS cannot reach.

Single-bench EPYC evidence: ibm17 (the hardest bench) went 1.18269
(Lane A) → 1.17408 (Lane C with Hungarian) = **−0.73 %** with zero
overlaps. M3 smokes on ibm04/09/10/17 all show Lane C lifts in the
−0.35 % to −1.25 % range.

#### v2 vs v1 (8 % lift inherited from Lane A's v2-extCD pipeline)

| Change | Lift | Why |
|---|---:|---|
| Per-net-trace congestion (E111) | −5 % | Matches canonical objective on hard benches |
| Gaussian density (E117) | −2 % | Adam follows smooth gradient instead of jumping at cell boundaries |
| FastDiffProxy (E115) | −1 % | Fast `index_select` + dropped chunk loop → faster convergence, better basin on hard benches |

3-lane ensemble adds an additional **−0.6 %** on top of v2-extCD via
the Hessian + Hungarian basin escape lanes.

See:
  - [`experiments/E111_per_net_trace_congestion/`](experiments/E111_per_net_trace_congestion/) — per-net trace congestion
  - [`experiments/E115_triton_kernels/`](experiments/E115_triton_kernels/) — FastDiffProxy backbone
  - [`experiments/E117_gaussian_density/`](experiments/E117_gaussian_density/) — Gaussian density
  - [`experiments/E138_bounded_saddle/`](experiments/E138_bounded_saddle/) — Lane B Hessian saddle escape
  - [`experiments/E159_hungarian_polish/`](experiments/E159_hungarian_polish/) — Lane C K=50 Hungarian permutation
  - [`experiments/E155_parallel_ensemble/`](experiments/E155_parallel_ensemble/) — multiprocess parallel ensemble framework
  - [`experiments/E127_v4_gaussian/`](experiments/E127_v4_gaussian/) — V4 + Gaussian composition (what v2 runs)

## Outside Docker (development / sanity check)

```bash
uv run evaluate placer.py --all --json
uv run evaluate placer.py --ng45 --json

# Or test the placers directly:
uv run evaluate submissions/thinkorplace-v2/placer.py --all --json
uv run evaluate submissions/thinkorplace/placer.py --all --json
```

## Resource expectations

Per the `eval_docker/run_eval.sh` limits:
- `--memory 64g` — fits within container limit; peak usage ~4-6 GB per lane × 3 lanes ≈ 15 GB
- `--cpus 16` — each of the 3 lanes is a spawn subprocess; OMP/MKL/OPENBLAS thread count auto-sized to `cpu_count // 3` per lane
- `--gpus all` — lanes auto-select CUDA → MPS → CPU; falls back gracefully if no GPU
- `timeout 7200` — wall-time budget per full `--all` run

Per-bench placer budget is 45 minutes (`budget_seconds=2700`); each
lane runs within that budget in its own subprocess. Total wall per
bench ≈ max(Lane A, Lane B, Lane C) + setup ≈ 30-45 min on EPYC.
17 IBM benches run sequentially in ~7-12 hours on a single 16-core EPYC.

## Defensive fallbacks (v2)

The 3-lane ensemble is robust by construction: any lane that fails or
produces overlaps is excluded from the canonical-best selection. If
**all three** lanes fail, the master falls back to SDF init +
`project_overlaps` recovery.

Within each lane (in `lane_worker.py`):
1. Descent retry with stronger overlap penalty (`ovl_end` 50 → 100)
2. `project_overlaps` recovery step before CD polish
3. Pickle write inside try/except so a failed lane still reports status

The picked placement is re-validated by `compute_overlap_metrics`; if
the picked lane has residual overlaps, the master falls through to the
next-best lane.

## Switching to v1 (manual fallback)

If v2 underperforms on the judges' hardware:

```bash
# Edit placer.py to point at v1 instead of v2:
sed -i 's|submissions/thinkorplace-v2|submissions/thinkorplace|' placer.py
```

## What's in this repo

- `placer.py` — entry launcher (delegates to v2)
- `submissions/thinkorplace/` — v1 submitted 2026-05-13 (cascade pipeline)
- `submissions/thinkorplace-v2/` — v2 submission (3-lane parallel ensemble)
  - `placer.py` — master that spawns 3 lanes and picks canonical-best
  - `lane_worker.py` — subprocess entry, dispatches A/B/C per `lane_idx`
  - `lib/` — self-contained experiment modules (11 files, no external `experiments/` imports)
- `submissions/common/` — shared base modules used only by v1
- `submissions/_archive/` — prior champions + experimental variants (incl. v2-extCD single-pipeline ancestor)
- `experiments/` — research history (manifests + code), not required at runtime
- `macro_place/` — challenge evaluation framework
- `eval_docker/` — partcl's eval harness

## Contact

Brendan Murrell — brmurrell3 on GitHub.
