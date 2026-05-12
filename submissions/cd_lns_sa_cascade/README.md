# cd_lns_sa_cascade — wall-safe E84 cascading saddle escape

E48 plateau pick (best of {E25, E41}) followed by **cascading** Hessian
saddle escape on the smooth proxy. Iterates: Lanczos smallest-algebraic
eigenvector → ±ε perturbation → CD-adaptive polish → repeat from the new
state until one of:

- the Hessian's smallest eigenvalue is non-negative (true local min reached)
- the cascade iteration produced no improvement
- `max_iters` exhausted
- wall budget exhausted

## Files

| File | Role |
|------|------|
| **`placer_adaptive.py`** | **Submission entry** — `CDLNSSACascadeAdaptivePlacer`, property-tunes CD parameters by canvas area (IBM-class vs NG45-class) |
| `placer.py` | Base — `CDLNSSACascadePlacer`, fixed CD parameters, parent of `placer_adaptive` |
| `README.md` | This file |

The 12 cascade tuning variants explored during the wall-safe wave
(`placer_finegrain*`, `placer_widesaddle`, `placer_morecascade`, etc.)
were archived to `submissions/_archive/cd_lns_sa_cascade_variants/` after
the canvas-area dispatch in `placer_adaptive.py` proved to dominate them.

## Verified numbers

Uncapped (M3 cached, via `experiments/E84_cascading_saddle/code/loader_placer.py`):

| Metric | Value | Reference |
|--------|------:|-----------|
| IBM avg `--all` | **1.0612** | M3, no wall cap |
| vs E48 (1.08151) | **−1.88 %** | prior champion |
| vs E74 (1.0666) | **−0.51 %** | prior champion |
| Overlaps | 0 on every bench | canonical eval |

Wall-safe (cloud EPYC, 60-min/bench cap, `placer_adaptive.py`):

| Metric | Value | Reference |
|--------|------:|-----------|
| IBM avg `--all` | **1.137** | cloud, `budget_seconds=3000` |
| NG45 avg `--ng45` | **0.6925** | cloud |
| Max wall | 57 min | safe under 60-min cap |
| vs RePlAce 1.4578 | **−22 %** | published baseline |
| vs leaderboard top ~1.01 | +13 % | gap to close (PATH A in top-level `TODO.md`) |

The cap-vs-uncapped delta (1.137 vs 1.0612) reflects unconverged CD —
96 % of CD time is single-threaded Python in `IncrementalProxyEvaluator.
move()/revert()`, so EPYC at ~2× slower per-core than M3 caps the polish
short of convergence. Closing the cap-vs-uncapped gap is PATH A in
`TODO.md` (revert-elimination + Cython/Numba port + batched candidate
eval).

## Pipeline

```
benchmark
  → Phase 1: E25 (SDF + CD + LNS + SA-v2)                ← budget-capped
  → Phase 2: E41 (DPO + CD + LNS + SA-v2 + K-joint)      ← budget-capped, skip if budget tight
  → Phase 3: cascading saddle on min(E25, E41) plateau   ← deadline-bound
  → final: best-of {E25, E41 if run, cascade}            ← overlap-validated
```

## Budget allocation

`placer_adaptive.py` defaults to `budget_seconds=3000.0` (50 min/bench).
`placer.py` defaults to `budget_seconds=3300.0` (55 min/bench) for offline
use. Internal split when `budget_seconds=B`:

- E25 (Phase 1): cd=B·0.20, lns=B·0.06, sa=B·0.06 ≈ 32 % budget
- E41 (Phase 2): cd=B·0.20, lns=B·0.05, sa=B·0.05, kjoint=B·0.05 ≈ 35 % budget
- Cascading saddle (Phase 3): remainder ≈ 33 % budget

Skip rules: E41 skipped if <300s left; cascade skipped if <60s left;
cascade stops cleanly at deadline — last good state returned and
overlap-validated.

Pass `budget_seconds=None` to either placer for offline mode (the
unbudgeted 1.0612 result on M3 used this path).

## Property-based dispatch (`placer_adaptive.py`)

Two CD-polish tunings dispatched on canvas area:

| Bench class | Canvas area threshold | `min_time_s` | `plateau_threshold` |
|-------------|----------------------:|-------------:|--------------------:|
| IBM (ICCAD04) | < 100 k μm² | 30 s | 1e-3 |
| NG45 (commercial) | > 100 k μm² | 180 s | 1e-4 |

Rationale: NG45 designs have ~6× the macro count and 1000× larger
canvas; CD plateaus more slowly. Tighter plateau detection prevents
premature exit on the larger benches.

Rule compliance: dispatching on `canvas_area` (a property of the loaded
benchmark) is permitted; dispatching on `benchmark.name` is not.

## Local smoke

```bash
# Single-bench, low budget, ~10 min on M3
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py -b ibm03
```

## Full evaluation (cloud)

```bash
# All 17 IBM, --jobs 4, ~5-6 hr wall on cloud EPYC
ssh mpc-cloud "bash ~/macro-place-challenge-2026/run_cloud_walltight_all.sh"

# Or specify alternative placer:
ssh mpc-cloud "bash ~/macro-place-challenge-2026/run_cloud_walltight_all.sh \
  submissions/cd_lns_sa_cascade/placer.py 4 cascade_fixed"
```

NG45 (4 designs, ~2.5 hr wall under `--jobs 4`):

```bash
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --ng45 --jobs 4 --json
```
