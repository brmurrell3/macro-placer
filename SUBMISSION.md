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
(`Placer`), our **2026-05-20 upcoming submission**.

## Two submissions in the repo

### `submissions/thinkorplace/` — original 2026-05-13 submission

Cascade pipeline composing SDF + DPO inits, K-joint LNS, simulated
annealing, cascade saddle escape, portfolio saddle escape, and a
periphery-bias post-pass. ~55 min/bench.

Public leaderboard rank #9 at IBM 1.0771 (partcl-verified on AMD EPYC).

| Mode | IBM avg | NG45 avg | Overlaps |
|---|---:|---:|---:|
| --all (M3 2026-05-17) | 1.0575 | 0.6893 | 0 |

### `submissions/thinkorplace-v2/` — 2026-05-21 submission

Smooth-gradient placer combining three architectural changes vs v1's
cascade pipeline, **plus extended CD polish budget**:

1. **Per-net-trace congestion** (E111) — matches canonical PlacementCost
   within 15–25 % (vs the bbox-uniform ±200–260 %).
2. **Gaussian-smeared erf density** (E117) — replaces piecewise-linear
   `_grid_density`. C∞ smooth at cell boundaries → Adam descent finds a
   lower basin on dense benches (ibm12/14/17/18).
3. **FastDiffProxy backbone** (E115) — drops per-net pair_chunk loop +
   uses `index_select` for advanced indexing → 3–16× faster forward +
   backward on GPU.
4. **Extended CD polish budget** (2026-05-21) — `cd_polish_s` 600 → 900s,
   `budget_seconds` 720 → 1500s. CD on hard benches (ibm12/14/16/17) was
   budget-limited, not plateau-saturated. +0.7 % lift on EPYC.

Adam descent → greedy legalize → project_overlaps → CD polish. Device
selection prefers CUDA → MPS → CPU. ~25 min/bench at extended budget
(well under the 60 min/bench partcl cap).

| Mode | IBM avg | Overlaps | Qualified |
|---|---:|---:|---:|
| --all (AWS g5.2xlarge CUDA, 4-way) | 0.99115 | 0 | ✓ (prior config) |
| **--all (AWS g5.2xlarge CUDA, 2-way, extCD)** | **0.98387** | **0** | **✓** |

The 2-way result models the judges' 16-core EPYC: ≥4 vCPU per worker, no CD
contention. The 4-way result on 8-vCPU EPYC was CD-contention-limited.

**Projected public leaderboard rank #2-#3:**
- vs Carrotato #1 (0.967): +1.7 % gap
- vs Shoom #2 (0.978): **+0.6 % gap** (was +1.3 %)
- vs vmallela (1.011): **−2.7 % (we beat)**
- vs MultiDreamPlace (1.012): **−2.8 % (we beat)**

## Why v2 beats v1 by 8 %

Three compounding lifts:

| Change | Lift | Why |
|---|---:|---|
| Per-net-trace congestion (E111) | −5 % | Matches canonical objective on hard benches |
| Gaussian density (E117) | −2 % | Adam follows smooth gradient instead of jumping at cell boundaries |
| FastDiffProxy (E115) | −1 % | Fast `index_select` + dropped chunk loop → faster convergence, better basin on hard benches |

Per-bench lift highlights (final M3 vs v1):
  - ibm17 1.179 (vs prior 1.200 = −1.8 %)
  - ibm18 1.197 (vs prior 1.236 = −3.2 %)
  - ibm04 0.923 (vs prior 0.946 = −2.4 %)
  - ibm01 0.826 (vs prior 0.847 = −2.5 %)

See:
  - [`experiments/E111_per_net_trace_congestion/`](experiments/E111_per_net_trace_congestion/) — per-net trace congestion
  - [`experiments/E115_triton_kernels/`](experiments/E115_triton_kernels/) — FastDiffProxy backbone
  - [`experiments/E117_gaussian_density/`](experiments/E117_gaussian_density/) — Gaussian density
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
- `--memory 64g` — fits within container limit; peak usage ~4-6 GB
- `--cpus 16` — v2 uses ~6 cores during V3 descent + CD polish
- `--gpus all` — v2 doesn't use GPU; v1 doesn't either (DREAMPlace removed)
- `timeout 7200` — wall-time budget per full `--all` run

Per-bench placer budget for v2 is 12 minutes (`budget_seconds=720`);
17 IBM benches run sequentially in ~3-4 hours on a single 16-core EPYC.

## Defensive fallbacks (v2)

v2 includes a retry-and-fallback chain in case V3 descent ever produces
residual overlaps:
1. Retry with stronger overlap penalty (ovl_end 50 then 100)
2. Project_overlaps recovery step
3. Hard fallback: SDF init + project_overlaps + CD polish

This ensures the placer always returns a zero-overlap, qualifying
placement on any hardware.

## Switching to v1 (manual fallback)

If v2 underperforms on the judges' hardware:

```bash
# Edit placer.py to point at v1 instead of v2:
sed -i 's|submissions/thinkorplace-v2|submissions/thinkorplace|' placer.py
```

## What's in this repo

- `placer.py` — entry launcher (delegates to v2)
- `submissions/thinkorplace/` — v1 submitted 2026-05-13 (cascade pipeline)
- `submissions/thinkorplace-v2/` — v2 upcoming submission (V3 gradient placer)
- `submissions/common/` — shared base modules used by v1
- `submissions/_archive/` — prior champions + 26 experimental variants
- `experiments/E111_per_net_trace_congestion/` — v2's per-net-trace proxy
- `experiments/E110_smooth_global_placer/` — v2's smooth descent driver
- `macro_place/` — challenge evaluation framework
- `eval_docker/` — partcl's eval harness

## Contact

Brendan Murrell — brmurrell3 on GitHub.
