# thinkorplace submission

## TL;DR

```bash
git clone https://github.com/brmurrell3/macro-placer.git
cd macro-placer
./eval_docker/run_eval.sh thinkorplace placer.py
```

That's it. The first run will build the eval Docker image (PyTorch 2.5.1 + CUDA 12.4), then run our placer on all 17 IBM benchmarks. Results land in `eval_docker/results/thinkorplace.log`.

## Entry point

`placer.py` (at the repo root) is the launcher. It locates the rest of the
repo on `sys.path` and dispatches to
[`submissions/cd_lns_sa_cascade_dp_lane/placer.py`](submissions/cd_lns_sa_cascade_dp_lane/placer.py)
(`CDLNSSACascadeDPLanePlacer`), our 3-lane cascade pipeline:

  1. E25 lane: SDF init → CD → LNS → SA-v2 polish
  2. E41 lane: DPO init → CD → LNS → SA-v2 → K-joint
  3. DP lane (optional, see below): DREAMPlace GP → polish
  4. Plateau pick = best of {E25, E41, DP-polished}; cascade saddle escape on plateau

## Optional DREAMPlace 3rd lane

The placer **falls back gracefully** to a 2-lane configuration (Option A
equivalent) if DREAMPlace is unavailable, landing at IBM 1.0782 /
NG45 0.6810.

The bundled `submit_deps/dreamplace_install/` is a **CPU-only** build
(no CUDA kernels — built for portability). The placer defaults to
`gpu=1` and will log a "CANNOT enable GPU without CUDA compiled" error,
then return None — at which point the 2-lane fallback runs cleanly. So
the bundled install gives the **2-lane number** (1.0782), not the 3-lane
number (1.0665).

To get the 3-lane lift (IBM 1.0665 / NG45 0.6809, verified
2026-05-14 on lambda cloud A100), supply a CUDA-enabled DREAMPlace via
`$DREAMPLACE_ROOT`:

```bash
DREAMPLACE_ROOT=/path/to/cuda-dreamplace ./eval_docker/run_eval.sh thinkorplace placer.py
```

The placer auto-discovers DREAMPlace at:
  1. `$DREAMPLACE_ROOT` (env var, if set + has `dreamplace/Placer.py`)
  2. `/submission/dreamplace_install` (eval_docker extras mount)
  3. `submit_deps/dreamplace_install/` (relative to repo root — the
     bundled CPU-only fallback)

For a CUDA build, follow upstream DREAMPlace install instructions against
PyTorch 2.5.1 + CUDA 12.4 to match the eval_docker base image.

## Outside Docker (development / sanity check)

```bash
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --ng45 --json
```

Or use the launcher's resolution path:

```bash
uv run evaluate placer.py --all --json
```

## Resource expectations

Per the `eval_docker/run_eval.sh` limits:
- `--memory 64g` — fits within container limit; peak usage ~8-12 GB
- `--cpus 16` — placer uses up to ~16 cores during CD/LNS/SA phases
- `--gpus all` — used only if DREAMPlace lane is enabled
- `timeout 7200` — wall-time budget per full `--all` run

Per-bench placer budget is set to 55 minutes by default
(`budget_seconds=3300`); the 17 IBM benches run sequentially via the
default `--all` flow.

## Verified results

| Mode | IBM `--all` avg | NG45 `--ng45` avg | Overlaps | Date | Hardware |
|---|---:|---:|---:|---|---|
| 3-lane (with DREAMPlace) | **1.06650** | **0.68086** | 0 | 2026-05-14 | lambda cloud 129.213.89.145 (A100) |
| 2-lane fallback (no DREAMPlace) | 1.07820 | 0.68102 | 0 | 2026-05-16 | AWS EPYC c6a.4xlarge |

Composite avg across 21 benches (17 IBM + 4 NG45): **0.993** (3-lane) /
**0.998** (2-lane fallback). Both beat the public leaderboard reference
of 1.1172 by ≥3.5 % and RePlAce 1.4578 by ≥26 %.

## What's in this repo

- `placer.py` — entry launcher
- `submissions/cd_lns_sa_cascade_dp_lane/placer.py` — main 3-lane placer
- `submissions/cd_lns_sa_cascade/placer_adaptive.py` — 2-lane fallback path
- `submissions/cd_lns_sa/placer.py`, `submissions/cd_lns_sa_cascade/placer.py` — component lanes
- `experiments/` — research code that gets imported by the placer
- `macro_place/` — challenge evaluation framework (from upstream)
- `eval_docker/Dockerfile`, `eval_docker/run_eval.sh` — partcl's eval harness
- `submit_deps/` — optional binary mounts (DREAMPlace install goes here)

## Contact

Brendan Murrell — brmurrell3 on GitHub.
