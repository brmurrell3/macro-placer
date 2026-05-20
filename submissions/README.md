# Submissions

Two clearly-labeled submissions:

## `thinkorplace/` — the 2026-05-13 submitted placer

The placer originally submitted on **May 13** to the partcl leaderboard.
This is `CDLNSSACascadeStackedPeripheryPlacer` (a.k.a. "Option C"): a
cascade pipeline composing SDF + DPO inits, K-joint LNS, simulated
annealing, cascade saddle escape (canonical eigvec), portfolio saddle
escape (3 non-canonical Hessian weights), and a periphery-bias
post-pass.

Run-time: ~55 minutes per benchmark (close to the 60-minute hard cap).

| Mode | IBM avg | NG45 avg | Overlaps |
|---|---:|---:|---:|
| --all (17 IBM, M3-verified 2026-05-17) | **1.0575** | **0.6893** | 0 |

Public leaderboard rank with this submission (as of 2026-05-19): **#9**
at IBM 1.0771 (verified by partcl on AMD EPYC + RTX 6000 Ada).

## `thinkorplace-v2/` — the upcoming submission

The new placer built 2026-05-19. Drops the entire cascade pipeline and
replaces it with a smooth-gradient placer using a per-net-trace
congestion model that matches the canonical PlacementCost objective much
more closely than the previous bbox-uniform approximation. Adam descent
on the smooth proxy → `greedy_macro_legalize` → CD polish.

Run-time: ~12 minutes per benchmark (5× faster than v1).

| Mode | IBM avg | NG45 avg | Overlaps |
|---|---:|---:|---:|
| --all M3 (champion 2026-05-19) | 1.00279 | 0.67861 | 0 |
| **--all AWS EPYC (2026-05-20)** | **1.00835** | TBD | **0** |

EPYC cross-validation showed only **+0.55%** variance from M3 — much
tighter than the v1 cascade pipeline.

Projected public leaderboard rank with this submission: **#3** (between
Shoom 0.978 and vmallela 1.011).

## What changed v1 → v2

The bbox-uniform smooth congestion in v1 diverged 3–4× from the
canonical per-net trace on hard benches. v2 replaces it with a
differentiable per-net trace approximation that matches canonical
within 15–25%. This single change lets Adam find basins on hard benches
(ibm14 −9.1%, ibm17 −9.4%, ibm18 −7.9%) that v1's cascade couldn't
reach with 50 minutes of polish.

See [`experiments/E111_per_net_trace_congestion/`](../experiments/E111_per_net_trace_congestion/)
for the proxy implementation and
[`experiments/E110_smooth_global_placer/notes/`](../experiments/E110_smooth_global_placer/notes/)
for the iteration log.

## Direct runs (development)

```bash
# v1
uv run evaluate submissions/thinkorplace/placer.py --all --json

# v2
uv run evaluate submissions/thinkorplace-v2/placer.py --all --json
```

## Docker submission

The repo-root `../placer.py` is a thin launcher that delegates to the
**v2** placer; it handles the `eval_docker` mount/path setup.

```bash
./eval_docker/run_eval.sh thinkorplace placer.py
```

## Layout

```
submissions/
├── README.md           ← this file
├── _archive/           ← prior champions + 26 experimental variants
├── common/             ← shared base modules (used by thinkorplace v1)
├── examples/           ← reference placers (greedy, random)
├── thinkorplace/       ← v1 submission (cascade pipeline, ~55 min/bench)
└── thinkorplace-v2/    ← v2 upcoming submission (gradient + CD, ~12 min/bench)
```
