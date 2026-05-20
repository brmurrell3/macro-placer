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

### `submissions/thinkorplace-v2/` — upcoming submission

Smooth-gradient placer using a **per-net-trace congestion model** that
matches the canonical PlacementCost objective much more closely than
the previous bbox-uniform approximation. Adam descent → greedy legalize
→ CD polish. ~12 min/bench.

| Mode | IBM avg | NG45 avg | Overlaps |
|---|---:|---:|---:|
| --all (M3 2026-05-19) | 1.00279 | 0.67861 | 0 |
| **--all (AWS EPYC g5.2xlarge 2026-05-20)** | **1.00835** | TBD | **0** |

EPYC variance: +0.55 % from M3 — much tighter than v1's cascade
pipeline ever showed.

**Projected public leaderboard rank #3** (between Shoom 0.978 and
vmallela 1.011).

## Why v2 beats v1 by 5 %

v1's bbox-uniform smooth congestion diverges 3–4× from the canonical
per-net trace on hard benches (ibm10/12/17). v2 replaces it with a
differentiable per-net-trace approximation that matches canonical
within 15–25 %. Adam descent on this corrected proxy finds basins on
hard benches that v1's cascade couldn't reach with 50 minutes of polish.

Per-bench lifts on hard benches (v2 vs v1):
  - ibm14: −9.1 %  ibm17: −9.4 %  ibm18: −7.9 %
  - ibm13: −8.3 %  ibm09: −5.7 %  ibm15: −4.4 %

See [`experiments/E111_per_net_trace_congestion/`](experiments/E111_per_net_trace_congestion/)
for the implementation.

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
