---
id: E84
name: dreamplace_saddle
status: in_progress
parent: E74
created: 2026-05-06
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E84: DREAMPlace + Hessian saddle

## Hypothesis

DREAMPlace (limbo018/DREAMPlace, GPU analytical placer) finds **a different
basin** than our SDF / DPO inits.  Top leaderboard entries are
DREAMPlace-based (Cezar 1.037, vmallela 1.0109, MTK 1.2818).  Combining
DREAMPlace's basin with our Hessian saddle escape (E74's mechanism) might
beat E74's 1.0666 with massive wall headroom (~20-30 min/bench vs
~96 min for E74 ibm01).

## Status: IN PROGRESS — basin advantage confirmed; legalization is the bottleneck

The pipeline runs end-to-end with valid output (0 overlaps), but the
basin advantage gets destroyed during legalization.  Result: E84 v6 ibm01
**proxy 0.895 / wall 44.7 min** — worse than E83 v1's 0.8736 by +2.4 %.

## Method (current)

```
Phase 1   benchmark.pb.txt → benchmark_to_bookshelf.py → .aux/.nodes/.nets/.wts/.pl/.scl
Phase 2   DREAMPlace global_place + legalize  (Docker, GPU, ~30 sec on RTX 4070)
Phase 3   read .gp.pl → torch.Tensor [num_macros, 2] in microns
Phase 4   project_overlaps_extended (max_iter=500)
Phase 5   CD adaptive #1 — reaches the basin (~12 min for ibm01)
Phase 6   force_legalize — moves overlapping macros to nearest LEGAL grid bin
Phase 7   CD adaptive #2 — re-optimize within legal manifold (~12 min)
Phase 8   LNS gridbin polish (~3-5 min)
Phase 9   Hessian saddle escape (k=1, ε∈{0.3, 1.0, 3.0}, polish 180s; ~18 min)
```

## What works

| Component | Status |
|---|---|
| Docker setup, GPU passthrough on RTX 4070 | ✅ |
| `docker pull limbo018/dreamplace:cuda` (20.2 GB) | ✅ |
| Build DREAMPlace (after CMakeLists patch for CUDA 11.0 / arch 8.6) | ✅ |
| Snapshot image with deps installed: `macro-placer/dreamplace:built` | ✅ |
| `Benchmark` → Bookshelf converter (matches adaptec1's exact format) | ✅ |
| DREAMPlace runs on our converted ibm01 in ~30 sec | ✅ |
| **DREAMPlace's basin → CD reaches 0.85009 plateau (vs SDF's ~0.87)** | ✅ confirmed |
| Hessian saddle escape works on this plateau (eigvalue −0.59, lift −0.058) | ✅ |
| End-to-end pipeline produces 0-overlap output | ✅ (since v5) |

## What doesn't work

| Issue | Detail |
|---|---|
| `legalize_flag=1` doesn't fully legalize | DREAMPlace still leaves 60+ overlapping pairs on ibm01 |
| `project_overlaps` plateaus at ~39 pairs after 500 iters | Not enough to reach 0 overlaps |
| `force_legalize` destroys basin | Moves macros to grid-bin centers regardless of cost — proxy jumps from 0.85 → 1.10 (Δ=+0.25) |
| 2nd CD pass partially recovers | Gets to 0.92, not back to 0.85 — the discrete per-axis CD can't reverse the large displacements force_legalize made |

## Version progression

| Version | Pipeline | ibm01 result |
|---|---|---|
| v1 | DP → project → saddle (legalize=0) | ❌ 78 overlaps |
| v2 | DP + legalize=1 → project → saddle | ❌ 78 overlaps |
| v3 | + 1st CD pass | 35 ovl, **0.85 (illegal)** |
| v4 | + LNS legalize attempt | ❌ 23 ovl |
| v5 | + force_legalize (all bins) | ✅ 0 ovl, **0.916** |
| v6 | + 2nd CD pass | ✅ 0 ovl, **0.895**, 44.7 min |

## Why it's not winning yet

Two paradigms disagree on overlap handling:
* DREAMPlace ASSUMES overlaps are allowed during search → its output uses this freedom
* Our pipeline ASSUMES legality at every step → can't take a deeply-overlapping placement and "polish" it

`force_legalize` is too coarse a hammer.  CD's per-axis discrete moves can't
re-sort macros that need to "swap through" each other.  The basin (0.85)
exists in the overlapping manifold; CD can't reach it from a force-legalized
state.

## Next experiments to try (priority order)

### 1. **`target_density=0.5` in DREAMPlace** (cheapest, ~30 min)

Currently we use `target_density=1.0` — DREAMPlace allows full overlap.
Setting `0.5` makes density penalty stricter from the start, producing
fewer overlaps.  If output has ≤ 5 overlapping pairs, our existing
`project_overlaps` can finish them off.

Edit `experiments/E84_dreamplace_saddle/code/run_dreamplace.py:_make_dp_config`:
```python
"target_density": 0.5,
```

### 2. **`detailed_place_flag=1`** (cheapest, ~30 min)

We've had this set to 0 (we wanted raw output for our own saddle).  Their
detailed placer is row-aligned and produces a fully-legal `.dp.pl` output.
Untested for our IBM benchmarks but mature code.

Edit `experiments/E84_dreamplace_saddle/code/run_dreamplace.py:_make_dp_config`:
```python
"detailed_place_flag": 1,
```

Update `run_dreamplace.py` candidate ordering to prefer `*.dp.pl` first.

### 3. **Hungarian-matching legalizer** (best shot, 2-3 days)

Replace `force_legalize` with a global optimal assignment:
* Build cost matrix: 60 overlapping macros × M legal grid bins
* `cost(i, j) = displacement(i to bin j) + Δproxy_estimate(i moves to bin j)`
* Solve as assignment via `scipy.optimize.linear_sum_assignment`
* Resulting placement minimizes total cost change

E63 had a row-pack legalizer using Hungarian — can adapt that code.

### 4. **DREAMPlace + full E25/E41 pipeline** (~6 hr, half-bet)

Currently E84 only runs CD + LNS (no SA, no K-joint).  Use DREAMPlace
output as init for FULL E25 (CD + LNS + SA + K-joint).  K-joint can do
3-macro joint moves which might recover more basin than per-axis CD.

Replace SDF init in E25 with DREAMPlace's output.

## Resume checklist (next session)

1. Read this manifest + `docs/results.md` E83 / E84 sections
2. State of art: E83 v1 verified `--all` 1.0859; E84 v6 ibm01 0.895 (worse)
3. Submission deadline: **May 21**, ~14 days remaining
4. Decision pending: ship E83 v1 or push E84
5. Lowest-effort high-value next test: **option 1 (target_density=0.5)** — 30 min
6. To run: `cd /c/Users/fany8/PycharmProjects/macro-placer && .venv/Scripts/python.exe -m macro_place.evaluate experiments/E84_dreamplace_saddle/code/dreamplace_init_then_polish.py -b ibm01 --json --hypothesis E84_density_test`

## Pointers

* Source clone: `experiments/E84_dreamplace_saddle/DREAMPlace/`
* Built install: `experiments/E84_dreamplace_saddle/DREAMPlace/install/`
* Snapshot Docker image: `macro-placer/dreamplace:built` (local only, not pushed)
* CMake fix applied: `experiments/E84_dreamplace_saddle/DREAMPlace/CMakeLists.txt:132-138`
  (skip arch 8.6 when CUDA == 11.0)
* Code:
  * `code/benchmark_to_bookshelf.py` — `Benchmark` → Bookshelf
  * `code/run_dreamplace.py` — Docker wrapper
  * `code/dreamplace_saddle.py` — E84 v1-v2 (deprecated)
  * `code/dreamplace_init_then_polish.py` — E84 v3-v6 (current; v6 with 2nd CD)
* Logs: `logs/E84/{v3_smoke_ibm01.log, v4_smoke_ibm01.log, v5_smoke_ibm01.log, v6_smoke_ibm01.log}`
* Test runs: `experiments/E84_dreamplace_saddle/dp_runs/`
* Build logs: `experiments/E84_dreamplace_saddle/{build.log, build2.log, build3.log}`

## Submission impact

E84 doesn't currently improve over E83 v1 (1.0859) on ibm01.  The
submission strategy depends on whether one of the Next experiments
(target_density, detailed_place, Hungarian) lifts proxy below 0.86.
* If yes → ship E84 (proxy ≤ 1.06 expected on `--all`).
* If no → ship E83 v1 / v2 (1.0859 confirmed, cap-safe).

vmallela's verified leaderboard #1 at 1.0109 sets the bar.  Only
DREAMPlace + Hessian saddle has a shot at sub-1.05; E83 alone won't
match the top.
