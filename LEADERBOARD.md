# Leaderboard — Macro Placement Challenge 2026

Auto-updated record of every placer run. **Hierarchy** (cheapest → costliest):

| Tier | Method | Bench Set | Wall (cloud --jobs 4) | Use For |
|-----:|--------|-----------|----------------------:|---------|
| 0 | Smoke single-bench (`-b ibm04`) | 1 | ~5-10 min | Crash + obvious regression check |
| 1 | 4-hardest-IBM probe (ibm10/12/14/17) | 4 | ~50 min | Catches most regressions, not NG45 |
| 2 | `--fast` (4-bench predictive subset) | 4 | ~50 min | Standard fast iteration |
| 3 | `--all` IBM | 17 | ~3-5 hr | Definitive IBM aggregate |
| 4 | `--ng45` | 4 | ~50 min | Tier 2 OpenROAD audit |
| 5 | `--all` + `--ng45` | 21 | ~4-6 hr | Final submission gate |

**Hardware bias:** results below are split by hardware. M3 ≠ EPYC.
M3 cores are ~2× faster per-thread; M3 results NOT predictive of partcl.

---

## SUBMISSION CANDIDATES

| Run | Hardware | Budget | IBM avg | NG45 avg | Max wall | Date | Verified by |
|-----|----------|-------:|--------:|---------:|---------:|------|-------------|
| **`placer_finegrain_adaptive.py` ← TARGET** | EPYC cloud | 3000s | **1.12189** | **0.6938** | 55.9min | 2026-05-12 02:14 | finegrain --all 17/17 + adaptive --ng45 4/4 |
| `placer_finegrain_safe.py` (b=2700, safer wall) | EPYC cloud | 2700s | in progress | (TBD via adaptive wrapper) | TBD | 2026-05-12 02:15 | --all running, ETA ~06:00 |
| `placer_finegrain.py` (IBM-only) | EPYC cloud | 3000s | 1.12189 | 0.6954 | 55.9min | 2026-05-12 | full --all done |
| `placer_adaptive.py` (prior) | EPYC cloud | 3000s | 1.137 | 0.6978 | 57min | 2026-05-11 | prior submission |
| `placer_b3000.py` (simpler) | EPYC cloud | 3000s | 1.137 | 0.7034 | 57min | 2026-05-11 | overnight |
| `placer_extended.py` (ceiling) | EPYC cloud | 5400s | partial 1.015 (8/17) | — | 92min ⚠️ | 2026-05-12 | OVERSHOOTS cap, ref only |

NG45 audit per-design (finegrain_adaptive.py, EPYC cloud, 2026-05-12 00:00):
| Design | Finegrain-adaptive | Prior adaptive | E48 ref | E74 ref |
|--------|-------------------:|---------------:|--------:|--------:|
| ariane133 | **0.6518** | 0.6638 | 0.6861 | 0.6641 |
| ariane136 | 0.6823 | 0.6848 | 0.6685 | 0.6518 |
| mempool_tile | 0.7374 | 0.7374 | 0.7375 | 0.7376 |
| nvdla | 0.7036 | 0.7053 | 0.6767 | 0.6716 |
| **avg** | **0.6938** | 0.6978 | 0.6922 | 0.6813 |

NG45 audit per-design (placer_adaptive.py, EPYC cloud, 2026-05-11 19:08-19:58):
| Design | Adaptive proxy | E48 ref | Δ vs E48 |
|--------|---------------:|--------:|---------:|
| ariane133 | **0.6638** | 0.6861 | **−3.25%** |
| ariane136 | 0.6848 | 0.6685 | +2.44% |
| mempool_tile | 0.7374 | 0.7375 | tied |
| nvdla | 0.7053 | 0.6767 | +4.23% |
| **avg** | **0.6978** | 0.6922 | +0.81% |

---

## ALL IBM `--all` RESULTS (canonical, descending proxy = worst → best)

| Avg proxy | Variant | Hardware | Budget | Max wall | Notes |
|----------:|---------|----------|-------:|---------:|-------|
| 2.1362 | SA baseline | — | — | — | weak baseline reference |
| 1.5338 | Will's seed (SA 3000) | M3 | ∞ | 50min | pre-fork |
| 1.4578 | RePlAce | — | — | — | published baseline |
| 1.1172 | public leaderboard target | — | — | — | beat this for Tier 1 |
| 1.15119 | cloud E74 b=2800 | EPYC | 2800s | 48min | too tight, no lift |
| 1.14322 | cloud E48 hybrid b=3000 | EPYC | 3000s | 56min | E25+E41 no saddle, worse than cascade |
| 1.14031 | cascade_j4 (--jobs 4 OPENBLAS=8) | EPYC | 3000s | 57min | +0.28% worse than j8 = contention tied |
| **1.13709** | **cloud cascade b=3000** ← SUBMISSION | EPYC | 3000s | 57min | wall-safe cascade variant |
| 1.12734 | cloud cascade b=3300 | EPYC | 3300s | 60.4min ⚠️ | ibm17 over 60-min cap |
| 1.08401 | local E74 b=3300 | M3 | 3300s | 56min | M3 fast cores, NOT EPYC-predictive |
| 1.08151 | E48 cached uncapped | M3 | ∞ | 130min | ADR-011 reference baseline |
| 1.0666 | E74 cached uncapped | M3 | ∞ | 96min | ADR-012 reference baseline |
| **1.0612** | cascade cached uncapped | M3 | ∞ | 60min | **the ceiling** (only achievable with multi-day cloud compute or A2/A3 CD speedup) |

---

## NG45 `--ng45` RESULTS (descending proxy = worst → best)

| Avg proxy | Variant | Hardware | Notes |
|----------:|---------|----------|-------|
| 0.7034 | cloud cascade b=3000 | EPYC | early adaptive exit on small NG45 |
| 0.6954 | cloud cascade b=3300 | EPYC | better than b=3000 |
| 0.6925 | cloud cascade NG45-tuned | EPYC | min_time_s=180, plateau_threshold=1e-4 |
| 0.6922 | E48 reference | M3 | reference baseline |
| 0.6848 | local cascade NG45 b=3300 | M3 | M3 cores; not EPYC-predictive |
| 0.6813 | E74 cached uncapped | M3 | ADR-012 reference |

### NG45 per-design (ariane133 / ariane136 / mempool_tile / nvdla)
| Variant | ariane133 | ariane136 | mempool_tile | nvdla |
|---------|----------:|----------:|-------------:|------:|
| E48 ref | 0.6861 | 0.6685 | 0.7375 | 0.6767 |
| E74 ref | 0.6641 | 0.6518 | 0.7376 | 0.6716 |
| Cloud cascade b=3300 | **0.6569** | 0.6872 | 0.7372 | 0.7004 |
| Cloud cascade b=3000 | 0.6898 | 0.6834 | 0.7372 | 0.7031 |
| Cloud cascade tuned | 0.6764 | **0.6563** | 0.7372 | 0.7000 |

---

## FALSIFIED variants (don't re-run)

| Variant | Result | Reason |
|---------|--------|--------|
| DREAMPlace direct basin | proxy 0.95-2.47 post-legalize | DP optimizes its own proxy, mismatch with PlacementCost |
| DREAMPlace + 60s CD polish | 21-372 residual overlaps remain | polish budget too short |
| DREAMPlace + 300s CD polish | 10-23% worse than cascade | DP basin structurally inferior |
| DP 50-config hyperparameter sweep | 5-17% worse than cascade on all 4 hardest | sweep can't bridge objective mismatch |
| cascade max_iters=10 | +0.67% worse | cascade phase budget saturates iters anyway |
| cascade_j4 (--jobs 4 OPENBLAS=8) | +0.28% worse | contention isn't the bottleneck |
| GPU smooth-proxy CD (E87) | hangs on CPU | O(N²) in _extract_net_data |
| E62 WillSeed init lane | --fast +1.44% | bad polished basin |
| E54 congestion-targeted destroy | --ng45 +1.45% | NG45-blind |
| E53 GPU DPO basin polish | 0/350 GPU restarts accepted | not additive on converged CD-LNS-SA |
| numpy-grids fast_evaluator patch | 1.04× CD speedup | bottleneck is per-net torch ops not grid ops |

---

## ACTIVE PROBES

| Probe | Hardware | Lanes | Expected outcome | ETA |
|-------|----------|------:|------------------|----:|
| cloud wide-saddle (k=1, 6 eps, 60s polish) on 4 hardest IBM | EPYC | 4 | ±1% vs cascade | ~21:00 |
| cloud more-cascade (40% cascade budget) on 4 hardest IBM | EPYC | 4 | ±1% vs cascade | ~21:00 |
| local wide-saddle ibm17 b=3300 | M3 | 1 | M3 reference | ~22:00 |

---

## How to read this file

- Numbers in **bold** = best/critical
- All EPYC results use cloud OCI A100 box (mpc-cloud); cloud env requires
  `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` or numpy
  defaults to 1 thread (9× slowdown).
- M3 results are reference only; partcl uses EPYC.
- "Max wall" = largest per-bench wall in the run; must stay under 60-min cap.
- Cascade cached 1.0612 is the IBM ceiling reachable on M3 uncapped.
  Path to reach it on EPYC under cap = PATH A2 (Cython port of move()).

## 5-VARIANT PROBE RESULTS (2026-05-11 21:00 PDT, EPYC cloud, ibm10/12/14/17)

vs cascade b=3000 baseline (avg 1.28178 on the 4 hardest):

| Variant | ibm10 | ibm12 | ibm14 | ibm17 | Wins | Avg |
|---------|------:|------:|------:|------:|-----:|----:|
| cascade b=3000 (baseline) | 1.0775 | 1.3031 | 1.2919 | 1.4546 | — | 1.28178 |
| wide-saddle (k=1, 6 eps, 60s polish) | 1.0989 | 1.3371 | 1.3051 | 1.4607 | 0/4 | 1.30045 |
| more-cascade (40% cascade budget) | 1.0904 | 1.3143 | 1.3051 | 1.4549 | 0/4 | 1.29117 |
| dual-basin (cascade from E25 AND E41) | 1.1020 | 1.3279 | 1.3083 | 1.4769 | 0/4 | 1.30377 |
| aggressive-kjoint (K=8) | 1.1038 | 1.3169 | 1.3068 | 1.4657 | 0/4 | 1.29830 |
| **finegrain** (tight eps + 300s polish) | **1.0781** | **1.2959** | 1.2927 | **1.4520** | **2/4** | **1.27968** |

Finegrain wins ibm12 (-0.6%) and ibm17 (-0.2%), ties on ibm10 and ibm14.
Avg -0.16% vs baseline — marginal but consistent. Launched finegrain --all
to test if pattern holds across all 17.

Other 4 variants STRICTLY WORSE than baseline cascade — falsified.

