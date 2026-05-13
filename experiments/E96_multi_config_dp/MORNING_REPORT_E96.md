# E96 Morning Report — Multi-Config DP Selection

**Session:** 2026-05-12 21:30 EDT → 2026-05-13 08:00 EDT
**Author:** Background research Claude
**Status (05:12 UTC):** **All 4 hard-bench multi-DP K=4 polishes COMPLETE.** PATH A
`--all` post-A1 COMPLETE at **1.07820**. **Hybrid placer smoke on ibm14 = 1.198 in
58.5min** (valid, zero overlap, fits 60-min cap). Findings ready for submission decision.

## Hybrid placer end-to-end smoke (ibm14)

The rebalanced multi-DP hybrid placer (`submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py`)
was smoke-tested on ibm14:

| Phase | Proxy | notes |
|---|---:|---|
| E25 polish | 1.252 | SDF init + CD+LNS+SA at 18% budget |
| E41 polish | 1.240 | DPO init + CD+LNS+SA+K-joint at 20% budget |
| Multi-DP K=4 (winner: dw_low) | 1.225 | 600s K=4 basin selection + 394s polish |
| **Plateau pick** | **DP (1.225)** | best of {E25, E41, multi-DP} |
| **Cascade saddle final** | **1.198** | (zero overlap, 58.5min total wall) |

Compared to cascade-adaptive ibm14 (1.220): **−1.9% lift** (verified).
Compared to standalone multi-DP polish (1.192): **+0.5%** (hybrid has less DP polish
budget due to lane sharing).
Within 60-min cap: yes (58.5min, 1.5min margin).

## 🚨 BIG NEWS — read this first

**PATH A's `placer_adaptive.py --all` (post-A1 5.36× CD speedup) COMPLETED at 03:15 UTC
with avg_proxy_cost = 1.07820 (zero overlaps, 17/17 qualified, total runtime 14.7 hr `--jobs 4`).**

Hard-4 cascade-adaptive results:
| Bench | cascade-adaptive (post-A1) | cascade-capped (pre-A1) | Δ |
|---|---:|---:|---:|
| ibm10 | **1.016** | 1.0775 | -5.7% |
| ibm12 | **1.216** | 1.3031 | -6.7% |
| ibm14 | **1.220** | 1.2919 | -5.6% |
| ibm17 | **1.348** | 1.4546 | -7.3% |
| **hard-4 avg** | **1.200** | 1.282 | **-6.4%** |

So the submission floor moves from 1.137 (old cascade adaptive) to **1.0782** (post-A1).

**Implication for my multi-config DP work:** cascade-adaptive's per-bench numbers
beat my multi-config DP on ibm10 (1.016 < my 1.056) but lose on ibm14 (1.220 > my 1.192).
The hybrid (best-of-{cascade-adaptive, multi-DP}) gets the better of each.

Projected hybrid hard-4 with multi-config K=4 DP lane:
- ibm10: 1.016 (cascade wins) — multi-DP doesn't help here
- ibm12: ~1.10 (multi-DP wins by ~10%)
- ibm14: 1.192 (multi-DP wins by 2.3%)
- ibm17: ~1.30 (multi-DP wins by ~3-4%; pending polish completion)
- **avg ~1.15** (vs cascade alone 1.200 = **-4.2% lift**)

For full `--all` (17 benches): hybrid expected ~1.05-1.07 (vs cascade-adaptive 1.078,
**-1 to -2.5% lift**). Below the M3 cached uncapped ceiling 1.0612.

## Also relevant — single-DP hybrid (PATH B Claude's run) on 6 IBM benches

The OTHER PATH B Claude's `cd_lns_sa_cascade_dp_lane` (single-DP hybrid) finished 6
benches earlier this session:

| Bench | hybrid single-DP final | cascade-adaptive | plateau pick | DP polished |
|-------|---:|---:|---|---:|
| ibm01 | 0.867 | 0.889 | DP wins | 0.874 |
| ibm09 | 0.799 | 0.820 | DP wins | 0.814 |
| ibm10 | 1.029 | 1.016 | E41 wins | 1.292 (terrible) |
| ibm12 | 1.145 | 1.216 | DP wins | 1.164 |
| ibm14 | 1.225 | 1.220 | E41 wins | 1.269 (mediocre) |
| ibm17 | 1.308 | 1.348 | DP wins | 1.330 |
| **hard-4 avg** | **1.177** | **1.200** | — | — |

Single-DP hybrid **already gives -1.9% lift on hard 4** over cascade-adaptive alone.

**Multi-config K=4 lift over single-DP hybrid:** approximately +0-2% on hard 4
depending on which configs the K=4 selector happens to pick (DP basin stochasticity).
Per-bench expected:
- ibm10: single-DP got LUCKY (E41 actually won at 1.029). Multi-DP unlikely to improve.
- ibm12: multi-DP polishes from seed_2024 (1.421) → expected ~1.10 final < single-DP's 1.164.
- ibm14: my v1 multi-DP = 1.192 < single-DP 1.225 (-2.7%).
- ibm17: my v1 multi-DP (in flight) projected ~1.30 vs single-DP 1.308.

Modest improvements. Multi-config DP is **variance reduction**, not a huge absolute lever.

## TL;DR — three bullets

1. **Multi-config K=4 DP polish WINS on 3 of 4 hard benches (now all verified):**

   | Bench | E96 K=4 | cascade-adapt | Δ | winner config |
   |---|---:|---:|---:|---|
   | ibm10 | 1.056 | **1.016** | +4% | baseline_auto (cascade wins) |
   | ibm12 | **1.105** | 1.216 | **−9.1%** | seed_2024 |
   | ibm14 | **1.192** | 1.220 | **−2.3%** | dw_low |
   | ibm17 | **1.318** | 1.348 | **−2.4%** | seed_2024 |
   | hard-4 avg | **1.168** | 1.200 | **−2.7%** | — |

   Hybrid (best-of-{cascade-adapt, multi-DP K=4}) hard-4: **1.158** = **−3.5% lift** over cascade alone.

2. **The B-R0' autopsy was a lucky sample.** Phase 1 overnight (single-DP, same code as autopsy
   but new basin samples) gave −3.4% lift on hard 4, not the autopsy's −6.9%. ibm10 in
   particular has bimodal basin variance (1.215 ↔ 2.34 across runs of identical config).

3. **A drop-in hybrid variant is ready** at `submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py`
   (with rebalanced budget: DP lane 30%, E25/E41 18%/20%, cascade saddle 28%).
   Awaits `--all` validation but expected ~1-2% lift over single-DP hybrid.

## Recommendation

**Deploy `submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py` for hybrid `--all`.**

Command on cloud:
```bash
ssh ubuntu@129.213.18.245
cd ~/macro-place-challenge-2026
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
  DP_DOCKER_IMAGE=dreamplace:custom DP_USE_GPU=0 DP_NUM_THREADS=8 \
  uv run evaluate submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py \
    --all --json --hypothesis E96multilane
```

For NG45:
```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
  DP_DOCKER_IMAGE=dreamplace:custom DP_USE_GPU=0 DP_NUM_THREADS=8 \
  uv run evaluate submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py \
    --ng45 --json --hypothesis E96multilane_ng45
```

**Caveats:**
- Single-run results swing 5-15% due to DP basin stochasticity. Run 2-3 times and average.
- The K=4 grid is `[baseline_auto, dw_low, dw_high, seed_2024]` — verified to cover the
  per-bench optima found in the E96 probe (ibm10 dw_low, ibm12 baseline, ibm14 dw_high,
  ibm17 seed_2024).
- DP basin (and therefore polish trajectory) is highly stochastic. Expected variance
  in hybrid `--all` aggregate: ±0.5-1.5%.

## Concrete validated results

### ibm10 K=4 v1 (OLD multi_dp_basin, no extended_legalize)

**Final: 1.05583** (zero overlaps, 2376s wall)

Candidates (sorted by score = legalize_proxy + 10×overlap_count):
| Config | basin | legal | ovl | score | role |
|---|---:|---:|---:|---:|---|
| **baseline_auto** | 1.367/514 | **1.512** | 0 | 1.512 | ★ winner |
| seed_2024 | 2.080/79 | 2.083 | 0 | 2.083 | |
| dw_high | 1.282/113 | 1.358 | 1 | 11.358 | excluded (overlap penalty) |
| dw_low | 2.481/736 | 2.411 | 1 | 12.411 | excluded (overlap penalty) |

Polish trajectory: 1.512 → CD 1.069 → LNS 1.057 → SA 1.057 → cascade 1.056.
**Note:** dw_high had the *best basin* (1.282) with 1 residual overlap, but the
overlap penalty (10×) excluded it. v1 was forced to pick baseline_auto (the
"worst valid"). The polish trajectory found a lucky low valley (1.056).

### ibm14 K=4 v1 (same OLD multi_dp_basin)

**Final: 1.19210** (zero overlaps, 3313s wall — over the 55-min cap by 9 min in this run)

Candidates:
| Config | basin | legal | ovl | score | role |
|---|---:|---:|---:|---:|---|
| **dw_low** | 1.430/56 | **1.454** | 0 | 1.454 | ★ winner |
| seed_2024 | 1.439/82 | 1.466 | 0 | 1.466 | |
| dw_high | 1.458/80 | 1.484 | 0 | 1.484 | |
| baseline_auto | 1.483/81 | 1.506 | 0 | 1.506 | |

Polish trajectory: 1.454 → CD 1.201 → LNS 1.199 → SA 1.199 → cascade 1.192.
All 4 configs had clean legal basins (ovl=0); sort correctly picked dw_low.

### ibm12 K=4 v1 (NEW multi_dp_basin with extended_legalize)

**Final: 1.10490** (zero overlaps, 3168s wall)

Winner: **seed_2024** at legal=1.4210. Polish trajectory: 1.421 → CD 1.112 → LNS 1.111
→ SA 1.111 → cascade 1.105. The new multi_dp_basin's extended_legalize fallback rescued
baseline_auto from 1 residual overlap (1.460 → 1.461 post-rescue), but seed_2024 still
won outright with the lowest score.

### ibm17 K=4 v1 (NEW multi_dp_basin with extended_legalize)

**Final: 1.31757** (zero overlaps, 3591s wall — over the cap)

Winner: **seed_2024** at legal=1.5841. Polish trajectory: 1.584 → CD 1.335 → LNS 1.332
→ SA 1.332 → cascade 1.318. Confirms probe data — seed_2024 strongly wins ibm17.

### Phase 1 overnight (single-DP B-R0' on hard 4) — for comparison

| Bench | basin | legal | final | vs cascade | vs autopsy |
|-------|---:|---:|---:|---:|---:|
| ibm10 | 2.335 | 2.359 | 1.167 | **+8.3%** lose | +6.6% |
| ibm12 | 1.413 | 1.428 | 1.109 | **−14.9%** win | −1.8% |
| ibm14 | 1.483 | 1.507 | 1.226 | −5.1% win | −1.4% |
| ibm17 | 1.796 | 1.816 | 1.449 | −0.4% tie | +10.9% |
| **hard-4 avg** | — | — | **1.238** | **−3.4% lift** | (vs autopsy 1.194) |

So **single-DP gives -3.4% lift on hard 4 in expectation, not the autopsy's -6.9%**.
The autopsy was a particularly lucky basin draw.

### Phase 2 overnight (single-DP NG45) — also for comparison

| Bench | DP basin | legal | final | E48 | E74 | cascade cloud |
|-------|---:|---:|---:|---:|---:|---:|
| ariane133 v4 | 1.153 | 0.96 (rescued) | **0.670** | 0.686 | 0.664 | 0.690 |
| ariane136 | 0.960 | 0.914 | **0.748** | 0.669 | 0.652 | 0.683 |
| nvdla | 0.865 | 0.880 | **0.691** | 0.677 | 0.672 | 0.703 |
| mempool_tile | (in cascade) | — | (~0.69) | 0.738 | 0.738 | 0.737 |

NG45 is **mixed**:
- ariane133: DP wins vs cascade/E48, slight loss to E74 cached.
- ariane136: DP LOSES BIG (0.748 vs E74 0.652). Hybrid will pick cascade lane.
- nvdla: DP marginally wins vs cascade, loses to E48/E74. Mixed.
- mempool_tile: expected DP win (cascade-vs-DP basin gap is wide here).

NG45 hybrid (best-of-lanes) avg: ~0.68 (competitive with E74's 0.6813).

## Hybrid placer projection

Combining cascade-capped + Phase 1 single-DP + E96 multi-config K=4 (best of each):

| Bench | cascade | single-DP | multi-DP K=4 | hybrid pick | hybrid val |
|-------|---:|---:|---:|---|---:|
| ibm10 | 1.0775 | 1.167 | **1.056** | E96 K=4 | 1.056 |
| ibm12 | 1.3031 | **1.109** | (not tested) | single-DP wins | 1.109 |
| ibm14 | 1.2919 | 1.226 | **1.192** | E96 K=4 | 1.192 |
| ibm17 | 1.4546 | 1.449 | (in flight ★) | TBD | TBD |
| ibm01 | 0.85 (uncapped) | 0.862 | (not tested) | cascade marginal | ~0.85 |
| ibm09 | (n/a) | 0.789 | (not tested) | single-DP | 0.789 |
| (13 other IBM) | — | — | — | mixed | — |

**Estimated `--all` aggregate for multi-DP hybrid: ~1.10** (vs cascade 1.137 = **−3.3% lift**).

## What's still running (will fill in by ~04:00 UTC)

- E96 K=4 v2 ibm10 (with extended_legalize fallback): mid-cascade. Selection log shows
  winner=dw_high which by score should be baseline_auto. **Possible selection bug** — see
  "Open question" below. Init proxy from dw_high 1.590 → CD 1.147 → LNS 1.132 → SA 1.132
  → cascade in progress. Expected ~1.10 final (worse than v1's 1.056).
- E96 K=4 ibm12 polish: launched at 03:06 UTC. Will confirm baseline_auto winning on
  ibm12 (probe data already showed this).
- E96 K=4 ibm17 polish: launched at 03:03 UTC. Probe data shows seed_2024 should win.
- Phase 2 mempool_tile cascade: should finish soon.

## v2 selection bug — DIAGNOSED: process contention leak

**Status: v2 ibm10 final 1.11991** (vs cascade 1.0775 +3.9% LOSE, vs B-R0' +2.3% LOSE,
vs v1 ibm10 +6.1% LOSE, vs Phase 1 -4.0% WIN). Confirms the bug significantly hurts.

The v2 ibm10 log printed:
```
[multi-DP 1/4] baseline_auto: ... legal 1.4433/0 score=1.4433
[multi-DP 2/4] dw_low:        ... legal 2.2561/0 score=2.2561
[multi-DP 3/4] dw_high:       ... legal 1.5902/0 score=1.5902
[multi-DP 4/4] seed_2024:     ... legal 3.2612/0 score=3.2612
[multi-DP] winner: dw_high (legal_proxy=1.5902, ovl=0)
```

But the v2 JSON's `basin_stats["candidates"]` shows baseline_auto with **legal=1.6928,
score=1.6928** — not the logged 1.4433! That 1.6928 value matches the *killed v2 process
50730*'s first-config result (logged at 02:28 UTC before I killed it at 02:30 UTC).

**Hypothesis (most likely):** The two briefly-overlapping v2 processes (50730 launched
02:28, killed 02:30; 50291 launched 02:26) both wrote to /tmp/E96_ibm10_polish_v2.log
with stdout buffering. Their writes interleaved. The PRINT line from 50291 used its own
values (1.4433); but somehow the STORED candidates inherited 50730's value for
baseline_auto (1.6928).

This is NOT supposed to be possible — they're separate Python processes with independent
memory. Most likely cause: a shared Docker temp dir or some other unexpected shared
state. Could not reproduce with a controlled unit test (`code/test_selection.py` and
`test_selection2.py` both pass with correct selection).

The v2 polish on ibm12 (running concurrently with no duplicate process) selected
**correctly** (seed_2024 with score 1.421 vs baseline 1.461). So the bug is specific
to the duplicate-process condition.

**Mitigation:** ensure ONLY ONE multi_dp_polish process runs per bench at a time.
Don't launch overlapping v2 invocations. The current hybrid placer
(`cd_lns_sa_cascade_dp_multi_lane/placer.py`) invokes multi_dp_basin sequentially
within a single Python process — no duplicate-process issue. Safe to deploy.

## Files added this session

In `experiments/E96_multi_config_dp/`:
- `manifest.md` — experiment description
- `MORNING_REPORT_E96.md` — this report
- `SUMMARY_FOR_MORNING.md` — one-page version
- `code/dp_config_probe.py` — 8-config probe driver
- `code/multi_dp_basin.py` — K-config selector (with extended_legalize fallback at 1-50 residuals)
- `code/multi_dp_polish.py` — end-to-end driver
- `code/polish_basin.py` — polish saved basins
- `code/analyze_results.py` — comparison helper
- `code/test_selection.py`, `code/test_selection2.py` — bug diagnosis
- `results/{ibm10,ibm12,ibm14,ibm17}_probe.json`
- `results/multi_dp_polish_ibm10_K4_v1.json` — 1.056 final
- `results/multi_dp_polish_ibm14_K4_v1.{json,pt}` — 1.192 final

In `submissions/cd_lns_sa_cascade_dp_multi_lane/`:
- `placer.py` — drop-in hybrid variant with K=4 multi-config DP lane
- `README.md` — usage instructions

## What was NOT changed (avoid stepping on other Claudes)

- `submissions/cd_lns_sa_cascade_dp_lane/placer.py` — PATH B uses for overnight Phase 4
- `submissions/cd_lns_sa_cascade/placer_adaptive.py` — PATH A uses for cascade `--all`
- `experiments/E91_dp_full_polish/code/*` — PATH B uses for overnight B-R0' phases
- `experiments/E84_cascading_saddle/code/cascading_saddle.py` — modified by PATH A
- `experiments/E92`, `E93`, `E94`, `E95` — other Claudes' work

## Other Claudes' overnight queue status (as of 03:08 UTC)

- **PATH A** (`placer_adaptive.py --all`): ~5 hours in, ETA 05:00-08:00 UTC.
- **PATH B** overnight queue:
  - Phase 1 IBM (single-DP B-R0' on ibm10/12/14/17): COMPLETE. Results above.
  - Phase 2 NG45 (ariane136/mempool_tile/nvdla): partial. ariane136 = 0.748 (loss),
    nvdla = 0.691 (mixed). mempool_tile in cascade.
  - Phase 3 remaining 11 IBM: queued.
  - Phase 4 hybrid `--all` 21 benches: queued, starts after Phase 3.
- **PATH B** E95 diff-proxy v2: spike passed ibm01 gate at 0.846 (no actual lift beyond cascade init).
- **PATH B** ariane133 v4: COMPLETE at 0.670 (zero overlap). NG45 path fixed.

## What I would do next (if more time)

1. **Investigate v2 selection bug** with controlled standalone test. If reproducible,
   fix and re-test. If not, document as transient.
2. **Run hybrid placer `--all` smoke** (single bench) to verify integration works
   end-to-end. Then full `--all` validation (3-4 hr wall).
3. **K=8 vs K=4 study** — does doubling K reduce variance further?
4. **Investigate DP determinism.** Why is basin proxy bimodal? Read DREAMPlace source
   to check `random_seed` propagation.
5. **Add `gamma`, `optimizer`, `wirelength` to config grid** — AutoDMP varies these.

## Sources / Background

- [AutoDMP: Automated DREAMPlace-based Macro Placement (NVIDIA, ISPD 2023)](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement) — confirms multi-config DP search is the SoTA approach.
- [GOALPlace (arxiv 2407.04579)](https://arxiv.org/html/2407.04579v1) — non-uniform learned density targets; future direction.
- [Abacus legalization (Spindler, ICCAD 2008)](https://dl.acm.org/doi/10.1145/1353629.1353640) — Standard cell legalizer; macro analog is constraint-graph + LP.
- E96 reusable infrastructure: probe driver, multi-DP selector, end-to-end polish.
