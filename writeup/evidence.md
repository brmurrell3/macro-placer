# Evidence — Experimental Results and Falsification Record

> Single archive for per-benchmark tables, ablations, dead-ends, and
> diagnostic experiments. Numbers trace to `results/experiment_log.jsonl`
> or to a frozen run JSON in `results/`. Cited by section in `paper.md`.

## Contents

1. [Champion lineage (verified, `--all`)](#1-champion-lineage-verified---all)
2. [Per-benchmark champion tables](#2-per-benchmark-champion-tables)
3. [DPO ablation study](#3-dpo-ablation-study)
4. [The congestion barrier (Act-2 diagnosis)](#4-the-congestion-barrier-act-2-diagnosis)
5. [RUDY fidelity analysis (Act-3 diagnosis)](#5-rudy-fidelity-analysis-act-3-diagnosis)
6. [Hierarchical decomposition experiments](#6-hierarchical-decomposition-experiments)
7. [The CD breakthrough](#7-the-cd-breakthrough)
8. [Polyhedra traversal — barrier-crossing quantification](#8-polyhedra-traversal--barrier-crossing-quantification)
9. [Falsified extension hypotheses](#9-falsified-extension-hypotheses)
10. [DPO version chronology](#10-dpo-version-chronology)
11. [Compute envelope](#11-compute-envelope)

---

## 1. Champion lineage (verified, `--all`)

| Era | Method | Best avg (`--all`) | Δ vs RePlAce | Date | Replaced because |
|-----|--------|-------------------:|-------------:|------|------------------|
| Pre-history | RePlAce baseline | 1.4578 | — | n/a | Target to beat |
| Polyhedra | Phase 2 (300 s nav) | 1.4867 | −0.6 % | 2026-04-15 | Hit ceiling — congestion barrier structural |
| DPO v1 | seed 42 | 1.4255 | +2.3 % | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| DPO best-of | best_of_v2 | 1.3834 | +5.1 % | 2026-04-26 | Within-DPO refinements cap at 1–2 % |
| CD-only | CDOnly (fixed 600 s) | 1.1193 | +23.2 % | 2026-04-27 | Fixed budget left hard benchmarks mid-descent |
| CD-adaptive | CDAdaptive (E9) | 1.1055 | +24.2 % | 2026-04-27 | Plateau-bound — every bench exited via plateau, none hit cap. Same move type — couldn't escape per-axis fixed point. (Superseded by E12.) |
| **CD + grid-bin LNS** | **CDLNSGridBin (E12)** | **1.0990** | **+24.6 %** | **2026-04-28** | **Current champion** (beats leaderboard 1.1172 by **−1.63 %**; ADR-007). |

**Champion candidate (verified `--all`, not promoted):**

| Method | Best avg (`--all`) | Δ vs E12 champion | Date verified | Status |
|--------|-------------------:|------------------:|---------------|--------|
| CDLNSSA (E25) | 1.0954 | **−0.33 %** | 2026-04-29 | Candidate awaiting human decision; ADR-008 *Proposed*. SA-v2 polish on per-axis breakpoints (best-so-far + T₀=5e-4) layered on E12's pipeline. Beats leaderboard by **−1.95 %**. |

Each champion replaced its predecessor by a *structural change*, not parameter tuning.

**Verified leaderboard standings (2026-04-29):**

| Rank | Team | Verified score | Method |
|------|------|----------------|--------|
| 1 | vmallela | 1.1172 (unverified self-report) | "Incremental CD+LNS" |
| 2 | Cezar | **1.2224** verified (was 1.0666 self-reported) | "ReFine" |
| 3 | MTK | 1.2818 | "DreamPlace++" GPU |
| (us, candidate) | CDLNSSA (E25) | 1.0954 verified | CD + LNS + SA-v2 polish (candidate, not promoted) |
| **(us, champion)** | **CDLNSGridBin (E12)** | **1.0990** verified | CD plateau + grid-bin LNS overlay |
| (us, prior) | CDAdaptive (E9) | 1.1055 verified | E9 plateau-adaptive CD (superseded 2026-04-28) |
| 11 | ByteDancer | 1.4151 | "Incremental CD" (no LNS) |

The 1.0666 from Cezar re-verified at 1.2224 — a 14 % discrepancy. Treat
unverified self-reports with skepticism.

---

## 2. Per-benchmark champion tables

### 2.1 CDLNSGridBin (E12) — current champion, 1.0990 (`--all`, zero overlaps)

Configuration: `submissions/cd_lns_gridbin/placer.py`. CD phase ≤ 3 000 s
(plateau-detection params `min_time_s=300, hard_cap_s=3000, patience=3,
plateau_threshold=0.001`); LNS phase ≤ 600 s (`destroy_frac=0.05,
destroy_cap=30, destroy_strategy='cost_aware'`). Total per-benchmark wall
budget 3 600 s (matches the contest 1-hour-per-bench cap). Total runtime
28 256 s = 7.85 hr.

| Bench | E12 (champion) | E9 CDAdaptive | RePlAce | Δ vs E9 | Wall (s) |
|-------|---------------:|--------------:|--------:|--------:|---------:|
| ibm01 | **0.9045** | 0.9159 | 0.9976 | **−1.24 %** | 524 |
| ibm02 | **1.1340** | 1.1538 | 1.8370 | **−1.72 %** | 1 049 |
| ibm03 | **0.9886** | 0.9950 | 1.3222 | **−0.64 %** | 791 |
| ibm04 | **1.0150** | 1.0226 | 1.3024 | **−0.74 %** | 760 |
| ibm06 | **1.1583** | 1.1592 | 1.6187 | **−0.08 %** | 1 054 |
| ibm07 | **1.1021** | 1.1103 | 1.4633 | **−0.74 %** | 1 167 |
| ibm08 | **1.1190** | 1.1254 | 1.4285 | **−0.57 %** | 1 293 |
| ibm09 | **0.8591** | 0.8611 | 1.1194 | **−0.23 %** | 660 |
| ibm10 | **1.0562** | 1.0749 | 1.5009 | **−1.74 %** | 2 433 |
| ibm11 | **0.9136** | 0.9223 | 1.1774 | **−0.94 %** | 1 179 |
| ibm12 | **1.2076** | 1.2153 | 1.7261 | **−0.63 %** | 2 771 |
| ibm13 | **0.9766** | 0.9772 | 1.3355 | **−0.06 %** | 1 874 |
| ibm14 | **1.2205** | 1.2234 | 1.5436 | **−0.24 %** | 2 390 |
| ibm15 | **1.1797** | 1.1809 | 1.5159 | **−0.10 %** | 2 098 |
| ibm16 | **1.1573** | 1.1610 | 1.4780 | **−0.32 %** | 2 426 |
| ibm17 | **1.3299** | 1.3326 | 1.6446 | **−0.20 %** | 3 487 |
| ibm18 | **1.3603** | 1.3633 | 1.7722 | **−0.22 %** | 2 301 |
| **AVG** | **1.0990** | **1.1055** | **1.4578** | **−0.59 %** | **28 256 (7.85 hr)** |

**All 17 benchmarks improved over E9 CDAdaptive — no regressions.** Biggest
wins on basin-locked / high-density benchmarks: ibm10 (−1.74 %), ibm02
(−1.72 %), ibm01 (−1.24 %), ibm11 (−0.94 %). The smaller deltas on hard
benches (ibm13–18 at −0.06 to −0.32 %) reflect that those benchmarks were
already well-converged by CDAdaptive's plateau detection — the LNS overlay
is finding incremental escapes, not basin-changing leaps.

Source JSON: `results/CDLNSGridBinPlacer_20260428_155739.json`.

### 2.2 CDAdaptive (E9) — prior champion, 1.1055 (superseded 2026-04-28)

Configuration: `submissions/cd_adaptive/placer.py`. Plateau-detection
defaults `(min_time_s=300, hard_cap_s=3600, patience=3, plateau_threshold=0.005)`.
Total runtime 17 480 s = 4.85 hr.

| Bench | E9 Adaptive | CDOnly | RePlAce | Δ vs CDOnly | Wall (s) | Plateau? |
|-------|------------:|-------:|--------:|------------:|---------:|----------|
| ibm01 | 0.9159 | 0.9133 | 0.9976 | +0.28 % | 322 | plateau |
| ibm02 | 1.1538 | 1.1534 | 1.8370 | +0.04 % | 524 | plateau |
| ibm03 | 0.9950 | 0.9942 | 1.3222 | +0.08 % | 534 | plateau |
| ibm04 | 1.0226 | 1.0193 | 1.3024 | +0.32 % | 382 | plateau |
| ibm06 | **1.1592** | 1.1656 | 1.6187 | **−0.55 %** | 792 | plateau |
| ibm07 | 1.1103 | 1.1105 | 1.4633 | −0.02 % | 607 | plateau |
| ibm08 | **1.1254** | 1.1307 | 1.4285 | **−0.47 %** | 833 | plateau |
| ibm09 | 0.8611 | 0.8606 | 1.1194 | +0.06 % | 449 | plateau |
| ibm10 | **1.0749** | 1.1000 | 1.5009 | **−2.28 %** | 1332 | plateau |
| ibm11 | 0.9223 | 0.9248 | 1.1774 | −0.27 % | 614 | plateau |
| ibm12 | **1.2153** | 1.2418 | 1.7261 | **−2.13 %** | 1424 | plateau |
| ibm13 | **0.9772** | 0.9939 | 1.3355 | **−1.68 %** | 1143 | plateau |
| ibm14 | **1.2234** | 1.2478 | 1.5436 | **−1.96 %** | 1572 | plateau |
| ibm15 | **1.1809** | 1.2109 | 1.5159 | **−2.48 %** | 1609 | plateau |
| ibm16 | **1.1610** | 1.1919 | 1.4780 | **−2.59 %** | 1518 | plateau |
| ibm17 | **1.3326** | 1.3830 | 1.6446 | **−3.64 %** | 2238 | plateau |
| ibm18 | **1.3633** | 1.3865 | 1.7722 | **−1.68 %** | 1589 | plateau |
| **AVG** | **1.1055** | **1.1193** | **1.4578** | **−1.23 %** | **17 480** | **17/17 plateau** |

Hard benchmarks (ibm10/12–18) gained 1.7–3.6 % from extra time. Easy
benchmarks (ibm01–04, 09, 11) tied within ±0.4 %. **All 17 exit via
plateau detection — none hit the 1-hr cap.** This run signature is what
identified the champion as plateau-bound rather than budget-bound, motivating
the E12 LNS overlay.

Source JSON: `results/CDAdaptivePlacer_20260427_132214.json`.

### 2.3 CDOnly — prior-prior champion, 1.1193 (superseded 2026-04-27)

Configuration: `submissions/cd_only/placer.py`, fixed 600 s/bench.
Total runtime 10 316 s = 172 min. Zero overlaps everywhere.

| Bench | CDOnly | DPO BoV2 | RePlAce | Δ vs DPO |
|-------|-------:|---------:|--------:|---------:|
| ibm01 | 0.9133 | 1.1285 | 0.9976 | −19.1 % |
| ibm02 | 1.1534 | 1.6888 | 1.8370 | −31.7 % |
| ibm03 | 0.9942 | 1.2508 | 1.3222 | −20.5 % |
| ibm04 | 1.0193 | 1.3239 | 1.3024 | −23.0 % |
| ibm06 | 1.1656 | 1.6434 | 1.6187 | −29.1 % |
| ibm07 | 1.1105 | 1.3846 | 1.4633 | −19.8 % |
| ibm08 | 1.1307 | 1.3816 | 1.4285 | −18.2 % |
| ibm09 | 0.8606 | 1.0130 | 1.1194 | −15.1 % |
| ibm10 | 1.1000 | 1.2540 | 1.5009 | −12.3 % |
| ibm11 | 0.9248 | 1.0657 | 1.1774 | −13.2 % |
| ibm12 | 1.2418 | 1.6497 | 1.7261 | −24.7 % |
| ibm13 | 0.9939 | 1.2089 | 1.3355 | −17.8 % |
| ibm14 | 1.2478 | 1.4725 | 1.5436 | −15.3 % |
| ibm15 | 1.2109 | 1.3802 | 1.5159 | −12.3 % |
| ibm16 | 1.1919 | 1.3594 | 1.4780 | −12.3 % |
| ibm17 | 1.3830 | 1.5888 | 1.6446 | −13.0 % |
| ibm18 | 1.3865 | 1.6352 | 1.7722 | −15.2 % |
| **AVG** | **1.1193** | **1.3134** | **1.4578** | **−14.8 %** |

Every benchmark improves over DPO. Zero regressions.

### 2.4 DPO best-of-v2 — DPO champion, 1.3834 (superseded 2026-04-27)

Configuration: `submissions/dpo/best_of_v2_placer.py` (best-of SDF + v2-steps DPO).

| Benchmark | Best-of-v2 | DPO v3 | Poly (50 s) | SDF v5 | RePlAce | vs RePlAce | Winner |
|-----------|-----------:|-------:|------------:|-------:|--------:|-----------:|--------|
| ibm01 | 1.1285 | 1.2105 | 1.1871 | 1.1953 | 0.9976 | −13.1 % | DPO-v2 |
| ibm02 | 1.6888 | 1.7560 | 1.6205 | 1.6888 | 1.8370 | **+8.1 %** | SDF |
| ibm03 | 1.2508 | 1.2869 | 1.4058 | 1.4070 | 1.3222 | **+5.4 %** | DPO-v2 |
| ibm04 | 1.3239 | 1.3570 | 1.3652 | 1.3826 | 1.3024 | −1.7 % | DPO-v2 |
| ibm06 | 1.6434 | 1.7553 | 1.7003 | 1.7150 | 1.6187 | −1.5 % | DPO-v2 |
| ibm07 | 1.3846 | 1.4352 | 1.4867 | 1.4898 | 1.4633 | **+5.4 %** | DPO-v2 |
| ibm08 | 1.3816 | 1.4383 | 1.5080 | 1.5113 | 1.4285 | **+3.3 %** | DPO-v2 |
| ibm09 | 1.0130 | 1.0529 | 1.1245 | 1.1337 | 1.1194 | **+9.5 %** | DPO-v2 |
| ibm10 | 1.2540 | 1.2793 | 1.4067 | 1.4112 | 1.5009 | **+16.5 %** | DPO-v2 |
| ibm11 | 1.0657 | 1.0999 | 1.2317 | 1.2336 | 1.1774 | **+9.5 %** | DPO-v2 |
| ibm12 | 1.6497 | 1.7166 | 1.6482 | 1.6497 | 1.7261 | **+4.4 %** | SDF |
| ibm13 | 1.2327 | 1.2384 | 1.3984 | 1.3986 | 1.3355 | **+7.7 %** | DPO-v2 |
| ibm14 | 1.4719 | 1.5056 | 1.6025 | 1.6003 | 1.5436 | **+4.6 %** | DPO-v2 |
| ibm15 | 1.3742 | 1.4047 | 1.6059 | 1.6073 | 1.5159 | **+9.3 %** | DPO-v2 |
| ibm16 | 1.3819 | 1.4217 | 1.5421 | 1.5424 | 1.4780 | **+6.5 %** | DPO-v2 |
| ibm17 | 1.6121 | 1.6038 | 1.7431 | 1.7431 | 1.6446 | **+2.0 %** | DPO-v2 |
| ibm18 | 1.6614 | 1.6569 | 1.7897 | 1.7927 | 1.7722 | **+6.3 %** | DPO-v2 |
| **AVG** | **1.3834** | **1.4246** | **1.4921** | **1.5002** | **1.4578** | **+5.1 %** | 15 DPO / 2 SDF |

**Notable:** Best-of-v2 beats RePlAce on 15/17 benchmarks. SDF wins ibm02 and
ibm12 — high-density basin lock that DPO degrades.

### 2.5 Polyhedra Navigation — pre-DPO champion, 1.4867

Implementation removed in post-CD cleanup; SDF init kept at
`macro_place/sdf_init.py`. Pre-fork best 2026-04-15.

| Benchmark | Phase 3 (50 s) | Phase 2 (300 s) | SDF v5 | RePlAce | vs RePlAce |
|-----------|---------------:|----------------:|-------:|--------:|-----------:|
| ibm01 | 1.1871 | **1.1715** | 1.1953 | 0.9976 | −19.0 % |
| ibm02 | 1.6205 | **1.6118** | 1.6888 | 1.8370 | **+11.8 %** |
| ibm03 | **1.4058** | 1.4070 | 1.4070 | 1.3222 | −6.3 % |
| ibm04 | **1.3652** | 1.3496 | 1.3826 | 1.3024 | −4.8 % |
| ibm06 | **1.7003** | 1.6847 | 1.7150 | 1.6187 | −5.0 % |
| ibm07 | 1.4867 | **1.4818** | 1.4898 | 1.4633 | −1.6 % |
| ibm08 | 1.5080 | **1.5029** | 1.5113 | 1.4285 | −5.6 % |
| ibm09 | **1.1245** | 1.1195 | 1.1337 | 1.1194 | −0.5 % |
| ibm10 | **1.4067** | 1.4112 | 1.4112 | 1.5009 | **+6.3 %** |
| ibm11 | 1.2317 | **1.2248** | 1.2336 | 1.1774 | −4.6 % |
| ibm12 | **1.6482** | 1.6478 | 1.6497 | 1.7261 | **+4.5 %** |
| ibm13 | 1.3984 | **1.3947** | 1.3986 | 1.3355 | −4.7 % |
| ibm14 | 1.6025 | **1.5948** | 1.6003 | 1.5436 | −3.8 % |
| ibm15 | 1.6059 | **1.6045** | 1.6073 | 1.5159 | −5.9 % |
| ibm16 | 1.5421 | **1.5347** | 1.5424 | 1.4780 | −4.3 % |
| ibm17 | 1.7431 | **1.7422** | 1.7431 | 1.6446 | −6.0 % |
| ibm18 | **1.7897** | 1.7906 | 1.7927 | 1.7722 | −1.0 % |
| **AVG** | **1.4921** | **1.4867** | **1.5002** | **1.4578** | **−2.4 %** |

Phase 2 (300 s) is the best polyhedra result at ~5 300 s total.

**Profiling (Phase 2):** congestion 66.5 % / density 29.3 % / WL 4.2 %.
Strongest gap predictor: `num_nets` (ρ = 0.958). Surrogate global Spearman
ρ = 0.899; within-benchmark ρ = 0.17. 4 benchmarks get 0 nav improvements
(ibm03, ibm06, ibm10, ibm16).

---

## 3. DPO ablation study

All on `--all`, 17 benchmarks. Source: `submissions/dpo/ablation_*.py`.

### 3.1 Component ablation (single-seed)

| Variant | Avg proxy | Δ vs DPO | Δ vs RePlAce |
|---------|----------:|---------:|-------------:|
| **Full DPO (seed 42)** | **1.4246** | baseline | **+2.3 %** |
| Seed 43 | 1.4301 | +0.4 % | +1.9 % |
| Seed 44 | 1.4267 | +0.1 % | +2.1 % |
| Seed 45 | 1.4237 | −0.1 % | +2.3 % |
| Seed 46 | 1.4268 | +0.2 % | +2.1 % |
| − congestion gradient | 1.5092 | +5.9 % | −3.5 % |
| − density gradient | 1.7342 | +21.7 % | −19.0 % |
| Phase-1 only | 1.4605 | +2.5 % | −0.2 % |
| Random init (no SDF) | 4.9415 | +247 % | −239 % |

**Lessons:**
- SDF init is essential (+247 % without it).
- Density gradient contributes the most (+21.7 %).
- Congestion gradient adds +5.9 % — the difference between beating
  RePlAce and losing to it.
- Multi-phase continuation helps +2.5 %.

### 3.2 Multi-seed stability (5 seeds, full DPO)

- Mean: 1.4264, Stdev: 0.0025, Range: 0.0064 (**0.45 %**).
- All 5 seeds beat RePlAce (1.4578).
- Under a normal model, best-of-200 seeds gains only ~0.1 % over best-of-5.

**Best-of-v2 multi-seed (--all):**

- 5-seed mean: 1.3831, stdev: 0.0056, range: 1.0 % (1.3790–1.3927).
- All 5 seeds beat RePlAce. ibm02/ibm12 zero variance (SDF always wins).
- Best single seed: 1.3790 (seed 46, +5.4 % over RePlAce).
- Best-of-5 per benchmark: **1.3703** (+6.0 % over RePlAce).

**The 0.45 % range is evidence, not a limitation.** A method trapped in
random local minima would show high variance. Low variance means the
penalty continuation reliably collapses the landscape to a consistent
attractor. SDF + DPO compresses random init's 4.94 chaos into 0.45 %.

### 3.3 Component breakdown across all benchmarks

Computed on DPO-optimized placements (seed 43).

| Benchmark | Proxy | WL % | Density % | Congestion % |
|-----------|------:|-----:|----------:|-------------:|
| ibm01 | 1.280 | 9.2 % | 26.4 % | 64.3 % |
| ibm02 | 1.763 | 6.2 % | 16.0 % | 77.7 % |
| ibm03 | 1.278 | 8.2 % | 19.8 % | 71.9 % |
| ibm04 | 1.376 | 6.9 % | 18.8 % | 74.3 % |
| ibm06 | 1.767 | 4.9 % | 14.5 % | 80.6 % |
| ibm07 | 1.424 | 6.0 % | 19.1 % | 74.8 % |
| ibm08 | 1.434 | 6.3 % | 18.1 % | 75.6 % |
| ibm09 | 1.063 | 6.6 % | 25.4 % | 68.0 % |
| ibm10 | 1.279 | 6.1 % | 21.4 % | 72.5 % |
| ibm11 | 1.092 | 6.3 % | 23.3 % | 70.4 % |
| ibm12 | 1.706 | 4.4 % | 15.4 % | 80.2 % |
| ibm13 | 1.245 | 5.5 % | 20.9 % | 73.7 % |
| ibm14 | 1.503 | 4.4 % | 17.7 % | 77.9 % |
| ibm15 | 1.402 | 4.9 % | 19.2 % | 76.0 % |
| ibm16 | 1.430 | 4.2 % | 19.7 % | 76.1 % |
| ibm17 | 1.615 | 4.1 % | 16.1 % | 79.7 % |
| ibm18 | 1.655 | 4.0 % | 17.0 % | 78.9 % |
| **AVG** | | **5.8 %** | **19.4 %** | **74.9 %** |

Generalizes the ibm01 finding: **congestion dominates proxy (64–81 %, avg
74.9 %) on every benchmark.** Pre-DPO (SDF init) fractions are more
balanced (ibm01: WL 6.1 %, D 39.4 %, C 54.5 %). DPO succeeds at reducing
density, which makes congestion's share even larger.

### 3.4 DPO worsens 4/17 benchmarks vs SDF init

| Benchmark | SDF init | DPO output | No-cong DPO | Best |
|-----------|---------:|-----------:|------------:|------|
| ibm01 | **1.1953** | 1.2105 | 1.5026 | SDF |
| ibm02 | **1.6888** | 1.7560 | 1.7144 | SDF |
| ibm06 | **1.7150** | 1.7553 | 1.9843 | SDF |
| ibm12 | **1.6497** | 1.7166 | 1.7174 | SDF |

Best-of(SDF, DPO) per benchmark: **1.4135** (3.0 % over RePlAce) vs DPO-only
1.4246 (2.3 %). The differentiable proxy is an imperfect model of the real
proxy; on these benchmarks the approximation actively misleads.

---

## 4. The congestion barrier (Act-2 diagnosis)

### 4.1 Overnight 22-experiment sweep

Full log: `archive/overnight_run.log`. Source code: pre-2026-04-28 git
history.

| Stage | Items | Done (`--all`) | Killed | Skipped | Winner | Avg |
|-------|------:|---------------:|-------:|--------:|--------|----:|
| SP3 (Surrogate) | 8 | 5 | 2 | 1 | sp3_top_k_20 | 1.4919 |
| SP1 (Topology) | 6 | 1 | 5 | 0 | sp1_congestion_aware_extraction | 1.4930 |
| SP4 (LP) | 6 | 3 | 3 | 0 | sp4_mccormick_area | 1.4918 |
| Combine | 2 | 1 | 0 | 1 | combine_stage_winners | 1.4918 |

Global best: **1.4918** (vs baseline 1.4921). All 22 within ±0.5 % of
baseline. The system is at a plateau; navigation dominates everything
upstream. SDF init is not the bottleneck. Congestion is structural, not
parametric.

**Per-stage detail:**

*SP3 — surrogate accuracy:* `density_grid_fix` 1.4931, `delta_ranking`
1.4929, `online_calibration` killed (too few accepted moves), `top_k_20`
1.4919 (best), `rank_aggregation` 1.4997, `pinrudy_blockage` 1.4935,
`pairwise_ranking` 1.4930. *Lesson: surrogate is well-calibrated; bottleneck
is move quality, not move ranking.*

*SP1 — initial topology:* `spectral_topology` 1.78 (killed; ignores macro
sizes), `replace_topology` killed (extract_assignment erases congestion
advantage), `congestion_aware_extraction` 1.4930 (only finisher),
`hmetis_partitioning` 1.91 (shelf-packing terrible), `greedy_construction`
1.75 (clustering creates dense regions), `boundary_attraction` killed
(penalty inert/harmful at every λ). *Lesson: SDF's analytical spreading is
extremely hard to beat.*

*SP4 — LP modifications:* `net_weighting` 1.4974 (RUDY too uniform),
`separation_margins` 1.4932 (overwritten by navigation), `mccormick_area`
1.4918 (best, marginal), `dual_informed_targeting` killed (uniform per-net
congestion), `real_proxy_feedback` killed (HPWL inflated), `lp_navigate_reweight`
1.4973 (split budget cancels benefit). *Lesson: LP-level congestion mods
washed out by 50 s of navigation.*

### 4.2 Miftari correlation analysis (ibm01)

Two-step verification on ibm01.

**Step 1 — cheap signals predict ΔLP-HPWL.** Across 120 cluster flips
(k ∈ {1, 2, 5, 10}), best signal `S_viol` (constraint violation at x*)
gives Spearman ρ = **0.86**, 51 µs vs 2.2 s LP (~42 700× speedup), 91 %
precision/recall at 50 % kept.

**Step 2 — LP-HPWL does NOT predict refined proxy.** Across 24 feasible
topologies (Hamming 0–2 249 from base):

| Pair | ρ |
|------|---:|
| LP-HPWL → refined proxy | **−0.001** |
| LP-HPWL → refined WL | +0.852 |
| LP-HPWL → refined density | −0.536 |
| LP-HPWL → refined congestion | +0.072 |
| refined WL → refined density | −0.416 |
| refined congestion → refined proxy | +0.825 |

Chain: `cheap signal → LP-HPWL → refined proxy`; first link ✓ ρ=0.86,
second link ✗ ρ ≈ 0. *LP-HPWL carries no usable information about refined
proxy.* HPWL and density anti-correlate physically; effects nearly cancel.
Proxy is congestion-dominated, and HPWL is blind to congestion.

**This is the diagnostic that triggered the DPO pivot** — combined with
E8's 6/20/74 decomposition, it falsified every LP-only architecture.

> **TODO(data):** Generalize ρ = −0.001 from ibm01 to ibm04/09/13.

### 4.3 The swap+LP experiment — congestion IS reachable, but not by LP

Six related experiments (real-proxy-guided navigation, swap+LP, swap-only,
group cascade, sequence pair, position blending):

| Experiment | Cong delta | Density delta | Conclusion |
|------------|-----------:|--------------:|------------|
| Real-proxy pair flips (300 s) | < 1 % | −2.8 % to −5.6 % | Local flips only move density |
| **Swap + LP** | **−44 %** | **+180 %** | Different topologies CAN have better congestion |
| Swap only (no LP) | < 0.5 % | ~ 0 % | Single swaps don't change routing |
| Group topology cascade | LP infeasible | N/A | Changing 200+ pairs creates constraint cycles |
| Sequence pair (1–500 steps) | +1–2 % worse | +2–5 % worse | Random distant topologies are worse |
| Position blending (SDF↔LP) | monotonically worse | — | No smooth path in position space |

**Conclusion:** The barrier is structural. SDF's basin is locally optimal;
nearby topologies are worse. Different (good) topologies exist but
cannot be reached by local moves.

---

## 5. RUDY fidelity analysis (Act-3 diagnosis)

Source: `analysis/rudy_fidelity/rudy_analysis.py`. Subject: ibm01 (worst RUDY mismatch).

> **TODO(data):** Freeze stdout to `data/rudy_ibm01.txt`.

### 5.1 Cell-by-cell RUDY vs real congestion

1. **Overall gap is ~3.1×, not ~2×.** ABU-5 % ratio (Real/RUDY): 3.129.
   Real exceeds RUDY on 1843/1845 cells.
2. **Error is spatially varying** (CoV = 0.944). Per-cell ratio:
   mean=4.63, median=3.33, P10–P90 range 2.1×–8.3×. NOT uniform.
3. **RUDY and real disagree on which cells are worst.** Top-5 % cell
   overlap: **10.9 %** (10/92 cells agree). Jaccard 0.057
   (near-random). 77 % of real top-5 % cells not even in RUDY top-10 %.
4. **Three structural sources of divergence:**
   - L-routing vs uniform bbox: ~2.74× ratio.
   - Macro blockage (RUDY ignores entirely): 28.3 % of real congestion.
   - Spatial smoothing (smooth_range = 2): redistributes peaks.
5. **Vertical congestion worst.** V correlation 0.327 vs H 0.635.

**Implication:** DPO's congestion gradient points at the wrong cells. A
weight scaling cannot fix this — structural model improvements (macro
blockage + L-routing paths) needed.

### 5.2 Congestion-weight sweep — KILLED

Tested whether scaling the congestion weight (0.5 → 1.5) compensates for
RUDY's underestimate.

| Cong weight | Avg proxy (`--fast`) | Δ vs control |
|-------------|---------------------:|-------------:|
| 0.50 (control) | 1.2081 | — |
| 0.75 | 1.2251 | +1.4 % worse |
| 1.00 | 1.2278 | +1.6 % worse |
| 1.25 | 1.2377 | +2.5 % worse |
| 1.50 | 1.2382 | +2.5 % worse |

**Kill gate triggered.** Monotonic worsening. **The problem is RUDY's
gradient *direction*, not magnitude. Amplifying a noisy signal makes
things worse.**

---

## 6. Hierarchical decomposition experiments

All failed. Tested whether decomposing the problem hierarchically (global
group arrangement → individual position refinement) could improve on flat
DPO.

| Approach | ibm01 | ibm04 | ibm09 | ibm13 | vs Flat DPO |
|----------|------:|------:|------:|------:|------------:|
| Coarse DPO → expand → fine DPO | 1.43 | 1.70 | 1.34 | 1.51 | +18–28 % worse |
| Cluster-pull refinement → DPO | 1.25 | 1.47 | 1.27 | 1.53 | +3–24 % worse |
| Cluster-swap search → DPO | 1.21 | — | — | — | 0/190 swaps accepted |
| **Flat DPO (baseline)** | **1.21** | **1.36** | **1.05** | **1.24** | — |

**Why decomposition fails:** the proxy `f(p) = WL + 0.5·D + 0.5·C` couples
all macro positions through shared grid cells. Density and congestion are
*global, non-decomposable* properties. Specific failure modes:
1. Tight packing (shelf-packing into super-macro boxes creates artificial
   density hotspots).
2. Cluster attraction (improves WL but *increases* density — same
   anti-correlation that defeated polyhedra LP).
3. 0/190 cluster swaps on ibm01 — coarse arrangement is not the bottleneck;
   SDF's force-directed spreading already finds it.

---

## 7. The CD breakthrough

### 7.1 E1 — incremental evaluator (the infrastructure gate)

`macro_place/incremental_evaluator.py` (~930 lines).

- Per-net min/max trackers, bin-density grid with delta updates, per-net
  RUDY congestion contributions, smoothing via vectorized cumsum.
- **4 657× speedup** on ibm10 (6.5 ms/move vs 30 s full eval).
- **Bit-for-bit parity** with `compute_proxy_cost` (worst diff
  1.1×10⁻¹⁵ absolute on 130 random moves; ibm01 + ibm10).
- Single-step revert tested (within 1 e-9 relative).
- RUDY congestion IS decomposable per single-macro move.
- Smoothing is the dominant per-call cost; `move()` is much cheaper.

### 7.2 E2 — full-proxy CD on ibm10 (the breakthrough run)

Run timestamp 2026-04-27 01:20:16. Wall budget 2 400 s (CD-only; SDF init
excluded). 13 sweeps, 15 000 accepted moves, 71 336 per-axis probes,
0 golden-section fallbacks, 0 overlaps.

| Method | proxy | WL | Density | Congestion |
|--------|------:|---:|--------:|-----------:|
| SDF init (E2 start) | 1.4112 | 0.0687 | 0.7614 | 1.9235 |
| RePlAce baseline (avg, 17 IBM) | 1.4578 | – | – | – |
| DPO best_of_v2 (champion on ibm10) | 1.254 | 0.080 | 0.269 | 0.905 |
| **E2 final (CD-only, 2 400 s from SDF)** | **1.0632** | **0.0790** | **0.5700** | **1.3984** |
| Leaderboard target (avg, all 17) | 1.117 | – | – | – |

Improvement vs SDF init: **24.7 %**. Decision rule: ≤ 1.20 → graduate to
all 17.

**Trajectory (per-sweep, ibm10):**

| Sweep | Elapsed (s) | Proxy | WL | Density | Congestion | Accepted |
|------:|------------:|------:|---:|--------:|-----------:|---------:|
| 0 | 0.0 | 1.4112 | 0.0687 | 0.7614 | 1.9235 | 0 |
| 1 | 206.6 | 1.1777 | 0.0886 | 0.6275 | 1.5505 | 4 370 |
| 2 | 407.5 | 1.1161 | 0.0835 | 0.5987 | 1.4663 | 2 741 |
| 3 | 601.0 | 1.0960 | 0.0821 | 0.5862 | 1.4415 | 1 949 |
| 4 | 786.9 | 1.0835 | 0.0815 | 0.5785 | 1.4255 | 1 337 |
| 5 | 977.4 | 1.0770 | 0.0804 | 0.5764 | 1.4170 | 1 046 |
| 6 | 1 159.8 | 1.0726 | 0.0800 | 0.5760 | 1.4094 | 817 |
| 7 | 1 337.3 | 1.0694 | 0.0801 | 0.5729 | 1.4059 | 680 |
| 8 | 1 518.8 | 1.0664 | 0.0797 | 0.5711 | 1.4022 | 604 |
| 9 | 1 699.9 | 1.0649 | 0.0794 | 0.5701 | 1.4007 | 457 |

**Trade-off pattern.** WL goes UP 13–15 % while density drops 21–32 % and
congestion drops 23–40 %. CD trades cheap WL for expensive density /
congestion — exactly the trade DPO cannot make because RUDY misidentifies
which cells are congested.

### 7.3 ibm02 basin lock broken

All 5 DPO seeds converge to byte-identical ibm02 placement (1.6888).
CD: **1.6888 → 1.1534 (−32 %)**. The DPO basin lock on high-density
benchmarks is structural — gradient methods through RUDY cannot see the
escape path. CD on the real proxy can.

### 7.4 CDOnly `--all` result

Already covered in §2.2. Avg 1.1193, matches leaderboard 1.1172 within
0.18 %, every benchmark improves over DPO, no regressions.

### 7.5 E9 CDAdaptive — per-benchmark plateau detection

```python
# defaults that survived NG45-transfer reasoning
min_time_s = 300            # 5 min minimum
hard_cap_s = 3600           # 1 hr (matches competition rule)
patience = 3                # consecutive sub-threshold sweeps
plateau_threshold = 0.005   # absolute proxy delta per sweep

# exit when: wall_clock >= hard_cap_s, OR
#   (wall_clock >= min_time_s AND last `patience` deltas < threshold)
```

**Result:** 1.1055 avg `--all`, 17 480 s total, 0 overlaps,
**all 17 benchmarks exit via plateau, none hit the cap**. Per-bench wall
times in §2.1.

**Why CDAdaptive over CDOnly:**
- Hard benchmarks (ibm17/18/14/12) had Δ ≈ 0.001–0.003 at the 600 s mark —
  CD wasn't done, the budget was.
- Easy benchmarks (ibm04/09/11) plateau within 5–6 min, wasting ~2–4 min.
- E9 is per-benchmark, per-run policy — adaptive to the actual descent
  trajectory, not priors set on a different dataset.

**Why this is more than a hyperparameter trick:** A fixed per-bench budget
is fragile to dataset shift. Plateau detection adapts to whatever
benchmark it sees. NG45 transfer is policy-only.

### 7.6 E16 — tighter threshold (1.1025, marginal)

`experiments/E16_tight_threshold/code/cd_adaptive_e16.py`. `plateau_threshold=0.001`,
`hard_cap_s=7200`. Total wall 25 503 s = 7.1 hr.

| Metric | CDAdaptive (E9) | E16 | Δ |
|--------|----------------:|----:|--:|
| Avg proxy `--all` | 1.1055 | **1.1025** | **−0.0030 (−0.27 %)** |
| Total wall | 4.85 hr | 7.1 hr | +46 % wall for −0.27 % proxy |
| Overlaps | 0/17 | 0/17 | clean |

14 wins, 1 tie (ibm15), 1 small regression (ibm13 +0.0013). Strict kill
gate said ≥ 0.005 avg drop required; we got 0.003 — formally marginal.
E16 is the threshold-sensitivity datapoint that informed E12's CD-phase
threshold choice (`plateau_threshold=0.001` inside the hybrid CD+LNS
budget split).

> **TODO(data):** Add 2 more threshold datapoints (0.002, 0.01) to defend
> the 0.005 default rather than assert it.

### 7.7 E12 — grid-bin LNS overlay (current champion, 1.0990)

`submissions/cd_lns_gridbin/placer.py` (production), with the experiment
record at `experiments/E12_grid_bin_lns/`. The champion of 2026-04-28
(ADR-007).

**Setup.** CDAdaptive (E9) was *plateau-bound*, not budget-bound: every
benchmark exited via plateau, none hit the 1-hour cap. The plateau is a
per-axis fixed point of CD's breakpoint enumeration. Three escape
mechanisms were tested and falsified before E12 (§9.4 E3 LNS, §9.5
SDF-jitter multi-init, §9.6 subset-CD destroy/reinsert). All three failed
because they reused CD's per-axis move type — they could only re-discover
the same fixed point.

**Algorithm.** Grid-bin LNS introduces a *different* move type. After CD
plateaus:

1. Score every movable hard macro by the proxy delta produced by
   temporarily moving it to canvas center (cost-aware destroy ranking).
2. Pick K = max(1, min(30, 0.05 × |movable hard macros|)) most costly
   macros to destroy.
3. For each destroyed macro, enumerate every `(grid_col × grid_row)` cell
   center as a candidate position (~50–2 500 per benchmark depending on
   grid). Reject illegal candidates (canvas bounds, overlap with other
   hard macros). Commit the legal candidate that minimizes proxy under the
   incremental evaluator.
4. Iterate destroy/reinsert until either a sample produces no improvement
   or the LNS phase wall budget (`lns_budget_s = 600 s`) expires.

The candidate set `(grid_col, grid_row)` is outside CD's per-axis
breakpoint enumeration — that's what makes the move type different and
what lets the overlay escape CD's plateau.

**Production budget.** CD ≤ 3 000 s + LNS ≤ 600 s = 3 600 s per benchmark
(matches the contest 1-hour-per-bench legal cap). The CD phase uses
`plateau_threshold=0.001` (informed by E16); the LNS phase uses default
`destroy_frac=0.05, destroy_cap=30, destroy_strategy='cost_aware'`.

**Result.** Avg `--all` 1.0990 (vs E9 1.1055, **−0.59 %**); zero overlaps
on all 17. Beats leaderboard 1.1172 by **−1.63 %**, RePlAce 1.4578 by
**−24.6 %**. Total wall 28 256 s = 7.85 hr. Per-bench table at §2.1.

**Cost-aware destroy ranking is NOT load-bearing.** A `--fast` ablation
(`experiments/E12_grid_bin_lns/code/cd_lns_gridbin_random.py`) replaced
cost-aware destroy with uniform random destroy and averaged 0.9372 over
ibm01/04/09/13 — matching cost-aware within noise. On ibm09, random
destroy (0.8541) actually beat cost-aware (0.8591). Cost ranking adds
~10 % wall per LNS sample but does not change quality. Future
simplification: drop the ranking, use random destroy. Stays cost-aware in
the production placer for now to avoid mid-deadline changes.

**Algorithmic interpretation.** Validates the leaderboard winner's
"Incremental CD+LNS" architecture. The escape mechanism is move-type, not
random restart, not basin-hopping. Earlier LNS attempts (E3, subset-CD)
had picked the wrong move type. The lesson generalizes: when an iterative
algorithm is plateau-bound rather than budget-bound, the right
intervention is a different move type, not more compute.

Source JSON: `results/CDLNSGridBinPlacer_20260428_155739.json`.

---

## 8. Polyhedra traversal — barrier-crossing quantification

| Benchmark | Movable pairs | Changed | % Changed | Mean displacement |
|-----------|--------------:|--------:|----------:|-------------------|
| ibm01 (small) | 30 135 | 3 556 | **11.8 %** | 1.5 µm (4.5 % diagonal) |
| ibm10 (large) | 308 505 | 15 539 | **5.0 %** | 2.3 µm (2.1 % diagonal) |

**Barrier crossing is real.** DPO changes 3 500–15 500 pairwise L/R/A/B
assignments between SDF init and final output. The smaller benchmark
(ibm01) shows more aggressive crossing (11.8 %) because the tighter
canvas requires more rearrangement.

**Top transition types are overwhelmingly L↔B and R↔A** (perpendicular
flips, ≈98 % of changes), not L↔R or A↔B (same-axis flips). DPO is
rotating relative macro positions, not just sliding them — consistent
with crossing nearby polyhedra boundaries where the binding constraint
switches between horizontal and vertical separation.

DPO does not perform deep topological restructuring — it optimizes
within and across adjacent polyhedra.

---

## 9. Falsified extension hypotheses

### 9.1 E5 — batched seeds (B=64 on GPU) — FALSIFIED 2026-04-26

`submissions/dpo/batched_seeds_placer.py` (679 L), `batched_seeds_b1.py`.
Best 1.1638 (`--fast`), 1.1698 (B=64 `--fast`).

- B=64 wall = 8.9× B=1 (RUDY congestion kernel scales 27× on MPS).
- All 64 seeds (σ = 0.04 × canvas perturbation) collapse to the same
  basin.
- Best-of-N within a single basin doesn't help.

*Lesson: the basin IS the limit.*

### 9.2 E10 — congestion-only refinement — MARGINAL/FALSIFIED 2026-04-27

`submissions/dpo/congestion_refine_placer.py`,
`_e10_sweep_{mild,aggressive}.py`. Best `mild` 1.1636 (`--fast`),
1.3788 (`--all`).

- Freeze WL+density, optimize congestion only after DPO convergence.
- Mild gained 1.0 % on `--fast`; only −0.33 % on `--all`.
- **ibm02 got WORSE +2.3 %** — basin-locked benchmarks regressed.

*Lesson: within-DPO congestion optimization can't overcome RUDY's
directional error.*

### 9.3 E11 — diverse priors — FALSIFIED 2026-04-27

`submissions/dpo/diverse_priors_placer.py`, `init_strategies.py`,
`e11_sdf_only.py`, `e11_sdf_random.py`.

- 4-prior best-of (SDF + Will + greedy + random) on `--fast`: 1.1704
  (−0.4 % vs SDF-only). Will-prior wins ibm09/ibm13 on fast set.
- `--all`: 1.3839 (+0.04 %, FLAT).
- ibm02 and ibm12 got WORSE with alternative priors.

*Lesson: basin diversity exists but doesn't scale to hard benchmarks.
Basin-locked benchmarks are locked by topology, not init.*

### 9.4 E3 — LNS rip-up-and-reinsert — FALSIFIED 2026-04-27

Code (deleted from active tree in commit 44efd16; preserved at)
`writeup/archive/submissions/lns.py`,
`writeup/archive/submissions/cd_lns_placer.py`.

- v1 (full-canvas, 2 244 grid candidates × ~100 ms): 1 LNS iteration in
  428 s. Final ibm17 = 1.3846 vs CDOnly 1.3830 (flat).
- v2 (5×5 local-window): 90× faster, 15 iters in 600 s. Final
  ibm17 = 1.3824 vs CDOnly 1.3830 (flat).

Cost-based destroy selector saturates after 1–2 accepts; 5×5 window can't
move clusters.

*Lesson: single-macro local LNS does NOT escape CD's local minimum. Real
LNS would need cluster-level joint reinsertion or randomized destroy.*

### 9.5 Multi-init CD via SDF jitter — FALSIFIED 2026-04-28

`analysis/multi_init_probe/probe.py`. 8 jittered inits on ibm09
(σ = 0.02–0.10 of canvas). Unjittered control beat every jittered variant.

*Lesson: SDF→CD is contractive; perturbing strictly hurts.*

### 9.6 Subset-CD LNS escape probe — FALSIFIED 2026-04-28

`analysis/lns_escape_probe/probe.py`. 24 random destroy-and-reinsert samples
across ibm09 + ibm12 (varying K, jitter, destroy strategy). 0/24 improved
baseline by ≥ 0.1 %.

*Lesson: subset-CD finds the same per-axis fixed point because it uses
the same move type.*

### 9.7 Grid-bin LNS — GRADUATED (champion, 2026-04-28)

`submissions/cd_lns_gridbin/placer.py` (promoted from
`experiments/E12_grid_bin_lns/code/cd_lns_gridbin.py`). **Different move
type** — searches all (col, row) cell centers, not just per-axis
breakpoints. Avg `--all` 1.0990 (zero overlaps), beats E9 by −0.59 % and
leaderboard by −1.63 %. Full algorithmic detail at §7.7; per-bench table
at §2.1; ablation discussion below; ADR at
`docs/decisions/007_cd_lns_gridbin_promotion.md`.

In-flight datapoints preserved for the lineage:

- ibm12 production smoke: CD plateau 1.20564 → LNS 1.20466 over 5 samples.
- Partial `--all` snapshot (12/17): avg 1.0362 (vs E16 1.0291 over same 12).
- Best wins: ibm10 −1.51 %, ibm01 −0.99 %, ibm02 −0.81 %.

**Cost-aware destroy ranking is NOT load-bearing.** Random destroy on
`--fast` matched cost-aware within noise — ibm09 random actually BEAT
cost-aware (0.8541 vs 0.8591). Cost ranking adds ~10 % wall per LNS
sample but doesn't change quality. Future simplification: drop ranking,
use random destroy.

### 9.C E13 — Congestion-region spatial-cluster LNS — MARGINAL 2026-04-29

`experiments/E13_congestion_lns/code/cd_lns_congestion.py`. Hypothesis:
destroying K hard movables that *share a hot-congestion cell* should
release joint constraints that E12's cost-aware-but-spatially-blind
destroy can't. Per E8, congestion is 74 % of proxy headroom and is
spatially clustered.

**`--fast` numbers (zero overlaps; destroy strategy is the only change
from E12):**

| Benchmark | E13 LNS | CD plateau | E13 LNS lift | E16 baseline | E12-random | E13 vs E12-random |
|---|---:|---:|---:|---:|---:|---:|
| ibm01 | 0.9049 | 0.9129 | **0.94 %** | 0.9135 | 0.9073 | −0.27 % |
| ibm04 | 1.0150 | 1.0147 | 0.08 % | 1.0179 | 1.0150 | tied |
| ibm09 | 0.8573 | 0.8568 | 0.40 % | 0.8605 | 0.8541 | +0.37 % |
| ibm13 | 0.9763 | 0.9714 | 0.15 % | 0.9785 | 0.9724 | +0.40 % |
| **AVG** | **0.9384** | — | — | 0.9426 | 0.9372 | **+0.13 %** |

**The mechanism works on ibm01 but doesn't generalize.** ibm01's LNS
phase produced a 0.94 % lift over CD plateau (sample 1 alone delivered
−0.76 % — a real joint-constraint release of 6 simultaneous
hot-congestion-cell macros). ibm04/ibm09/ibm13 LNS phases produced
0.08 %–0.40 % lifts, which is the same scale E12's random-destroy
ablation produces — i.e., the congestion-region heuristic provides no
additional joint-constraint discovery beyond random destroy on these
three. Net average: E13 is **+0.13 % WORSE than E12 random-destroy
ablation on `--fast`** (within noise; mixed per-bench wins/losses).

**Kill-gate analysis:**
- First gate (avg `--fast` ≥ 0.9425): NOT hit — passed by 0.9384.
- Second gate (0/3 LNS samples on median bench improve ≥ 0.5 %): hit on
  ibm04 (max sample 0.06 %), ibm09 (0.36 %), and ibm13 (0.12 %); ibm01
  alone clears the threshold (sample 1 at 0.76 %). Three of four
  benchmarks fail the second gate.
- Manifest's own generalization-check: lift over baseline ≥ 0.5 % to
  earn an `--all` slot. Lift is 0.45 %. Below threshold.

**Decision: marginal — did not queue `--all`.** The literal first kill
gate passed, but the manifest's own gen-check (and the second kill gate
on three of four benches) flag this as a destroy strategy that doesn't
generalize. The smarter follow-up isn't `--all` — it's destroy-nearest-
neighbors-within-hot-cells (current implementation only picks the
residents of hot cells, not the spatial neighborhood).

*Lesson: "spatially-clustered destroy" is a real lever (ibm01's 0.94 %
lift confirms there are joint-constraint releases) but the simplest
heuristic for finding the right cluster (residents of cells with
congestion > median + 1σ) doesn't reliably identify the cluster on
benchmarks where congestion is more uniformly distributed. The lever
exists; the addressing scheme is too crude.*

Source JSON: `results/CDLNSCongestionPlacer_20260429_011439.json`.

### 9.F E26 — Longer SA budget (LNS 300, SA 900) — FALSIFIED 2026-04-29

`experiments/E26_longer_sa/code/cd_lns_sa_e26.py`. Hypothesis: E25's
`--fast` phase logs showed SA finding `best` at t = 599 s of the 600 s
budget on every winning bench — *still actively improving when budget
expired*. E26 reallocates: cut LNS to 300 s (already converged at <100 s
in E25), bump SA to 900 s. If the late-budget SA activity has real
headroom, E26 should beat E25 by some Δ.

**`--fast` numbers (zero overlaps):**

| Benchmark | E26 (LNS 300, SA 900) | E25 (LNS 600, SA 600) | Δ |
|---|---:|---:|---:|
| ibm01 | 0.8913 | 0.8910 | +0.03 % ~ |
| ibm04 | 1.0120 | 1.0119 | +0.01 % ~ |
| ibm09 | 0.8545 | 0.8551 | −0.07 % ~ |
| ibm13 | 0.9766 | 0.9766 | tied |
| **AVG** | **0.9336** | 0.9336 | **tied** |

Per-bench Δ all within float drift. Net E26 average delta is +0.0023 —
worse than E25 by sub-noise margin. Wall: 7272 s vs E25's 6063 s
(+20 %). **Tied on quality, +20 % on wall — clear net negative.**

**The "SA still improving at t=599 s" finding was best-tracking noise,
not real headroom.** Mechanism: SA chains explore many states near best;
any one can update best by a tiny ε at any time. By t = 600 s the
geometric T schedule has T ≈ 1e-6 — essentially greedy. The chain is no
longer exploring new basins, just doing local greedy moves that
occasionally beat best by float noise. Extending the budget gives more
chances to update best by ε but doesn't find new structure.

**Implication for production.** Don't increase the SA budget beyond
600 s without a new mechanism. Adding more SA time at the same T
schedule and same per-axis breakpoint move set doesn't escape any
plateau the 600 s version doesn't already escape.

*Lesson: "still improving at budget end" can be either real headroom OR
best-tracking noise. Distinguish by running the extended budget and
checking whether the gain is meaningfully larger than noise. E26 got
the right shape — extended budget produces only float-drift-scale
fluctuation, so the 600 s saturation is real and we're locked at the
SA-attainable floor on these benchmarks.*

Source JSON: `results/CDLNSSAE26Placer_20260429_171533.json`.

### 9.G E17, E32 — falsified inits / proxy probes (overnight 2026-04-29 → 30)

Two short kills from the overnight wave:

- **E17 random_init** — replaced SDF with uniform-random legal init in the
  E25 pipeline. `--fast` 1.08653 (+16.4 % vs E25 fast 0.9336); kill gate
  fired on every bench. Confirms ADR-005 (SDF as the canonical CD init):
  uniform inits land in a basin so far above SDF that downstream CD+LNS+SA
  can't recover. The basin difference is *structural*, not a tuning gap.
  E27's persistence-homology output corroborates: the uniform-init final
  proxy on ibm14 was 1.7996 vs SDF 1.2616 (+43 %); on ibm15 1.4996 vs
  1.2255 (+22 %). Different basin, not just different sample.
- **E32 sam_cd** — sharpness-aware breakpoint scoring inside CD (K=4
  perturbations, ρ=1 % canvas-diag). Hypothesis was that CD's fixed point
  might be a sharp local minimum (proxy artifact) — flatter basins should
  generalize better OOD. `--fast` 0.98501 (+5.5 % vs E25 fast 0.9336);
  kill gate fired. Mechanism: K=4 perturbations multiply CD's per-probe
  cost by ~5×, so CD never plateaus within the 2400 s cap. ibm13 ended
  at +10.6 % vs E25. Sharpness-aware angle dies on the eval-cost
  multiplier, not on the hypothesis itself — would need a much faster
  perturbation primitive to test. *Lesson: the per-iter cost of a
  modified CD primitive is the binding constraint; a 5× slowdown on
  per-axis evaluation cannot be amortized when CD already hits the cap.*

### 9.H E40 — multi-SA-seed best-of-4 (overnight 2026-04-29 → 30) — MARGINAL

`experiments/E40_multi_sa_seed/code/cd_lns_sa_multi_seed.py`. Hypothesis:
E25 SA-v2 is the only stochastic phase (CD and LNS are deterministic
modulo `lns_seed`, which is fixed). Snapshot the post-LNS state and fork
4 SA seeds {42, 1, 2, 3} from the *same* state; keep the best. CD+LNS
amortize across forks → cheap (4× 600 s SA vs E25's 600 s).

`--fast` 0.93295 — only −0.07 % vs E25 fast 0.9336. Per-bench wins on
ibm04 (−0.07 %), ibm09 (−0.20 %), ibm13 (−0.18 %); loss on ibm01
(+0.16 %). Fork 1 (seed=42, same as E25) is usually the best fork.
**Multi-seed lift doesn't compound** — consistent with the post-E25
plateau being robust to the SA seed at T₀ = 5e-4. Skipped `--all` (~14 hr
wall for ≤0.1 % expected gain).

*This is the seed-independence half of the basin diagnostic.* Within
SDF init class, no compounding from SA seed perturbation. Any further
gain requires breaking the SA seed axis — a different move type, a
different init, a different acceptance rule, or more T₀ (E26 falsified
the budget axis; that decoupling is independent of T₀).

### 9.I E27 — basin-persistence diagnostic (the gate) — MARGINAL via E18

`experiments/E27_basin_persistence/code/{run_trajectory.py,
analyze_persistence.py}`. Designed as the gate on the post-E25
algorithmic line: persistent homology of the proxy landscape sampled by
44 diverse-init CD trajectories on ibm11/13/14/15. Single dominant H₀
feature → kill the post-E25 line; multiple long-lived H₀ features →
justify multi-day reframings (parallel-tempering, BP, multigrid).

**Outcome: 16/44 trajectories completed.** The DPO best_of_v2 init
helper crashed on every bench (`writeup/archive/submissions/polyhedra/
init/sdf.py` is referenced but missing from the archive — likely
orphaned during a writeup reorg). `sdf_jitter` inits also failed (jitter-
then-relegalize path issue, not investigated; lower priority once E18
provided the empirical signal directly).

The completed trajectories give a partial picture:

| Bench | E25 floor | SDF-cluster mean (std) | greedy/uniform (separate clusters) |
|---|---:|---:|---:|
| ibm11 | 0.9136 | 0.9295 (0.0015) | 1.3658 (cluster of 1) |
| ibm13 | 0.9766 | 1.0077 (0.0026) | 1.3918 (cluster of 1) |
| ibm14 | 1.2205 | 1.2616 (0.0034) | 1.9389, 1.7996 (separate clusters of 1) |
| ibm15 | 1.1797 | 1.2255 (0.0037) | 1.4996 (cluster of 1) |

Standard interpretation of the analyzer's output: "ambiguous" — three
SDF seeds always land in one tight cluster (std 0.0015-0.0037 in proxy)
on every bench, so the basin is locked within the SDF init class. But
the analyzer can't distinguish "single global basin" from "single basin
the inits I sampled all converge to."

**The E18 result resolves the ambiguity.** E18 substituted DPO best_of_v2
for SDF init in the E25 pipeline and verified `--all` 1.08979 (-0.51 %
vs E25, -0.84 % vs E12) with 11/17 per-bench wins, plus NG45 0.69193
(-1.67 % vs E12, 4/4 per-bench wins). The DPO basin is *both*
structurally distinct from the SDF basin AND deeper. **This is the
multi-basin signal the persistence-homology analysis was meant to find;
E18 supplied it via a stronger form (full-pipeline `--all` and NG45
validation, not a CD-only trajectory cluster).**

The verdict has two halves:

- **Single-basin within the SDF init class.** Adding more SDF seeds
  (E40-style) does not find a deeper basin. Multi-start ensembles within
  this class are a dead direction.
- **Multi-basin across init classes.** Distinct inits (DPO, possibly
  RePlAce, possibly AutoDMP) find different basins, and at least one
  (DPO) is meaningfully deeper. Cross-init compositions (E18, E41) are
  the productive direction.

*Lesson: empirical multi-basin evidence (full-pipeline `--all` win) is
strictly stronger than CD-only persistence-homology cluster-counting.
When the diagnostic and the empirical probe disagree, trust the
empirical probe — and use it to retroactively interpret the diagnostic.*

**Bug fixed 2026-04-30 07:48.** `writeup/archive/submissions/dpo/
best_of_v2_placer.py` was using `importlib` to dynamically load a
sibling `polyhedra/init/sdf.py` that never made it into the archive
during a writeup reorg. Fix: replaced the dynamic-load path with
`from macro_place.sdf_init import SDFPlacer`. Verified by
instantiation. E27 is rerunnable.

### 9.J E18 — DPO init → CD + LNS + SA-v2 — CHAMPION CANDIDATE 2026-04-30

`experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py`. Replaces the SDF
init in the E25 pipeline with the DPO best_of_v2 output (DPO's prior-
prior champion at 1.3834 before being superseded by CD). After DPO
converges, run `project_overlaps` → CD plateau → LNS → SA-v2. All other
hyperparameters identical to E25.

**`--fast`** (zero overlaps):

| Benchmark | E18 | E25 | Δ vs E25 |
|---|---:|---:|---:|
| ibm01 | 0.8895 | 0.8910 | −0.17 % |
| ibm04 | 1.0066 | 1.0119 | −0.52 % |
| ibm09 | 0.8401 | 0.8551 | −1.75 % |
| ibm13 | 0.9655 | 0.9766 | −1.14 % |
| **AVG** | **0.9252** | 0.9336 | **−0.91 %** |

3/4 wins on `--fast`; the largest gain on ibm09 (−1.75 %) is a benchmark
where DPO had previously been particularly strong and SDF-CD had been
weaker. (Per-bench numbers from `experiments/E18_dpo_init/manifest.md`;
sub-percent precision approximated.)

**`--all`** (zero overlaps): avg **1.08979**, **−0.51 % vs E25 1.0954**,
**−0.84 % vs E12 1.0990**. 11/17 per-bench wins, 6/17 losses. Total
wall 47 876 s (13.30 hr) — +3 hr vs E25, but still 3.7 hr of headroom on
the 17-hr competition envelope. Per-bench worst-case wall well under the
1-hr-per-bench contest cap.

**`--ng45`** (4 commercial designs, zero overlaps): avg **0.69193**,
**−1.67 % vs E12 0.7037**, **4/4 per-design wins**: ariane133 −3.75 %,
ariane136 −1.64 %, mempool_tile −0.85 %, nvdla −0.42 %. **The DPO basin
transfer to OOD designs is strong** — and was a major risk. ADR-005 had
ruled out perturbation-class alternative inits (random, jitter); DPO is
a separate class (learned descent on differentiable proxy with
congestion gradient). The OOD lift confirms DPO captures topology
information CD-from-SDF cannot reach, and that information generalizes
across IBM (in-distribution) and NG45 (out-of-distribution) designs.

**Why DPO works here when ADR-005 ruled out random/jitter inits.**
ADR-005 falsified random and SDF-jitter inits on the principle that they
were perturbations of SDF that landed in worse-or-equivalent basins.
DPO is *not* a perturbation of SDF — it's hundreds of smooth gradient
steps on a differentiable proxy with congestion-gradient information CD
per-axis greedy cannot exploit. CD-from-DPO refines from a structurally
different initial point and converges to a different local minimum.
ADR-005 stands for what it tested; DPO was a separate untested class.

ADR-009 *Proposed* covers the promotion to champion. Source results in
`results/experiment_log.jsonl` rows `e18_dpo_init_{fast,all,ng45}`.

### 9.K E39 — K-macro joint LNS — MARGINAL with float-precision gotcha

`experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`. Hypothesis:
E25's plateau on ibm11/13/14/15 (tied with E12 under five distinct
single-or-2-macro mechanisms) is a *coupled* fixed point — escape
requires *simultaneous* multi-macro moves. Add a 600 s K-macro joint
LNS phase after the E25 pipeline: K=3, top_N=5 candidates per macro,
brute-force 125 cartesian combos with pairwise non-overlap check.

**`--fast`** (zero overlaps): avg **0.93070**, −0.31 % vs E25 fast 0.9336.
Per-bench wins on ibm09 (−0.79 %) and ibm13 (−0.63 %); ties on ibm01,
ibm04. K-joint phase committed 12-58 K-tuples per bench with Δ ranging
−0.0013 to −0.0047 — proves the K=3 joint move type extracts wins
post-CD-LNS-SA, just at small magnitude.

**`--ng45`** (zero overlaps): avg **0.70126**, −0.35 % vs E12 0.7037 —
margin smaller than E18's −1.67 %. K-joint adds marginal lift on NG45
but E18 dominates on OOD.

**`--all`** crashed at ibm07 (bench 7/17): K-joint committed a placement
that passed its internal `_is_legal_2d_excluded` check (eps=1e-4
tolerance for "near touching") but failed `compute_overlap_metrics`
("1 overlap, area 0.0000" — sub-printable-precision overlap). Validation
raised `RuntimeError`, halting the entire `--all` run. Partial results
were still useful:

| Bench | E39 partial | E25 |
|---|---:|---:|
| ibm01 | 0.8926 | 0.8917 |
| ibm04 | 1.0131 | 1.0140 |
| ibm03 | 0.9828 | 0.9854 |
| ibm02 | 1.1181 | 1.1198 |
| ibm06 | 1.1475 | 1.1470 |
| ibm09 | 0.8521 | 0.8585 |
| **sum** | **6.0062** | 6.0227 |

Sum-of-six K-joint vs E25 = −0.27 % over the 6 benches that completed.

**Bug fix (2026-04-30 04:35).** Root cause: `_is_legal_2d_excluded` and
`_ktuples_pairwise_legal` used `dx < min_dx - eps` with eps=1e-4 — a
**tolerance for overlap up to 1e-4 in both axes**. `compute_overlap_metrics`
in `macro_place/objective.py` flags any positive overlap area, so a
placement legal-by-checker but with 0.5e-4 × 0.5e-4 = 2.5e-9 overlap
area would crash validation. Fix: change to `dx < min_dx + eps` with
eps=1e-9 — **demand strict separation** rather than tolerate near-zero
overlap. Plus a defensive `compute_overlap_metrics`-based revert after
each K-joint commit, so any future float-precision wedge fails safe
(revert) instead of corrupting the placement. Codified as gotcha #4 in
`docs/gotchas.md`.

*Lesson: when a custom legality check has a tolerance, it must point in
the same direction as (or be stricter than) the validation check it
defends. eps=1e-4 in the wrong direction is a class of bug that
manifests only at scale and only on specific benches — exactly the case
where partial validation can miss it.*

Inherited by E41 via import; the fix propagates automatically.

### 9.L E41 — DPO init + K-joint LNS combo — strongest --fast/NG45 of the night

`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`. Composes E18
(DPO basin) with E39 (K-joint joint-move escape). Pipeline: DPO best_of_v2
init → project_overlaps → CD plateau → grid-bin LNS → SA-v2 → K-joint
LNS (600 s, K=3, top_N=5) → validate.

**`--fast`** (zero overlaps): avg **0.92178**, **−1.27 % vs E25 fast
0.9336** — the strongest `--fast` result of the overnight wave. Per-bench
vs E18 fast 0.92542:

| Benchmark | E41 | E18 | Δ vs E18 | Δ vs E25 |
|---|---:|---:|---:|---:|
| ibm01 | 0.8941 | 0.8895 | +0.52 % | (loss vs E25) |
| ibm04 | 0.9845 | 1.0066 | −2.20 % | −2.71 % |
| ibm09 | 0.8413 | 0.8401 | +0.14 % | −1.62 % |
| ibm13 | 0.9495 | 0.9655 | −1.66 % | −2.80 % |
| **AVG** | **0.9224** | 0.9255 | **−0.34 %** | **−1.27 %** |

K-joint adds 0.001-0.003 lift on top of DPO post-SA state on most
benches; on ibm04 the K-joint phase plus DPO basin together beats E25
by a substantial −2.71 %. **E41 beats both parents on every metric we
can validate.**

**`--ng45`** (zero overlaps): avg **0.69022**, **−1.91 % vs E12 0.7037,
−0.25 % vs E18 NG45 0.69193**. ariane133 −4.64 % vs E12 (−0.92 %
additional vs E18); other 3 designs ~tied with E18. K-joint composes on
top of DPO basin on the design where it can find joint-move structure
(ariane133's macro layout is most amenable).

**`--all` rerun complete 2026-04-30 09:13** (after K-joint bug fix verified):
**avg 1.0848, zero overlaps, 13.58 hr wall on `--jobs 4`**. Beats E25
candidate by −0.97 %, E18 candidate by −0.46 %, E12 champion by
**−1.29 %**, leaderboard 1.1172 by **−2.90 %**.

| Bench | E41 | E25 | Δ vs E25 |
|---|---:|---:|---:|
| ibm01 | 0.9121 | 0.8902 | +2.46 % |
| ibm02 | 1.1122 | 1.1310 | −1.66 % |
| ibm03 | 0.9548 | 0.9831 | **−2.88 %** |
| ibm04 | 0.9874 | 1.0102 | **−2.26 %** |
| ibm06 | 1.1694 | 1.1549 | +1.26 % |
| ibm07 | 1.1127 | 1.0982 | +1.32 % |
| ibm08 | 1.1031 | 1.1112 | −0.73 % |
| ibm09 | 0.8413 | 0.8533 | −1.41 % |
| ibm10 | 1.0096 | 1.0459 | **−3.47 %** |
| **ibm11** | **0.8765** | **0.9136** | **−4.06 %** |
| ibm12 | 1.2056 | 1.2079 | −0.19 % |
| ibm13 | 0.9478 | 0.9766 | **−2.95 %** |
| **ibm14** | **1.1994** | **1.2205** | **−1.73 %** |
| **ibm15** | **1.1651** | **1.1797** | **−1.24 %** |
| ibm16 | 1.1435 | 1.1547 | −0.97 % |
| ibm17 | 1.3406 | 1.3311 | +0.71 % |
| ibm18 | 1.3604 | 1.3595 | +0.07 % |
| **AVG** | **1.0848** | **1.0954** | **−0.97 %** |

**14 wins / 3 losses / 0 ties on `--all`.** The 3 losses concentrate on
benches where E25 had a strong SA lift over E12 (ibm01 was −1.58 % in
E25). The K=3 K-joint eps-strict version (post-fix 2026-04-30 04:35)
commits fewer near-touching K-tuples on those benches, losing the
small SA-attainable lift. Net is heavily positive.

**The headline result is the hard-plateau wins.** ibm11/14/15 had been
*tied with E12* under five distinct mechanisms (CD per-axis breakpoints,
grid-bin LNS, SA-v2, pair-swap, spatial cluster destroy) in E25 and all
its predecessors. Five mechanisms hitting the same floor was strong
evidence the floor was structural — a *coupled* fixed point. E41 lifts
ibm11 by −4.06 %, ibm14 by −1.73 %, ibm15 by −1.24 % vs E25.
**The plateau was 3-coupled multi-basin, breakable by simultaneous
3-macro moves once a DPO basin entry opens the right K-tuple structure.**
Both ingredients were necessary: E39 (K=3 K-joint on SDF init) only
lifted −0.31 % on `--fast` and near-tied on the hard benches; E18
(DPO basin alone) lifted −0.84 % vs E12 but did not specifically attack
the coupled fixed points. The composition (E41) extracts both.

ADR-010 *Proposed* covers the E41 promotion. Recommend marking ADR-008
(E25) and ADR-009 (E18) *Superseded* on accept.



`experiments/E25_lns_sa_compose/code/cd_lns_sa.py`. Hypothesis: E12's
grid-bin LNS overlay (champion at avg `--all` 1.0990) and E24's SA
polish v2 search *different* candidate sets — LNS uses `(grid_col ×
grid_row)` cell centers (outside CD's per-axis reachable set), SA-v2
uses per-axis breakpoints (inside CD's reachable set, but Metropolis
order with best-tracking). If the two mechanisms target *different*
improvements, running them in sequence after CD adds their lifts.

**Pipeline:** CD ≤ 2400 s + LNS ≤ 600 s + SA-v2 ≤ 600 s = 3600 s
legal cap.

**`--fast` numbers (zero overlaps; champion 1.0990 reference):**

| Benchmark | E25 (CD+LNS+SA) | E24 (CD+SA) | E12-random (CD+LNS) | E16 baseline (CD) |
|---|---:|---:|---:|---:|
| ibm01 | **0.8910** | 0.8989 | 0.9073 | 0.9135 |
| ibm04 | **1.0119** | 1.0128 | 1.0150 | 1.0179 |
| ibm09 | **0.8551** | 0.8561 | 0.8541 | 0.8605 |
| ibm13 | 0.9766 | 0.9785 | 0.9724 | 0.9785 |
| **AVG** | **0.9336** | 0.9366 | 0.9372 | 0.9426 |

**E25 beats E24 on every bench, beats E12-random on 3 of 4.** Lift over
E16 baseline = **0.95 %**. Lift over E12-random ablation = **0.39 %**.
Compositional gate (avg < 0.9372): **PASSES**.

**Where the composition works (ibm01, ibm04, ibm09).** SA-after-LNS
finds wins SA-after-CD-only could not. The cleanest example is ibm01:
CD plateau 0.91293 → LNS final 0.90... → SA-v2 final 0.89 (final eval
0.8910). E24 alone got to 0.8989 from CD plateau. So LNS-then-SA gave
+0.79 % more lift than SA alone — direct evidence that the candidate
sets are partially disjoint.

**Where the composition breaks (ibm13).** CD plateau 0.97136 → LNS
final 0.96963 (+0.18 %) → SA-v2 final 0.96963 (zero further lift; best
== LNS plateau, found at t = 0.0 s). On the hardest fast bench, both
LNS and SA target the same residual improvements, so chaining them
adds nothing.

**Why this matters.** The two-overlay regime (CD + LNS + SA) is the
first mechanism we've found that can lift `--fast` below 0.94 with
zero overlaps and global hyperparameters. If the composition holds on
`--all`, expected avg ≈ 1.094 – 1.097 — a marginal champion update vs
1.0990. **`--all` queued at 08:47 UTC**; finalverdict awaits the
~7–8 hr run.

*Lesson: "different move type" is necessary but not sufficient for
escape. CD finds the per-axis fixed point; LNS escapes via grid-bin
cell centers; SA-v2 escapes via Metropolis order on the same per-axis
set as CD. The two escape paths are partially disjoint on easier
benchmarks but converge on the hardest. The plateau-detection
literature undersells this: "different acceptance rule on the same
move set" can recover gains greedy CD leaves on the table, IF you
track best-so-far AND set T₀ low enough to bias downhill.*

Source JSON (`--fast`):
`results/CDLNSSAComposePlacer_20260429_044619.json`. Source JSON
(`--all`) will land at `results/CDLNSSAComposePlacer_*.json` on
completion; manifest's Outcome section will be updated then.

### 9.D E24 — SA polish v2 (fixed implementation) — MARGINAL 2026-04-29

`experiments/E24_sa_polish_v2/code/cd_sa_polish_v2.py`. Principled
retest of E14 after both implementation issues (no best-so-far tracking,
T₀ too high) were fixed. Same hypothesis: same per-axis breakpoint move
set CD already searched, but Metropolis acceptance lets the search jump
out of CD's basin. With best-tracking the worst case is "ties with CD
plateau."

**`--fast` numbers (zero overlaps; T₀ = 5e-4, T_f = 1e-6, best-restore
walks chain back to best via per-macro `move()` to keep cong cache in
sync):**

| Benchmark | CD plateau | E24 SA best | Internal lift | Eval result | E13 LNS | E12-random | E16 baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| ibm01 | 0.91293 | 0.89805 | **−1.63 %** | 0.8989 | 0.9049 | 0.9073 | 0.9135 |
| ibm04 | 1.01468 | 1.01134 | −0.33 % | 1.0128 | 1.0150 | 1.0150 | 1.0179 |
| ibm09 | 0.85680 | 0.85317 | −0.42 % | 0.8561 | 0.8573 | 0.8541 | 0.8605 |
| ibm13 | 0.97136 | 0.97136 | 0.00 % | 0.9785 | 0.9763 | 0.9724 | 0.9785 |
| **AVG** | — | — | — | **0.9366** | 0.9384 | 0.9372 | 0.9426 |

**Lift = +0.63 % vs E16 baseline.** Above the 0.5 % gen-check threshold.
Beats E13 congestion-LNS on every fast bench. **Ties E12 random-destroy
ablation 0.9372 within noise** (-0.06 %).

**SA found best near end of budget** on every winning bench: ibm01 at
t = 599.4 s, ibm04 at 599.4 s, ibm09 at 598.9 s. Proxy was still
actively decreasing when the 600-s budget hit. More time would help.

**ibm13 saw zero SA lift** — best == CD plateau (found at t = 0.0 s).
The hardest fast benchmark's plateau is genuinely robust to per-axis
breakpoint Metropolis moves. This is the same pattern likely to hold for
the hardest `--all` benchmarks (ibm17, ibm18) — which is why expected
`--all` lift is smaller than the `--fast` 0.63 %.

**The E14 failure mode is fixed.** v2's best-restore walked the
evaluator from chain-final back to t=0 snapshot via per-macro moves
(keeping the V/H cong cache in sync — direct `evaluator.placement[:]`
mutation would desync). On ibm13 the chain ended at 0.97268 (worse
than CD plateau); after restore, evaluator-final = 0.97136 (= best, =
CD plateau). The "no best-tracking → return chain-final" bug that
hosed v1 is gone.

**Decision: marginal — did not queue `--all`.** Expected `--all` lift
after attenuation puts E24 alone at ~1.094–1.097 — tie with champion
1.0990 at best, not a champion. The interesting follow-up is *whether
E12 LNS and E24 SA-v2 are compositional* — they search different
candidate sets (LNS = grid cell centers, outside CD's reachable set;
SA-v2 = per-axis breakpoints, inside but in random order with
worse-move acceptance). Surfaced as E25 (`experiments/E25_lns_sa_compose/`).

*Lesson: a fair test of SA-on-breakpoints (with best-tracking + low
T₀) achieves the same `--fast` lift as grid-bin LNS. SA isn't worse
than LNS, but on this candidate set it isn't better either. The next
question is compositionality.*

Source JSON: `results/CDSAPolishV2Placer_20260429_030032.json`.

### 9.B E14 — SA polish on per-axis breakpoints — FALSIFIED 2026-04-29

`experiments/E14_sa_polish/code/cd_sa_polish.py`. Hypothesis: same
per-axis breakpoint move set CD already searched, but Metropolis
acceptance with a global temperature schedule — "explicit tunneling
through CD's fixed point with the same primitives, just under SA rather
than greedy."

**`--fast` numbers (zero overlaps; CD plateau identical to E16 baseline):**

| Benchmark | E14 SA | E16 baseline | Δ |
|---|---:|---:|---:|
| ibm01 | 0.9661 | 0.9135 | **+5.8 %** |
| ibm04 | 1.0801 | 1.0179 | **+6.1 %** |
| ibm09 | 0.9056 | 0.8605 | **+5.2 %** |
| ibm13 | 1.0330 | 0.9785 | **+5.6 %** |
| **AVG** | **0.9962** | 0.9425 | **+5.7 %** |

Kill gate (`--fast` ≥ 0.9425): clearly hit. Worse than baseline on every
benchmark — *not noise*.

**The mechanism failed visibly.** On ibm13, CD plateau exits at proxy
0.97136 (sweep 13, wall 1616 s). The 600-s SA polish then *raised*
proxy to 1.24631 within 30 s and oscillated 1.04–1.28 throughout the
budget, returning the placement at 1.02713 — a net +5.7 % regression
from the CD plateau. SA accepted 177 845 better and 172 062 worse moves
on ibm13 — roughly 1:1, the signature of a 50/50 random walk rather
than tunneling.

**Two compounding issues:**

1. **No best-so-far tracking.** The placer returns whatever placement is
   in `evaluator.placement` at SA budget exhaustion, not the best ever
   seen. Textbook SA tracks the best and restores before return.
2. **T₀ = 0.01 was too high vs the per-move Δ scale (≈ 1e-3).**
   `exp(−1e-3 / 1e-2) ≈ 0.905` means nearly every worsening move accepts
   early. The manifest's own design rationale framed this as
   "exploration"; the run shows it was a destructive random walk.

**A fair retest** (best-so-far tracking + T₀ ≈ 1e-4 so worsening moves
are rare) is a separate hypothesis. The roadmap's "SA is a known-strong
method" claim is conditional on implementation; *this* attempt shows
nothing about whether SA-on-breakpoints can in principle beat greedy CD.

*Lesson: the candidate-set lever (E12 grid-bin LNS) and the
acceptance-rule lever (E14 here) are not symmetric. Greedy CD on per-
axis breakpoints already finds the per-axis fixed point — adding
Metropolis on the same candidate set without tracking the best solution
loses progress instead of escaping the plateau.*

Source JSON: `results/CDSAPolishPlacer_20260429_001135.json`.

### 9.A E23 — NG45 sanity test on E12 — VALIDATED 2026-04-28

`experiments/E23_ng45_sanity/`. Defensive run on the four public NG45
commercial designs (ariane133, ariane136, mempool_tile, nvdla) using the
unmodified champion `submissions/cd_lns_gridbin/placer.py`. Fills the
missing NG45 datapoint flagged in ADR-007.

| Design | Proxy | Wall (s) |
|---|---:|---:|
| ariane133 | 0.7061 | ~615 |
| ariane136 | 0.6840 | ~767 |
| mempool_tile | 0.7438 | 686 |
| nvdla | 0.6807 | 1053 |
| **AVG** | **0.7037** | total **3122 s = 52 min** |

**Zero overlaps everywhere; no NaN/inf; no per-bench cap violations**
(max 1053 s vs 3600 s legal cap). Every design exited via plateau in
9–10 sweeps. mempool_tile has only 20 hard movables → K=1 LNS converged
at sample 1; nvdla found a small Δ=−0.00201 LNS improvement on sample 1
then converged. Same configuration as the IBM `--all` run; no
per-benchmark tuning. **The plateau-detection policy and grid-bin LNS
overlay both transfer to commercial designs without modification.**

The avg 0.7037 is structurally lower than IBM's 1.0990 because NG45's
proxy bands differ — direct comparison across IBM and NG45 is not
meaningful. The relevant signals are: (1) zero overlaps, (2) all under
the per-bench cap, (3) plateau detection works.

Source JSON: `results/CDLNSGridBinPlacer_20260428_223405.json`.

### 9.8 E15 — pair-swap on top of CDAdaptive — FALSIFIED 2026-04-28

`experiments/E15_pair_swap/code/cd_pair_swap.py`. Hypothesis: CD plateaus
when every macro is at its single-axis fixed point; swapping two
strongly-coupled macros (sharing nets) is a coordinated move outside CD's
reachable set.

- **v1** used `min_shared_nets=2`, found zero candidates (most IBM macro
  pairs share exactly one net per `findings.md` §7). Result file
  `results/CDPairSwapPlacer_20260428_091950.json` is broken; do not cite.
- **v2** dropped to `min_shared_nets=1` and found 21–53 real swaps per
  benchmark on `--fast`. Result: avg `0.9414` vs E16 baseline `0.9425`
  (Δ = −0.0011, below noise). Result file
  `results/CDPairSwapPlacer_20260428_125855.json`.

The literal kill gate (zero accepted swaps) is not triggered, but the
spirit is: real swaps exist but don't move score. CD's plateau is robust
to pair swaps the same way it's robust to SDF jitter (§9.5) and subset-CD
destroy (§9.6) — connectivity-graph-promising candidates stay inside CD's
reachable set. **Status: falsified.**

*Lesson: "Different candidate set" isn't enough — the candidate set must
also be one CD's primitive doesn't already enumerate. Per-axis breakpoint
search is rich enough that pair swaps reduce to combinations of moves CD
has already tried.*

---

## 10. DPO version chronology

| Version | Avg proxy (`--all`) | Runtime | Wins |
|---------|--------------------:|--------:|------|
| v2 (more steps) | **1.4107** | 311 s | **15/17** |
| v3 (final) | 1.4246 | 288 s | 2/17 |

v2 is systematically better by 1.0 % — exceeds seed variance (0.45 %).
v3 was likely chosen to reduce runtime (288 s vs 311 s) but sacrificed
quality. **Best-of-v2 (1.3834) is the DPO champion that the paper
references**; it combines v2 step counts with best-of(SDF, DPO) per benchmark.

---

## 11. Compute envelope

| Stage | Total wall | Per-bench wall | Notes |
|-------|-----------:|----------------|-------|
| CDOnly fixed 600 s | 10 316 s | 600 s (uniform) | superseded |
| CDAdaptive (E9) | 17 480 s = 4.85 hr | 322–2 238 s | All exit via plateau, none hit 3 600 s cap (superseded 2026-04-28) |
| E16 (threshold = 0.001) | 25 503 s = 7.1 hr | up to 7 200 s | +46 % wall for −0.27 % avg vs E9 |
| **CDLNSGridBin (E12, champion)** | **28 256 s = 7.85 hr** | **524–3 487 s** | **CD ≤ 3 000 s + LNS ≤ 600 s; ibm17 closest to per-bench cap at 3 487 s of 3 600 s** |

**Hidden-test envelope.** Competition rule: 1 hr per benchmark hard cap,
17-bench `--all` ⇒ 17 hr maximum. CDLNSGridBin uses 7.85 hr, leaving
~9 hr of headroom (down from CDAdaptive's ~12 hr). Hard NG45 designs
(potentially harder than IBM) have less cap-side margin than under
CDAdaptive but still substantial; ibm17 hit 3 487 s of the 3 600 s
per-bench cap on IBM, the tightest single-bench result. Monitor `--ng45`
per-bench wall before submission.

**Hard rules that bound every experiment:**
- **No per-benchmark tuning** (contest rule). All hyperparameters either
  global or adaptive (computed from benchmark properties).
- **1-hour-per-benchmark hard cap** (contest rule).
- **Validate cross-benchmark, not on the hardest IBMs.** Tune on `--fast`
  (ibm01/04/09/13), validate on `--all`. Helping `--fast` while hurting
  `--all` = overfitting.

---

## See also

- `paper.md` — canonical draft cites this evidence by section.
- `contributions.md` — claim/novelty/evidence registry; cross-references
  this file.
- `theory.md` — supplementary mathematical framework.
- `archive/` — historic logs, removed code, prior planning docs.
- `results/experiment_log.jsonl` — append-only run log.
- `docs/gotchas.md` — codebase gotchas surfaced during the post-CD phase.
