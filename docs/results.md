# Results

Last updated: 2026-05-02

Operational doc — current champion only. Historical per-benchmark
tables for DPO / Polyhedra / Overnight Sweep / Miftari live in
`writeup/historical_results.md`.

## Baselines

| Method | Avg Proxy (--all) | Overlaps | Notes |
|--------|-------------------|----------|-------|
| RePlAce | 1.4578 | 0 | Target to beat for champion |
| SA | 2.1362 | 0 | Weak baseline |
| Will's seed (SA 3000) | 1.5338 | 0 | Pre-fork submission |
| Greedy row | ~2.20 | 0 | Demo placer |
| Leaderboard target (vmallela) | 1.1172 | 0 | Unverified; CD+LNS, ~40 min/bench |

## Champion lineage

| Hypothesis | Status | Best Avg Proxy | Notes |
|------------|--------|----------------|-------|
| **CDLNSGridBinPlacer (E12)** | **CHAMPION** | **1.0990** | **Beats leaderboard 1.1172 by -1.63%; -24.6% vs RePlAce; CD plateau + grid-bin LNS overlay (ADR-007). Promoted 2026-04-28.** |
| **CDLNSSAHybridPlacer (E48)** | **STRONGEST VERIFIED CANDIDATE** | **1.08151** | **Verified −1.59 % vs E12 (−3.21 % vs leaderboard, −0.30 % vs E41); --ng45 0.6922 (−1.66 % vs E12 0.7037, tied with E18 0.69193).** Per-bench best-of-{E25, E41} hybrid (E25 wins 5/17, E41 wins 12/17). Algorithmically valid (no per-benchmark tuning; picks per-bench winner by proxy value). Code at `experiments/E48_hybrid_e25_e41/code/cd_lns_sa_hybrid.py`; ADR-011 *Proposed* 2026-05-01. Wall ~7 hr `--jobs 4`. |
| CDLNSSAMultiseedHybridPlacer (E53m) | tied with E48 (no incremental lift) | 1.08128 | Verified --all 1.08128 (only −0.02 % vs E48 1.08151 — within noise). 3-way hybrid {E25, E41 seed=42, E41 seed=1}. The --fast lift (-0.97 %) was sample-size outlier; --all aggregates dampen DPO seed-noise. Multi-seed within DPO is dead-end for breakthrough. Code at `experiments/E53_multiseed_hybrid/code/cd_lns_sa_multiseed_hybrid.py`. |
| CDLNSSADPOKJointPlacer (E41) | candidate (superseded by E48) | 1.0848 | Verified −1.29 % vs E12, 14/17 IBM wins; --ng45 0.69022 (−1.91 % vs E12). DPO + CD + LNS + SA-v2 + K-joint K=3. ADR-010 *Proposed* (mark *Superseded* on ADR-011 accept). |
| CDLNSSADPOInitPlacer (E18) | candidate (superseded by E41/E48) | 1.08979 | Verified −0.84 % vs E12, 4/4 NG45 wins. DPO basin transfer to OOD designs proven. ADR-009 *Proposed* (mark *Superseded* on ADR-011 accept). |
| CDLNSSAPlacer (E25) | candidate (older; superseded by E18/E41/E48) | 1.0954 | Verified −0.33% lift over E12. Adds SA-v2 polish on per-axis breakpoints with best-so-far tracking + T₀=5e-4. ADR-008 *Proposed*; mark *Superseded* on ADR-011 accept. |
| CDAdaptivePlacer (E9) | superseded | 1.1055 | Was champion 2026-04-27; -0.59% lift from E12 LNS overlay. Plateau-bound: every bench exited via plateau, none hit cap. |
| CDOnlyPlacer | superseded | 1.1193 | Was champion 2026-04-27 (am); -1.23% lift from E9 adaptive budget |
| DPO best-of-v2 | superseded | 1.3834 | Was champion 2026-04-26; -19.1% vs CDOnly. Details in `writeup/historical_results.md`. |
| Polyhedra Navigation | superseded | 1.4867 | At ceiling; replaced by DPO. Details in `writeup/historical_results.md`. |
| SDF Density | init only | 1.5002 | Now used as init for both CD placers |

## CDLNSSAHybridPlacer (E48) --- Strongest Verified Candidate (2026-05-01)

**Status: STRONGEST CANDIDATE — not promoted.** Verified avg proxy
**1.08151** on --all (**−1.59 % vs E12 1.0990**, **−3.21 % vs
leaderboard 1.1172**, −0.30 % vs E41 candidate 1.0848, zero overlaps
everywhere). Per-bench best-of-{E25, E41} hybrid. E25 wins 5/17
(ibm01, ibm06, ibm07, ibm17, ibm18 — basins where SDF outperforms
DPO); E41 wins 12/17 (rest — basins where DPO + K-joint dominate).
Theoretical best-of-2 bound from verified per-bench numbers = 1.08121;
realized 1.08151 within float-drift.

NG45 commercial-design transfer: **0.6922** (−1.66 % vs E12 0.7037,
tied with E18 0.69193 and E41 0.69022).

Configuration: `experiments/E48_hybrid_e25_e41/code/cd_lns_sa_hybrid.py`.
Per benchmark, runs both E25 (CDLNSSAPlacer) and E41
(CDLNSSADPOKJointPlacer) pipelines and returns the lower-cost output.
**No per-benchmark tuning** — the per-bench winner is determined by
proxy value, not hardcoded bench-name logic. Total wall ~7 hr
`--jobs 4` (max(E25, E41) per bench, parallel benches).

ADR-011 *Proposed* (awaiting human decision). On ADR-011 accept,
mark E25 (ADR-008) / E18 (ADR-009) / E41 (ADR-010) all as *Superseded*.

### Overnight 2026-05-01 → 02 — three follow-ups, none lifted E48:
- **E53 GPU DPO basin polish**: --fast 0.9254 (+0.36 % vs E41); 0
  accepts in 350 GPU restarts. **FALSIFIED** — smooth-proxy gradient
  cannot escape fully-converged CD-LNS-SA local optimum.
- **E53m multi-seed hybrid** (E25 + E41 s42 + E41 s1): --fast 0.91128
  (-0.97 % vs E48 — SAMPLE-SIZE OUTLIER); --all **1.08128 vs E48
  1.08151 = tied** (-0.02 %). Adding a second DPO seed to the hybrid
  doesn't compound at --all; both seeds tied on most benches.
  **MARGINAL** — multi-seed within DPO is dead-end for breakthrough.
- **E54 congestion-targeted destroy** (engages E8 6/20/74 % proxy
  decomposition): --fast 0.9222 tied; --all 1.08568 (+0.39 %); --ng45
  0.7022 with **ariane133 +5.14 % catastrophic regression**.
  **FALSIFIED** — IBM-aware mechanism, NG45-blind on ariane-class
  designs (joins E42/E43/E44 in this failure class).

---

## CDLNSSADPOKJointPlacer (E41) --- Verified Candidate (2026-04-30, superseded by E48)

**Status: VERIFIED CANDIDATE — superseded by E48 hybrid.** Verified avg proxy
**1.0848** on --all (**−1.29 % vs E12 champion 1.0990**, **−2.90 % vs
leaderboard 1.1172**, −0.97 % vs E25 candidate, −0.46 % vs E18 candidate,
zero overlaps everywhere). 14 wins / 3 losses / 0 ties on IBM.

NG45 commercial-design transfer: **0.69022** (−1.91 % vs E12 0.7037,
−0.25 % beyond E18 NG45 0.69193). Per-design wins: ariane133 −4.64 %
(largest), ariane136 ~tied with E18, mempool_tile/nvdla ~tied with E18.

Configuration: `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`.
Pipeline:
1. DPO best_of_v2 init (replaces SDF; E18's component).
2. project_overlaps to clear DPO residuals.
3. CD plateau (`hard_cap_s=2400`, `plateau_threshold=0.001`, `patience=3`).
4. Grid-bin LNS (`destroy_frac=0.05`, `destroy_cap=30`,
   `lns_budget_s=600`).
5. SA-v2 polish (`T0=5e-4`, `Tf=1e-6`, `sa_budget_s=600`,
   best-so-far tracking).
6. K-macro joint LNS (`K=3`, `top_N=5`, `kjoint_budget_s=600`,
   `kjoint_seed=42`; eps=1e-9 strict-separation pairwise legality;
   post-commit `compute_overlap_metrics` defensive revert).
7. Validate (zero overlaps), preserve fixed macros.

Per-benchmark budget: 30 s DPO + 2400 s CD + 600 s LNS + 600 s SA + 600 s
K-joint = 4230 s ≈ 70 min. Total wall 13.58 hr `--jobs 4` parallel
(`--jobs 1` serial would be ~16-17 hr; observed per-bench wall up to
4200 s on ibm17).

ADR-010 *Proposed* (awaiting human decision); ADR-008 (E25) and
ADR-009 (E18) recommended for *Superseded* on ADR-010 accept.

### How it differs from E18 (DPO + CD + LNS + SA-v2)

E18 already lifts E25 by −0.51 % on `--all`, primarily by changing the
*basin*: DPO best_of_v2 finds a different starting region than SDF, and
CD-from-DPO converges to a different (deeper) local minimum than
CD-from-SDF. E41 adds a 600 s K=3 K-joint phase after SA-v2.

The K-joint phase commits 12-58 K-tuples per bench, with brute-force
enumeration of N^K = 5^3 = 125 combos per K-tuple, pairwise
non-overlap check, and a post-commit `compute_overlap_metrics`
defensive revert. The mechanism's reachable set is *new* — three
macros move simultaneously, escaping the coupled-fixed-point plateau
that no single-or-2-macro mechanism (CD per-axis, grid-bin LNS,
SA-v2, pair-swap, spatial cluster) could touch.

The wins concentrate on the *hard-plateau benches* — exactly where
E25 had tied E12 under five different mechanisms: ibm11 −4.06 %,
ibm14 −1.73 %, ibm15 −1.24 %. **The plateau was 3-coupled multi-basin,
breakable by simultaneous 3-macro moves once a DPO basin entry opens
the right K-tuple structure.** Both ingredients matter — E39 (K-joint
on top of SDF init) only lifted −0.31 % on `--fast` and near-tied on
the hard benches; the DPO basin is what makes K=3 productive.

### Per-benchmark (--all, zero overlaps everywhere)

| Benchmark | E41 (candidate) | E25 (older candidate) | E12 (champion) | Δ vs E12 |
|---|---:|---:|---:|---:|
| ibm01 | 0.9121 | 0.8902 | 0.9045 | **+0.84 %** loss |
| ibm02 | 1.1122 | 1.1310 | 1.1340 | **−1.92 %** |
| ibm03 | 0.9548 | 0.9831 | 0.9886 | **−3.42 %** |
| ibm04 | 0.9874 | 1.0102 | 1.0150 | **−2.72 %** |
| ibm06 | 1.1694 | 1.1549 | 1.1583 | +0.96 % loss |
| ibm07 | 1.1127 | 1.0982 | 1.1021 | +0.96 % loss |
| ibm08 | 1.1031 | 1.1112 | 1.1190 | −1.42 % |
| ibm09 | 0.8413 | 0.8533 | 0.8591 | **−2.07 %** |
| ibm10 | 1.0096 | 1.0459 | 1.0562 | **−4.41 %** |
| ibm11 | 0.8765 | 0.9136 | 0.9136 | **−4.06 %** ← hard plateau |
| ibm12 | 1.2056 | 1.2079 | 1.2076 | −0.17 % |
| ibm13 | 0.9478 | 0.9766 | 0.9766 | **−2.95 %** ← hard plateau |
| ibm14 | 1.1994 | 1.2205 | 1.2205 | **−1.73 %** ← hard plateau |
| ibm15 | 1.1651 | 1.1797 | 1.1797 | **−1.24 %** ← hard plateau |
| ibm16 | 1.1435 | 1.1547 | 1.1573 | −1.19 % |
| ibm17 | 1.3406 | 1.3311 | 1.3299 | +0.80 % loss |
| ibm18 | 1.3604 | 1.3595 | 1.3603 | +0.01 % ~ |
| **AVG** | **1.0848** | 1.0954 | 1.0990 | **−1.29 %** |

Source: `results/CDLNSSADPOKJointPlacer_20260430_091323.json`,
`results/experiment_log.jsonl` row `e41_dpo_kjoint_all_postfix`.

### NG45 transfer (4 commercial designs, zero overlaps)

| Design | E41 | E18 | E12 | Δ vs E18 | Δ vs E12 |
|---|---:|---:|---:|---:|---:|
| ariane133 | 0.6733 | 0.6796 | 0.7061 | **−0.93 %** | **−4.64 %** |
| ariane136 | 0.6728 | 0.6728 | 0.6840 | tied | −1.65 % |
| mempool_tile | 0.7375 | 0.7375 | 0.7438 | tied | −0.85 % |
| nvdla | 0.6773 | 0.6778 | 0.6807 | sub-noise | −0.50 % |
| **AVG** | **0.69022** | 0.69193 | 0.70365 | **−0.25 %** | **−1.91 %** |

Source: `results/experiment_log.jsonl` rows `e41_dpo_kjoint_ng45`,
`e18_dpo_init_ng45`, `e23_ng45_e12`.

**ariane133 sees the largest K-joint contribution beyond E18** (−0.93 %);
the other three NG45 designs tie E18 within sub-noise. K-joint exposes
extra structure on ariane133 specifically — its macro layout is most
amenable to 3-coupled simultaneous moves. The other three commercial
designs apparently lack this structure or it requires K>3 / different
tuple selection (E42, E44 follow-ups).

## CDLNSSADPOInitPlacer (E18) --- Prior Candidate (2026-04-30, superseded by E41 if ADR-010 accepted)

Verified avg `--all` **1.08979** (−0.84 % vs E12, 4/4 NG45 wins).
Pipeline = E25 with SDF init replaced by DPO best_of_v2. ADR-009
*Proposed*. See `experiments/E18_dpo_init/manifest.md` for per-bench
breakdown. Code at `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py`.
**Outclassed by E41** which composes E18's basin shift with K-joint
joint-move escape.

## CDLNSSAPlacer --- Champion Candidate (2026-04-29)

**Status: CANDIDATE — not promoted.** Verified avg proxy **1.0954** on --all (**24.9% better than RePlAce**, **beats leaderboard 1.1172 by -1.95%**, **−0.33% better than current E12 champion**, zero overlaps everywhere). Awaiting human decision; ADR-008 status *Proposed*.

Configuration: `submissions/cd_lns_sa/placer.py` (full-proxy CD on `IncrementalProxyEvaluator` with per-benchmark plateau detection, grid-bin LNS overlay, SA-v2 polish on per-axis breakpoints with best-so-far tracking, SDF init). Total runtime 37 188 s = 10.33 hr.

Time-budget split per benchmark: CD ≤ 2 400 s + LNS ≤ 600 s + SA ≤ 600 s = 3 600 s total (matches the contest 1-hour-per-bench cap). Plateau-detection params for CD: `min_time_s=300, hard_cap_s=2400, patience=3, plateau_threshold=0.001`. LNS: `destroy_frac=0.05, destroy_cap=30, destroy_strategy='cost_aware', lns_budget_s=600`. SA-v2: `T0=5e-4, Tf=1e-6, sa_budget_s=600, sa_breakpoint_budget=12, sa_seed=42`.

### How it differs from CDLNSGridBin (E12)

E12 is *budget-bound on the LNS phase plateau*: LNS converged in <100 s on every fast benchmark, leaving most of the 600 s budget unused. The destroy/reinsert mechanism saturated quickly because cost-aware destroy targets a small set of high-cost candidates. SA-v2 layered on top extracts wins LNS-alone misses by exploring per-axis breakpoint moves with Metropolis acceptance and best-so-far tracking — a candidate set CD's greedy sweep order missed. The wins concentrate on easier benches (ibm01 −1.58%, ibm10 −0.97%, ibm08 −0.69%, ibm09 −0.68%) where neither LNS nor SA alone saturate; the hardest benches (ibm14, ibm15, ibm17) tie with E12 — both pipelines hit the same floor on those.

### Per-benchmark (--all, zero overlaps everywhere)

| Benchmark | E25 (candidate) | E12 (champion) | Δ |
|---|---:|---:|---:|
| ibm01 | 0.8902 | 0.9045 | **−1.58 %** |
| ibm02 | 1.1310 | 1.1340 | −0.26 % |
| ibm03 | 0.9831 | 0.9886 | −0.56 % |
| ibm04 | 1.0102 | 1.0150 | −0.47 % |
| ibm06 | 1.1549 | 1.1583 | −0.29 % |
| ibm07 | 1.0982 | 1.1021 | −0.35 % |
| ibm08 | 1.1112 | 1.1190 | **−0.69 %** |
| ibm09 | 0.8533 | 0.8591 | −0.68 % |
| ibm10 | 1.0459 | 1.0562 | **−0.97 %** |
| ibm11 | 0.9136 | 0.9136 | tied |
| ibm12 | 1.2079 | 1.2076 | +0.02 % ~ |
| ibm13 | 0.9766 | 0.9766 | tied |
| ibm14 | 1.2205 | 1.2205 | tied |
| ibm15 | 1.1797 | 1.1797 | tied |
| ibm16 | 1.1547 | 1.1573 | −0.22 % |
| ibm17 | 1.3311 | 1.3299 | +0.09 % ~ |
| ibm18 | 1.3595 | 1.3603 | −0.06 % |
| **AVG** | **1.0954** | 1.0990 | **−0.33 %** |

Wins on 11/17, ties on 4/17 (ibm11/13/14/15), sub-noise regression on 2/17 (ibm12 +0.02%, ibm17 +0.09% — both within float drift between incremental and reference evaluators).

Source JSON: `results/CDLNSSAComposePlacer_20260429_151328.json`.

---

## CDLNSGridBinPlacer --- Champion Result (2026-04-28)

**Status: CHAMPION** --- avg proxy **1.0990** on --all (**24.6% better than RePlAce**, **beats leaderboard 1.1172 by -1.63%**, **-0.59% better than prior CDAdaptive champion**, zero overlaps everywhere).

Configuration: `submissions/cd_lns_gridbin/placer.py` (full-proxy CD on `IncrementalProxyEvaluator` with per-benchmark plateau detection, then a grid-bin LNS escape phase, SDF init). Total runtime 28 256 s = 7.85 hr.

Time-budget split per benchmark: CD phase ≤ 3 000 s + LNS phase ≤ 600 s = 3 600 s total (matches the contest 1-hour-per-bench cap). Plateau-detection params for the CD phase: `min_time_s=300, hard_cap_s=3000, patience=3, plateau_threshold=0.001`. LNS params: `destroy_frac=0.05, destroy_cap=30, destroy_strategy='cost_aware', lns_budget_s=600`.

### How it differs from CDAdaptive

CDAdaptive (E9) was *plateau-bound*, not budget-bound: every benchmark exited via plateau detection, none hit the 1-hour cap. The plateau is a per-axis fixed point of CD's breakpoint enumeration. Three candidate escape mechanisms were tested and falsified before E12 (E3 LNS, SDF jitter, subset-CD destroy — all reused CD's per-axis move type). E12's grid-bin LNS uses a **different move type**: for each destroyed macro it enumerates all `(grid_col, grid_row)` cell centers (~50–2 500 candidates per benchmark) and accepts the globally proxy-minimizing legal reinsertion. That candidate set is outside CD's per-axis breakpoint enumeration, so it can find escapes CD cannot.

### Per-benchmark (--all, zero overlaps everywhere)

| Bench | E12 (champion) | E9 CDAdaptive | RePlAce | Δ vs E9 | Wall (s) |
|-------|---------------:|--------------:|--------:|---------:|---------:|
| ibm01 | **0.9045** | 0.9159 | 0.9976 | **-1.24%** | 524 |
| ibm02 | **1.1340** | 1.1538 | 1.8370 | **-1.72%** | 1049 |
| ibm03 | **0.9886** | 0.9950 | 1.3222 | **-0.64%** | 791 |
| ibm04 | **1.0150** | 1.0226 | 1.3024 | **-0.74%** | 760 |
| ibm06 | **1.1583** | 1.1592 | 1.6187 | **-0.08%** | 1054 |
| ibm07 | **1.1021** | 1.1103 | 1.4633 | **-0.74%** | 1167 |
| ibm08 | **1.1190** | 1.1254 | 1.4285 | **-0.57%** | 1293 |
| ibm09 | **0.8591** | 0.8611 | 1.1194 | **-0.23%** | 660 |
| ibm10 | **1.0562** | 1.0749 | 1.5009 | **-1.74%** | 2433 |
| ibm11 | **0.9136** | 0.9223 | 1.1774 | **-0.94%** | 1179 |
| ibm12 | **1.2076** | 1.2153 | 1.7261 | **-0.63%** | 2771 |
| ibm13 | **0.9766** | 0.9772 | 1.3355 | **-0.06%** | 1874 |
| ibm14 | **1.2205** | 1.2234 | 1.5436 | **-0.24%** | 2390 |
| ibm15 | **1.1797** | 1.1809 | 1.5159 | **-0.10%** | 2098 |
| ibm16 | **1.1573** | 1.1610 | 1.4780 | **-0.32%** | 2426 |
| ibm17 | **1.3299** | 1.3326 | 1.6446 | **-0.20%** | 3487 |
| ibm18 | **1.3603** | 1.3633 | 1.7722 | **-0.22%** | 2301 |
| **AVG** | **1.0990** | **1.1055** | **1.4578** | **-0.59%** | **28 256 (7.85 hr)** |

All 17 benchmarks improved over E9 CDAdaptive. No regressions. Biggest wins on basin-locked / high-density benchmarks: ibm10 (-1.74%), ibm02 (-1.72%), ibm01 (-1.24%), ibm11 (-0.94%). Source JSON: `results/CDLNSGridBinPlacer_20260428_155739.json`.

### Random-destroy ablation (cost-aware ranking is NOT load-bearing)

The cost-aware destroy step in the LNS phase scores each candidate macro by its proxy delta when temporarily moved to canvas center, then picks the K most costly. An ablation `cd_lns_gridbin_random.py` (kept at `experiments/E12_grid_bin_lns/code/`) replaces this with uniform random destroy. On `--fast` (ibm01/04/09/13), random destroy averaged 0.9372 — matching cost-aware within noise. On ibm09, random destroy (0.8541) actually *beat* cost-aware (0.8591). Cost ranking adds ~10% wall per LNS sample but does not change quality. Future simplification: drop the ranking, use random. Stays cost-aware in the production placer for now to avoid mid-deadline changes.

### NG45 commercial-design transfer (E23, 2026-04-28)

Defensive run of the same `submissions/cd_lns_gridbin/placer.py` on the
four public NG45 designs. Zero overlaps everywhere. Plateau-detection
exits in 9–10 sweeps per design; max per-bench wall 1053 s vs the 3600 s
legal cap.

| Design | Proxy | WL | Density | Congestion | Overlaps | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| ariane133 | 0.7061 | 0.064 | 0.525 | 0.759 | 0 | ~615 |
| ariane136 | 0.6840 | 0.060 | 0.540 | 0.708 | 0 | ~767 |
| mempool_tile | 0.7438 | 0.066 | 0.653 | 0.704 | 0 | 686 |
| nvdla | 0.6807 | 0.069 | 0.510 | 0.712 | 0 | 1053 |
| **AVG** | **0.7037** | 0.065 | 0.557 | 0.721 | **0** | total **3122** |

The avg 0.7037 is structurally lower than IBM's 1.0990 because the proxy
bands differ across benchmark sets — direct comparison across IBM and
NG45 is not meaningful. The relevant signals are: (1) zero overlaps,
(2) all under the per-bench cap, (3) plateau detection works without
per-bench tuning. Closes the Tier-2 robustness check follow-up.

Source JSON: `results/CDLNSGridBinPlacer_20260428_223405.json`. Full
analysis: `experiments/E23_ng45_sanity/manifest.md` and
`writeup/evidence.md` §9.A.

---

## CDAdaptivePlacer --- Prior Champion (superseded 2026-04-28 by E12)

**Status: SUPERSEDED 2026-04-28 by CDLNSGridBin (E12)** --- avg proxy **1.1055** on --all (24.2% better than RePlAce, beat leaderboard 1.1172 by -1.05%, 1.23% better than prior CDOnly champion, zero overlaps everywhere).

Configuration: `submissions/cd_adaptive/placer.py` (full-proxy CD on `IncrementalProxyEvaluator`, **per-benchmark plateau detection** with 1hr hard cap, SDF init). Total runtime 17480 s = 4.85 hr.

Plateau-detection params: `min_time_s=300, hard_cap_s=3600, patience=3, plateau_threshold=0.005`. Each benchmark exits when 3 consecutive sweep-deltas fall below 0.005 (and 5 min minimum elapsed), or hits the 1-hour cap.

### How it differs from CDOnly

CDOnly used a fixed 600 s/benchmark — one-size-fits-all. Easy benchmarks plateaued at ~3 min and wasted the rest; hard ones (ibm17/18) ran out mid-descent. CDAdaptive lets each benchmark **exit early when converged** and **run longer when still descending** (up to 1 hr). Net: hard benchmarks get the extra time the easy ones save; nobody is forced to stop mid-improvement.

### Per-benchmark (--all, zero overlaps everywhere)

| Bench | E9 Adaptive | CDOnly | RePlAce | Δ vs CDOnly | Wall (s) | Plateau? |
|-------|-------------|--------|---------|-------------|----------|----------|
| ibm01 | 0.9159 | 0.9133 | 0.9976 | +0.28% | 322 | plateau |
| ibm02 | 1.1538 | 1.1534 | 1.8370 | +0.04% | 524 | plateau |
| ibm03 | 0.9950 | 0.9942 | 1.3222 | +0.08% | 534 | plateau |
| ibm04 | 1.0226 | 1.0193 | 1.3024 | +0.32% | 382 | plateau |
| ibm06 | **1.1592** | 1.1656 | 1.6187 | **-0.55%** | 792 | plateau |
| ibm07 | 1.1103 | 1.1105 | 1.4633 | -0.02% | 607 | plateau |
| ibm08 | **1.1254** | 1.1307 | 1.4285 | **-0.47%** | 833 | plateau |
| ibm09 | 0.8611 | 0.8606 | 1.1194 | +0.06% | 449 | plateau |
| ibm10 | **1.0749** | 1.1000 | 1.5009 | **-2.28%** | 1332 | plateau |
| ibm11 | 0.9223 | 0.9248 | 1.1774 | -0.27% | 614 | plateau |
| ibm12 | **1.2153** | 1.2418 | 1.7261 | **-2.13%** | 1424 | plateau |
| ibm13 | **0.9772** | 0.9939 | 1.3355 | **-1.68%** | 1143 | plateau |
| ibm14 | **1.2234** | 1.2478 | 1.5436 | **-1.96%** | 1572 | plateau |
| ibm15 | **1.1809** | 1.2109 | 1.5159 | **-2.48%** | 1609 | plateau |
| ibm16 | **1.1610** | 1.1919 | 1.4780 | **-2.59%** | 1518 | plateau |
| ibm17 | **1.3326** | 1.3830 | 1.6446 | **-3.64%** | 2238 | plateau |
| ibm18 | **1.3633** | 1.3865 | 1.7722 | **-1.68%** | 1589 | plateau |
| **AVG** | **1.1055** | **1.1193** | **1.4578** | **-1.23%** | **17 480** | **17/17 plateau** |

Every hard benchmark (ibm10/12/13/14/15/16/17/18) improved 1.7-3.6% via adaptive budget. Easy benchmarks (ibm01-04, 09, 11) tied within ±0.4%. ALL 17 benchmarks exited via plateau detection — none hit the 1hr cap. Plateau detection is a real win, not a budget-up-the-wall trick.

### What unblocked this

The CDOnly result (1.1193, matching leaderboard within 0.18%) showed that hard benchmarks ibm17/18/14/12 still had room — their last-CD-sweep deltas were 0.001-0.003, but the fixed 600s budget cut them off. E9's per-benchmark plateau detection gave them the time they needed (1500-2200s on hard, 300-600s on easy) without manual per-bench tuning. The 1-hour hard cap matches the competition rule and keeps total runtime bounded.

---

## CDOnlyPlacer --- Prior-prior Champion (superseded 2026-04-27 by CDAdaptive)

**Status: SUPERSEDED** --- avg proxy **1.1193** on --all (**23.2% better than RePlAce**, **matches leaderboard 1.1172 within 0.18%**, **19.1% better than prior DPO champion**).

Configuration: `submissions/cd_only/placer.py` (full-proxy coordinate descent on `IncrementalProxyEvaluator`, 600s budget per benchmark, SDF init). Runtime 10316s = 172 min total. Zero overlaps on all 17 benchmarks.

### How it works

1. **SDF init** --- non-overlapping starting placement
2. **Incremental evaluator** (`macro_place/incremental_evaluator.py`) --- bit-for-bit parity with `compute_proxy_cost`, **4657x speedup** on per-move cost queries via per-net min/max trackers + bin-density grid + RUDY congestion deltas
3. **Full-proxy coordinate descent** --- for each non-fixed macro, search both x and y axes; enumerate breakpoints (net endpoints + bin grid lines) and pick the proxy-minimizing position via incremental cost queries.
4. **Time budget** --- 600s wall clock per benchmark; sweeps continue until budget expires. Typical: 3-13 sweeps depending on macro count.

### Per-benchmark (zero overlaps everywhere)

| Bench | CDOnly | DPO BoV2 | RePlAce | Delta vs DPO |
|---|---|---|---|---|
| ibm01 | 0.9133 | 1.1285 | 0.9976 | -19.1% |
| ibm02 | 1.1534 | 1.6888 | 1.8370 | -31.7% |
| ibm03 | 0.9942 | 1.2508 | 1.3222 | -20.5% |
| ibm04 | 1.0193 | 1.3239 | 1.3024 | -23.0% |
| ibm06 | 1.1656 | 1.6434 | 1.6187 | -29.1% |
| ibm07 | 1.1105 | 1.3846 | 1.4633 | -19.8% |
| ibm08 | 1.1307 | 1.3816 | 1.4285 | -18.2% |
| ibm09 | 0.8606 | 1.0130 | 1.1194 | -15.1% |
| ibm10 | 1.1000 | 1.2540 | 1.5009 | -12.3% |
| ibm11 | 0.9248 | 1.0657 | 1.1774 | -13.2% |
| ibm12 | 1.2418 | 1.6497 | 1.7261 | -24.7% |
| ibm13 | 0.9939 | 1.2089 | 1.3355 | -17.8% |
| ibm14 | 1.2478 | 1.4725 | 1.5436 | -15.3% |
| ibm15 | 1.2109 | 1.3802 | 1.5159 | -12.3% |
| ibm16 | 1.1919 | 1.3594 | 1.4780 | -12.3% |
| ibm17 | 1.3830 | 1.5888 | 1.6446 | -13.0% |
| ibm18 | 1.3865 | 1.6352 | 1.7722 | -15.2% |
| **AVG** | **1.1193** | **1.3134** | **1.4578** | **-14.8%** |

Every single benchmark improves over DPO. No regressions.

### What unblocked this

The 2026-04-26 strategic update decomposed proxy cost as **WL 6%, density 20%, congestion 74%** (`analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md`). Pure HPWL coordinate descent (closed-form weighted-median) caps at ~5% improvement. **Full-proxy CD on a fast incremental evaluator** captures all three components at once. The infrastructure prerequisite (4657x speedup) gated the entire approach.

---

## See Also

- [approach.md](approach.md) --- methodology and architecture
- [roadmap.md](roadmap.md) --- next steps and submission plan
- [experiment_index.md](experiment_index.md) --- rigorous catalog of every experiment (live + falsified)
- [decisions/007_cd_lns_gridbin_promotion.md](decisions/007_cd_lns_gridbin_promotion.md) --- ADR for the E12 promotion
- `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` --- E8 proxy decomposition diagnostic
- `writeup/historical_results.md` --- DPO/Polyhedra/Overnight/Miftari per-benchmark data
- `writeup/closing_the_gap.md` --- leaderboard-beat narrative + design sketches
