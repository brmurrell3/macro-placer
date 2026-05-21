# Experiment Index

Catalog of every experiment run during the Partcl/HRT Macro Placement
Challenge 2026 — failures alongside wins. Two eras:
**§Post-E84 wave (2026-05-11 → 2026-05-16)** lists the recent
experiments and submission-day candidates. The historical body covers
E1 → E84.

Last updated: 2026-05-16

## Post-E84 wave (2026-05-11 → 2026-05-16)

Body tables below cover E1-E84. The post-E84 wave was driven by two
parallel paths (per `CLAUDE.md` §TWO-PATH WORK PLAN): PATH A cascade
speedup, PATH B DREAMPlace exploration; an unplanned PATH C (E97-E101
Lévy/portfolio/cong-grad probes) ran in parallel.

Verified candidates (Tier-1):

| Placer | IBM `--all` | NG45 `--ng45` | Notes |
|---|---:|---:|---|
| `cd_lns_sa_cascade_dp_lane/placer.py` | **1.06650** | **0.68086** | E91 hybrid: cascade + DP-polished as third init lane; verified 2026-05-14. |
| `cd_lns_sa_cascade/placer_adaptive.py` | **1.07820** | **0.68102** | PATH A post-A1 (LNS-delta + commit() accelerations); verified 2026-05-16 aws-cpu. |

Per-experiment summary (one-liners; full manifests pending for some):

| ID | Hypothesis | Best | Verdict | Date |
|---|---|---|---|---|
| E76-E83 | Wall-safe E74 variants | E83: 1.0859 --all | E83 marginal (5/17 within 1-min cap); not promoted | 2026-05-06 |
| E84 | Cascading saddle escape (iterate E74 saddle until min/no-improvement) | 1.0612 --all (uncapped) | **Verified canonical**; wall-safe at `cd_lns_sa_cascade/placer.py` | 2026-05-11 |
| E91 | DP-full-polish autopsy | hybrid IBM 1.0665 | **Falsifies the "DP basin structurally inferior" autopsy** — DP+full-polish wins ibm12 −13.3 %, ibm17 −10.2 %, ibm14 −3.8 % vs cascade-capped; ibm10 +1.6 % | 2026-05-12 |
| E92 | Diff-trace-route RUDY (rebuild for hard benches) | n/a | Blocked — smooth-RUDY/canonical mismatch 3-4× on hard benches; ~2-3 days dev | 2026-05-13 |
| E93 | DP top-K density | landed in DP-lane | Pipeline component | 2026-05-13 |
| E94 | DP canonical full | landed in DP-lane | Pipeline component | 2026-05-13 |
| E95 | Differentiable proxy v2 (calibrated LSE-HPWL+Gaussian+RUDY) | n/a | Falsified — AdamW destroyed cascade basin step 1; ρ=0.929 calibration necessary not sufficient | 2026-05-13 |
| E96 | Multi-config DP K=4 basin selector | partial | Verified WIN on ibm10 (−2.0 %) and ibm14 (−7.7 %); DP basin proxy bimodal | 2026-05-13 |
| E97 | Lévy heavy-tail ε saddle escape | +0.13-0.23 % on 3 H2H | **Shipped into dual_levy submission**; portfolio multi-eigvec validated | 2026-05-13 |
| E98 | Congestion-gradient hybrid | n/a | Falsified — 3rd smooth-RUDY mismatch on hard benches | 2026-05-13 |
| E99 | Tabu cascade | marginal | Partial | 2026-05-14 |
| E100 | Weight-portfolio saddle (cong-only Hessian softer eigvec) | +0.11 % extra ibm03 | Validated over single-eigvec Lévy | 2026-05-14 |
| E101 | Lévy curvature-adaptive | partial | Submission variant at `cascade_levy_curvadapt/` | 2026-05-14 |
| E102 | GPU Lévy LBFGS | blocked | GPU box not provisioned; variant at `cascade_xplace_levy/` | 2026-05-14 |
| E103 | Diff-trace-route RUDY rebuild | blocked | Same gap as E92 | 2026-05-14 |
| E104 | Worse-init saddle (probe whether bad init reveals more saddles) | partial | Probe; not promotion-class | 2026-05-14 |
| E107 | Periphery bias init | NEW BEST ariane133 **0.65212** (−1.8 % vs E74 0.6641) | 12-test multi-seed; 10-bench sweep table | 2026-05-16 |
| E110 | Smooth global placer (Adam on DiffProxyV2 + greedy_legalize) | partial | ibm01 raw 0.91, +CD60s 0.86 (vs cascade 0.85); lane-4 fallback | 2026-05-18 |
| E111 | Per-net-trace differentiable congestion | NEW HARD-BENCH win ibm17 +CD60s **1.246** (−24.9 % vs E110 bbox-uniform 1.66) | Drop-in replacement for `_rudy_congestion`; matches canonical ±15-25 % | 2026-05-19 |
| E113 | Xplace-recipe optimizer + margin overlap penalty | NEW best ibm01 +CD60s **0.840** (vs cascade 0.85, vs V3 0.846); --fast 4-bench mean **0.866** | Nesterov-BB falsified (worse than Adam); margin + multi-stage Adam wins | 2026-05-19 |
| E115 | Triton kernels / per-step proxy acceleration | marginal | per-step ibm17 CUDA 267 ms → 16 ms (16.8×) via `index_select` fix; no champion impact (smooth-global track) | 2026-05-20 |
| Xplace integration | Bookshelf converters + runner for 21 benches | blocked | GPU quota approval pending; targets Carrotato 0.967 mechanism | 2026-05-16 |
| Tier-2 ORFS | Per-design ship-with vs ship-without `MACRO_PLACEMENT_TCL` | partial | ariane133 ship-without (auto wins by 1.2 ns); ariane136 ship-with (cascade wins by 0.45 ns); mempool/nvdla untested | 2026-05-15/16 |

Sources:
[`handoffs/2026-05-14_champion_found_dp_lane.md`](handoffs/2026-05-14_champion_found_dp_lane.md),
[`handoffs/2026-05-16_morning_handoff.md`](handoffs/2026-05-16_morning_handoff.md),
[`handoffs/2026-05-16_tier2_orfs_findings.md`](handoffs/2026-05-16_tier2_orfs_findings.md).
Per-experiment manifests exist for E91-E107 directories but not all
have populated `status:` fields yet.

---

## (Historical body — 2026-05-06)

A rigorous catalog of every experiment run during the Partcl/HRT Macro Placement
Challenge 2026. Includes the failures alongside the wins — both are part of the
story for the writeup.

Numbers below come from `results/experiment_log.jsonl` (single source of truth).
Each row's `best` is the minimum `avg_proxy_cost` ever logged for that
hypothesis across all runs (`fast`, `all`, or `single` benchmark).

---

## Champion lineage

| Era | Champion | Best Avg (--all) | Date | Notes |
|---|---|---|---|---|
| Pre-history | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Polyhedra | `phase3_50s` (PolyhedraNavigation) | 1.4921 | 2026-04-15 | At ceiling; replaced by DPO |
| DPO v1 | `dpo_v1` | 1.4255 | 2026-04-23 | First to beat RePlAce |
| DPO v2 | `dpo_v2_moresteps` / `v2_steps_restore` | 1.3888-1.4107 | 2026-04-26 | More gradient steps |
| DPO best-of | `best_of_v2` | 1.3834 | 2026-04-26 | Best-of(SDF, DPO) per benchmark |
| CD-only | `cd_only_all_10min` (CDOnly) | 1.1193 | 2026-04-27 03:50 | Matches leaderboard 1.1172 within 0.18% |
| CD-adaptive | `e9_adaptive_all` (CDAdaptive) | 1.1055 | 2026-04-27 13:22 | Beat leaderboard by -1.05%; plateau-bound (every bench exited via plateau) |
| CD + grid-bin LNS | `e12_gridbin_lns` (CDLNSGridBin) | 1.0990 | 2026-04-28 15:57 | beats leaderboard 1.1172 by -1.63%; ADR-007. Superseded by E48 2026-05-02. |
| Best-of-{E25, E41} hybrid | `e48_hybrid_e25_e41` (CDLNSSAHybridPlacer) | **1.08151** | 2026-05-02 | beats leaderboard by -3.21 %; ADR-011 *Accepted*. Superseded by E74 2026-05-05. |
| **E48 hybrid + Hessian saddle escape** | **`e74_hessian_saddle` (CDLNSSAHessianPlacer)** | **1.0666** | **2026-05-05** | **CURRENT CHAMPION; beats leaderboard by -4.53 %, beats E48 by -1.38 %, ADR-012 *Accepted*. Wall ~96 min/bench on hard benches — over 60-min hard timeout. Subject of §Derisk wall-budget work in 2026-05-05/06 wave (E79-E83).** |

### Champion candidate (verified, not promoted)

| Era | Candidate | Best Avg (--all) | Date | Notes |
|---|---|---|---|---|
| **DPO + CD + LNS + SA-v2 + K-joint** | **`e41_dpo_kjoint_all_postfix` (CDLNSSADPOKJointPlacer)** | **1.0848** | **2026-04-30 09:13** | **STRONGEST VERIFIED CANDIDATE.** −1.29 % vs E12, −2.90 % vs leaderboard, −0.97 % vs E25 candidate, −0.46 % vs E18 candidate. NG45 0.69022 (−1.91 % vs E12, beats E18 NG45 by −0.25 %). 14/17 IBM wins; hard-plateau wins ibm11 −4.06 %, ibm14 −1.73 %, ibm15 −1.24 % vs E25. ADR-010 *Proposed*. |
| DPO + CD + LNS + SA-v2 | `e18_dpo_init_all` (CDLNSSADPOInit) | 1.08979 | 2026-04-30 02:32 | Prior candidate (now superseded by E41). −0.84 % vs E12, −2.45 % vs leaderboard. NG45 0.69193 (4/4 wins). ADR-009 *Proposed*. Mark *Superseded* on ADR-010 accept. |
| CD + LNS + SA-v2 | `e25_lns_sa_compose` (CDLNSSA) | 1.0954 | 2026-04-29 15:13 | Older candidate. −0.33% vs E12, −1.95% vs leaderboard. ADR-008 *Proposed*. Mark *Superseded* on ADR-010 accept. |

Lineage: each champion replaced the prior by a structural change, not parameter
tuning.

---

## Live experiments (current)

| ID | Hypothesis | Files | Best | Mode | Status |
|---|---|---|---|---|---|
| E1 | Incremental proxy evaluator | `macro_place/incremental_evaluator.py`, `test/test_incremental_evaluator.py`, `scripts/bench_incremental.py` | 4657× speedup, bit-for-bit parity | infrastructure | **VALIDATED** — load-bearing for E2/E9/E12/E25 |
| E2 | Full-proxy CD on ibm10 (40 min) | `scripts/cd_ibm10_diagnostic.py` | 1.0632 (ibm10 single) | single | **VALIDATED** — sparked CDOnly |
| E9 | Adaptive per-bench plateau detection | `submissions/cd_adaptive/placer.py` | 1.1055 (--all) | all | **prior champion²** (superseded 2026-04-28 by E12) |
| E12 | CD plateau + grid-bin LNS overlay | `submissions/cd_lns_gridbin/placer.py`, `experiments/E12_grid_bin_lns/` | **1.0990 (--all)** | all | **CHAMPION** (graduated 2026-04-28; ADR-007) |
| E25 | CD plateau + grid-bin LNS + SA-v2 polish (compositional) | `submissions/cd_lns_sa/placer.py`, `experiments/E25_lns_sa_compose/` | 1.0954 (--all) | all | **champion candidate** (verified 2026-04-29; ADR-008 *Proposed*; not promoted) |
| E13 | Congestion-region spatial-cluster LNS (destroy K macros sharing a hot cell instead of cost-ranked individuals) | `experiments/E13_congestion_lns/code/cd_lns_congestion.py`, manifest | 0.9384 (--fast); -0.45 % vs E16 baseline 0.9426 | fast | **MARGINAL 2026-04-29** — passes first kill gate, fails generalization-check threshold (lift 0.45 % < 0.5 %). Mechanism works on ibm01 (0.94 % LNS lift) but doesn't generalize to ibm04/ibm09/ibm13. +0.13 % worse than E12-random-destroy ablation. Did not queue `--all`. |
| ~~E14~~ | ~~SA polish on CD output (Metropolis acceptance over per-axis breakpoints)~~ | `experiments/E14_sa_polish/code/cd_sa_polish.py`, manifest | 0.9962 (--fast); +5.7 % vs E16 baseline 0.9425 | fast | **FALSIFIED 2026-04-29** — SA wandered proxy upward (CD plateau 0.971 → SA-final 1.027 on ibm13). Two issues: (1) no best-so-far tracking, (2) T₀ = 0.01 too high → 50/50 random walk. Kill gate hit on every benchmark. |
| E23 | NG45 sanity test on E12 champion (defensive — fills ADR-007's missing NG45 datapoint) | `experiments/E23_ng45_sanity/`, run.log | **0.7037 (--ng45 4 designs)** | ng45 | **VALIDATED 2026-04-28** — zero overlaps; max per-bench wall 1053 s vs 3600 s cap; plateau detection transfers to commercial designs |
| E24 | SA polish v2 — principled retest of E14 with best-so-far tracking + T₀ = 5e-4 (fair test of "Metropolis escapes CD basin") | `experiments/E24_sa_polish_v2/code/cd_sa_polish_v2.py`, manifest | 0.9366 (--fast); +0.63 % vs E16 baseline 0.9426 | fast | **MARGINAL 2026-04-29** — passes gen-check; ties E12 random-destroy ablation 0.9372; ibm13 saw zero SA lift (best == CD plateau). Compositionality test E25 launched. |
| E25 | Compositional CD + LNS + SA-v2 → **champion candidate** | `submissions/cd_lns_sa/placer.py`, `experiments/E25_lns_sa_compose/` | **1.0954 (--all)**; -0.33 % vs E12 1.0990; -1.95 % vs leaderboard 1.1172 | fast→all | **VERIFIED 2026-04-29; CHAMPION CANDIDATE** (ADR-008 *Proposed*; not promoted). Wins on 11/17, ties on 4/17, sub-noise regression on 2/17. Wall 10.33 hr (+2.5 hr vs E12, well within 17-hr cap). |
| ~~E26~~ | ~~Longer SA budget (LNS 300 s, SA 900 s) — tests if SA's "best at t≈599s" had real headroom~~ | `experiments/E26_longer_sa/code/cd_lns_sa_e26.py`, manifest | 0.9336 (--fast); **tied with E25 0.9336** | fast | **FALSIFIED 2026-04-29** — extending SA produces only float-drift-scale fluctuation. The "SA still improving" was best-tracking noise, not headroom. +20 % wall for zero quality gain. 600 s SA saturation is real. |

### Overnight wave 2026-04-29 → 30 (post-E25 candidate exploration)

Six hypotheses in three categories: orthogonal *inits* (E17, E18), orthogonal *move types* (E32, E39), within-basin *ensemble* (E40), and a *combo* (E41). Plus a basin-diagnostic probe (E27).

| ID | Hypothesis | Files | Best | Mode | Status |
|---|---|---|---|---|---|
| ~~E17~~ | ~~Random-uniform init + CD baseline (init-class probe)~~ | `experiments/E17_random_init/code/cd_lns_sa_random_init.py`, manifest | far worse than SDF on every bench (E27 confirms uniform basin = 1.79 mean on ibm14, vs SDF 1.26) | fast | **FALSIFIED 2026-04-30** — random init lands in a separate, much worse basin. Confirms SDF init is load-bearing. |
| **E18** | DPO best_of_v2 init → CD + LNS + SA-v2 polish (alternative-init probe) | `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py`, manifest | **1.08979 (--all)**; -0.51 % vs E25 1.0954; -0.84 % vs E12 1.0990; **0.69193 (--ng45)**, -1.67 % vs E12 0.7037 | fast→all+ng45 | **CHAMPION CANDIDATE 2026-04-30** — DPO basin transfers strongly through E25 polish. 11/17 IBM wins (zero overlaps); 4/4 NG45 wins. Awaiting human promotion decision. |
| E27 | Persistence-homology basin diagnostic (single-basin? multi-basin? — gates post-E25 line) | `experiments/E27_basin_persistence/code/{run_trajectory.py,analyze_persistence.py}`, manifest | 16/44 trajectories (DPO + sdf_jitter inits crashed); SDF basin tight across seeds (std 0.0015-0.0037); greedy/uniform inits separate worse basins | diagnostic | **MARGINAL 2026-04-30** — analyzer reports "ambiguous" due to incomplete data, BUT empirical signal from E18 (DPO basin -0.51 % deeper than SDF) is the actual basin diagnostic. Verdict: multi-basin across init classes (motivates E18 promotion); single-basin within SDF init class (motivates skipping E40 multi-seed). |
| ~~E32~~ | ~~SAM-CD (sharpness-aware proxy with K=4 perturbations) — flatter basins → better OOD~~ | `experiments/E32_sam_cd/code/cd_lns_sa_sam.py`, manifest | 0.98501 (--fast); +5.5 % vs E25 fast 0.9336 | fast | **FALSIFIED 2026-04-30** — K=4 perturbations multiply CD eval cost ~5×; CD hits 2400 s cap before plateau on every bench. ibm13 +10.6 % vs E25. Kill gate fired. |
| E39 | K-macro joint LNS (K=3, top_N=5, brute-force N^K=125 enumeration) — joint-move escape | `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`, manifest | initial 0.93070 (--fast); **post-fix 0.92835 (--fast)** -0.56 % vs E25 fast 0.9336; partial --all 6/17 valid then ibm07 crash (pre-fix); 0.70126 (--ng45 4 designs); single-bench ibm07 post-fix 1.09857 VALID | fast→all+ng45 | **MARGINAL 2026-04-30; ibm07 crash bug fixed and verified 2026-04-30 04:56** — K-joint mechanism extracts 12-68 commits/bench post-CD-LNS-SA. ibm07 crash root-caused to eps=1e-4 in wrong direction in `_is_legal_2d_excluded` and `_ktuples_pairwise_legal` (tolerated up-to-1e-4 overlap; `compute_overlap_metrics` flagged it). Fix: eps=1e-9 in opposite direction (require strict separation) + post-commit `compute_overlap_metrics` defensive revert. **Post-fix --fast 0.92835 BETTER than pre-fix 0.93070 by -0.25 %** (cleaner K-joint enumeration; no near-overlap candidates). **superseded_by: E41 (DPO+K-joint composes more strongly).** |
| ~~E40~~ | ~~Multi-SA-seed best-of-4 from same post-LNS state (within-basin ensemble)~~ | `experiments/E40_multi_sa_seed/code/cd_lns_sa_multi_seed.py`, manifest | 0.93295 (--fast); -0.07 % vs E25 fast 0.9336 | fast | **MARGINAL/SKIPPED-ALL 2026-04-30** — fork 1 (seed=42, same as E25) usually best. Multi-seed lift doesn't compound; not worth ~14 hr `--all`. Confirms basin is locked at SA-v2 T₀=5e-4 (consistent with E27 single-basin-within-SDF-init signal). |
| **E41** | DPO init + K-joint LNS combo (E18 ⊕ E39) — composes basin shift + joint-move escape | `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`, manifest | **0.92178 (--fast)**, -1.27 % vs E25 fast 0.9336; **0.69022 (--ng45)**, -1.91 % vs E12; **1.0848 (--all)**, -1.29 % vs E12 1.0990, -0.46 % vs E18 1.08979, -0.97 % vs E25 1.0954, -2.90 % vs leaderboard 1.1172 | fast→ng45→all | **STRONGEST VERIFIED CANDIDATE 2026-04-30** — 14/17 IBM wins (3 losses: ibm01 +2.46 %, ibm07 +1.32 %, ibm06 +1.26 %); hard-plateau wins ibm11 -4.06 %, ibm10 -3.47 %, ibm13 -2.95 %, ibm03 -2.88 %, ibm04 -2.26 %, ibm14 -1.73 %, ibm15 -1.24 % vs E25. K-joint commits 12-58 K-tuples per bench (12-95 on hard plateau benches); composition with DPO basin escapes the multi-mechanism floor. Wall 13.58 hr `--jobs 4`. ADR-010 *Proposed*. |

### K-joint variant exploration 2026-04-30 (post-E41 verified)

Three orthogonal K-joint variants tested to determine if E41 K=3 with 600 s budget can be improved further by tuning K, budget, or K-tuple selection. Result: E41 is the saturation floor; further K-joint variants don't lift meaningfully.

| ID | Hypothesis | Files | Best | Mode | Status |
|---|---|---|---|---|---|
| ~~E42~~ | ~~K=4 K-joint variant — tests if hard plateau benches need 4-coupled simultaneous moves (vs E41's K=3)~~ | `experiments/E42_kjoint_k4/code/cd_lns_sa_dpo_kjoint_k4.py`, manifest | 0.91859 (--fast); -0.35 % vs E41 fast 0.92178; **0.6960 (--ng45); +0.84 % REGRESSION vs E41 ng45 0.69022** | fast→ng45 | **FALSIFIED 2026-04-30 13:25** — `--fast` marginal (+0.35 % at noise floor); NG45 catastrophic on ariane133 (+3.57 % vs E41). K=4's per-K-tuple cost (5×) starves the budget at the wrong margin — the design where K=3 finds genuine joint structure trades enumeration depth for breadth and loses. **K=4 is IBM-fast-overfit.** |
| ~~E43~~ | ~~Longer K-joint budget (1200 s vs E41's 600 s) — tests if K=3 K-joint is budget-bound on larger benches~~ | `experiments/E43_kjoint_longer/code/cd_lns_sa_dpo_kjoint_longer.py`, manifest | 0.91867 (--fast); -0.33 % vs E41 fast 0.92178 (marginal). **0.7009 (--ng45); +1.55 % REGRESSION vs E41** — falsified | fast→ng45 | **FALSIFIED 2026-04-30 17:49** — `--fast` was at gen-check threshold (concentrated on ibm13 alone; smaller benches saturated K-joint within 600 s). NG45 verification: avg +1.55 % vs E41, with **ariane133 +4.10 % loss** and **ariane136 +2.48 % loss**. Same overfit class as E42 K=4 NG45. **K-joint is saturation-bound on small benches, marginally budget-bound on ibm13, and OVERFIT-AT-1200s on NG45 ariane designs.** Confirms K=3, 600 s, netlist-adjacency is the right K-joint operating point. |
| ~~E44~~ | ~~Spatial-adjacency K-tuple selection (instead of E41's netlist-adjacency) — tests if spatially clustered K-tuples extract more lift~~ | `experiments/E44_kjoint_spatial/code/cd_lns_sa_dpo_kjoint_spatial.py`, manifest | 0.9276 (--fast); +0.63 % vs E41 fast 0.92178; **kill gate fired** | fast | **FALSIFIED 2026-04-30 15:52** — `--fast` 0.9276 above kill gate (E41 +0.5 % = 0.9264). Spatial K-tuples find 40-91 commits per bench (vs E41's 12-58) but they are MYOPIC: locally tight cluster moves rather than netlist-structured. ibm01 +0.70 %, ibm04 +2.11 % (badly hurt). **Netlist-adjacency K-tuple selection is load-bearing** for the K-joint mechanism — picks tuples that share small nets, exactly the structural coupling that creates the multi-mechanism plateau. |

### Post-E41 wave 2026-05-01 → 02 (compositional + GPU + congestion-direction)

| ID | Hypothesis | Files | Best | Mode | Status |
|---|---|---|---|---|---|
| **E48** | Per-bench best-of-{E25, E41} hybrid (algorithmic best-of, not per-bench tuning) | `experiments/E48_hybrid_e25_e41/code/cd_lns_sa_hybrid.py`, manifest | **0.9202 (--fast); 1.08151 (--all); 0.6922 (--ng45)** | fast→all+ng45 | **STRONGEST VERIFIED CANDIDATE 2026-05-01** — −0.30 % vs E41 1.0848, −1.59 % vs E12 1.0990, −3.21 % vs leaderboard. Per-bench: E25 wins 5/17 (ibm01/06/07/17/18), E41 wins 12/17. Theoretical bound 1.08121 realized within float-drift. ADR-011 *Proposed*. |
| ~~E45~~ | ~~Multigrid hierarchical CD~~ | `experiments/E45_multigrid/`, manifest | (falsified per recent commit) | fast | **FALSIFIED 2026-04-30/05-01** — multigrid hierarchical CD didn't compose with K-joint mechanism. |
| E51 | SDF-K-joint (no DPO) — tests how much of E41 lift is K-joint mechanism alone | `experiments/E51_sdf_kjoint/`, manifest | (marginal per recent commit) | fast | **MARGINAL 2026-05-01** — K-joint mechanism is ~24 % of E41's lift; rest is DPO basin. Confirms compositional, not redundant. |
| E52 | E41 with seed=1 (DPO seed-noise characterization) | `experiments/E52_e41_seed1/`, manifest | (DPO seed-noise on ariane133) | fast | **DIAGNOSTIC 2026-05-01** — DPO seed-noise observed on ariane133 (~0.4 % swing). Motivates multi-seed verification before promoting any candidate. |
| E53m | Multi-seed hybrid: best-of-{E25, E41 seed=42, E41 seed=1} per bench (extends E48 with E41 seed=1 lane) | `experiments/E53_multiseed_hybrid/code/cd_lns_sa_multiseed_hybrid.py`, manifest | **0.91128 (--fast); 1.08128 (--all, −0.02 % vs E48); 0.6938 (--ng45, +0.23 % vs E48)** | fast→all→ng45 | **MARGINAL 2026-05-02** — --fast big lift (−0.97 % vs E48) is sample-size outlier; --all 1.08128 essentially tied with E48 1.08151; --ng45 0.6938 slight regression (+0.23 %; ariane133 0.68778 vs E48 0.6861 +0.24 %). NOT a falsification — per-bench best-of always picks E48's lane on NG45 anyway, no harm done. But also no lift. Multi-seed within DPO is dead-end for breakthrough; --fast was sample-size outlier. Best-of-{E48, E53m} = 1.08106 (--all) and 0.69206 (--ng45) — trivial improvements. |
| ~~E53~~ | ~~GPU DPO basin polish replacing K-joint (multi-restart full-pose Adam on smooth proxy + overlap penalty, MPS device)~~ | `experiments/E53_dpo_basin_eval/code/cd_lns_sa_dpo_basin.py`, manifest | 0.9254 (--fast); +0.36 % vs E41 fast | fast | **FALSIFIED 2026-05-01** — 350 GPU-DPO restarts across 4 benches, **0 accepts**. After full-budget CD-LNS-SA convergence, smooth-proxy gradient cannot find improvements. The −2.16 % lift seen in shortened-budget smoke disappeared at production budgets. Lesson: **GPU DPO basin polish is not additive on top of fully-converged CD-LNS-SA**; mechanism would need to replace CD itself, not polish after it. |
| ~~E54~~ | ~~LNS destroy ranked by direct contribution to abu-top-5 % congested cells (engages E8 6/20/74 % proxy decomposition)~~ | `experiments/E54_congestion_destroy/code/cd_lns_sa_dpo_kjoint_congdestroy.py`, manifest | 0.9222 (--fast tied); 1.08568 (--all +0.39 % vs E48); **0.7022 (--ng45); +1.45 % REGRESSION; ariane133 +5.14 % catastrophic loss** | fast→all+ng45 | **FALSIFIED 2026-05-02 02:53** — IBM-fast lift (ibm04 −1.12 %, ibm13 −0.15 %) does NOT transfer. --all 1.08568 slightly worse than E48 1.08151; --ng45 0.7022 vs E48 0.6922 = +1.45 % with ariane133 0.7214 vs E48 0.6861 = **+5.14 % catastrophic regression**. Same NG45 failure pattern as E42 K=4 (+3.57 %) and E43 longer-K-joint (+4.10 %). **Mechanism-aligned destroy is IBM-aware and NG45-blind on ariane-class designs**. Best-of-{E48, E53_multiseed, E54} --all = 1.08093 (only −0.05 % vs E48 — E54 contributes 2 per-bench wins ibm12/ibm16). Hybrid extension dead. |
| ~~E55~~ | ~~E54 ablation — `include_net_share=False` (direct macro routing only)~~ | `experiments/E55_congdestroy_ablation/`, manifest | dropped before launch | n/a | **DROPPED 2026-05-02** — parent E54 falsified on NG45 before this ablation was worth running. No point testing the net-share term when the destroy direction itself is NG45-blind. |
| ~~E60~~ | ~~Replica exchange / parallel tempering SA-v2 (K=4 chains at T₀ ∈ {5e-4 … 5e-3 or 5e-1})~~ | `experiments/E60_replica_sa/code/replica_sa.py`, manifest | falsified pre-fast (no lift over baseline SA on ibm01 smoke) | smoke | **FALSIFIED 2026-05-02** — wrong attack point. Energy gap between chains too large (Δproxy ≥ 0.05) → swap log-prob ≈ −1000 → 0/21 swaps accept at narrow spacing; wider spacing (×10) gave 27 % accepts but cold chain disrupted by hot-chain noise (took 250 s to recover, vs single-chain SA finding −0.014 in 600 s). **Replica exchange targets "SA stuck at single basin" — but our SA-v2 isn't actually stuck; the bottleneck is BASIN CHOICE upstream (CD/LNS), not SA dynamics.** Pivoted to E61 (basin-choice attack via crossover). |
| E61 | Genetic-algorithm crossover between E25 and E41 outputs (V1 per-macro Bernoulli falsified; V2 spatial-block 2×2 quadrants); polish via CD+LNS+SA-v2 | `experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`, manifest | V1 ibm01: falsified. V2 ibm12 smoke 1.19898 (−0.61 %). V2 --fast 0.91342 (−0.74 %). V2 --ng45 0.69078 (ariane133 0.6760, −1.47 %). **V2 --all 1.08083 (−0.066 % vs E48), 6 wins / 5 losses / 6 ties — wins concentrated on tied-parent benches: ibm12 −0.68 %, ibm14 −0.47 %, ibm15 −0.28 %.** | smoke→fast→ng45→all | **MARGINAL 2026-05-03 07:31** — V1 Bernoulli unrecoverable overlaps. V2 spatial-block produced real but tiny --all lift; misses promotion threshold (−0.3 %) by ~5× but **mechanism verified**: real wins on tied-parent benches (parent gap ≤ 1.7 %), polish reverts to dominant parent on basin-asymmetric benches. **Refines E65 wall finding**: uniform interpolation is blocked, but selective spatial-block swap threads through. Worth keeping in tree as small per-bench complement to E48; not a champion in isolation. (Foreground V2 --all run was killed by Claude at 8/17 benches; parallel agent re-launched and ran to completion overnight at 95871 s = 26.6 hr CPU.) |
| ~~E62~~ | ~~Will's seed v4 init replaces DPO best_of_v2 in E41 backbone — 3rd basin lane candidate~~ | `experiments/E62_will_seed_init/code/cd_lns_sa_will_kjoint.py`, manifest | **0.93347 (--fast); +1.44 % vs E48 fast 0.92024** | fast (kill gate fired) | **FALSIFIED 2026-05-02 16:02** — loses on every fast bench: ibm01 +1.54 %, ibm04 **+3.35 %**, ibm09 +0.99 %, ibm13 +1.31 %. Zero per-bench wins → no contribution to a hybrid extension. **4th falsification** of the "find a new basin via init class" attack (after E17 random, E32 SAM-CD, E53 GPU DPO polish). The SDF+DPO basin pair is not arbitrary — they're specifically near-optimal for the IBM proxy. **Pivot direction**: refining within the existing 2-basin envelope. NG45 not queued (kill gate on fast pre-empted). |

### §Derisk wave 2026-05-05/06 (post-E74 wall-budget exploration)

E74 graduated as champion at 1.0666 `--all`, but its ~96-min wall on
canonical bench (ibm01) and ~135-min wall on hardest benches (ibm13)
exceeds the **1-hour-per-benchmark hard timeout** spelled out in
`README.md`.  This wave attacks the wall-budget problem.

| ID | Hypothesis | Files | Best | Mode | Status |
|---|---|---|---|---|---|
| ~~E77~~ | Sharper Hessian: k=5 eigvecs + finer ε grid {0.1, 0.3, 1.0, 3.0, 10.0} on ibm12-18 | `experiments/E77_sharper_hessian/` | ibm12 1.20980 (≈ tied E74); only eig0 ε=0.1 lifted | single | **MARGINAL 2026-05-05** — k=5 didn't deliver promised +0.2-0.5 % lift; only eig0 with small ε productive. k=2 (E74) already near-optimal. Compute better spent on E80 streaks. |
| ~~E78~~ | Layered E61V2 plateau + Hessian saddle (vs E74's E48-plateau) on ibm08, 14, 16, 17, 18 | `experiments/E78_layered_e61v2_e74/` | ibm08 1.10131 (small lift; saddle didn't help — E61V2 fell back to E41 via crossover failure) | single | **MARGINAL 2026-05-05** — saddle escape failed to lift E61V2 plateau on ibm08 (zero NEW BEST polishes from 6 trials). The win came from E41 fallback after crossover overlap, not the layering hypothesis. |
| ~~E79~~ | Parallel E25⊥E41 (subprocess) + compressed Hessian (k=1, polish 180s) + hardware probe — §Derisk Mitigations #1+#2+#5 | `experiments/E79_hardware_portability/` | --fast avg 0.9131 (probe 1.0); ibm13 wall 89 min (over cap) | fast | **SUPERSEDED by E83 2026-05-06** — parallelism saves ~25 min, compressed Hessian saves ~30 min vs E74, but ibm13 still 78-89 min on Windows (over 60-min cap). Probe-floor bug fixed (was inverted, now `wall/baseline` clamped [1.0, 1.5]). |
| ~~E80~~ | Work-bounded streaks (LNS 5-streak, SA 1000-move) + drop K-joint — §Derisk Mitigations #3+#4 | `experiments/E80_work_bounded_streaks/` | ibm13 0.9586 / 76 min (still over cap; proxy regressed +1.4 % vs E79) | single | **FALSIFIED 2026-05-05** — LNS 5-streak didn't fire on hard benches (LNS hits full 600s budget without 5 consecutive non-improving samples). SA streak saved ~10 min/lane but proxy regressed. Drop K-joint saved 10 min but cost 0.002 lift. Trade-offs canceled. |
| ~~E81~~ | CD-only plateau + Hessian saddle escape (test if saddle is the lift mechanism, not the plateau depth) | `experiments/E81_cd_only_saddle/` | ibm01 0.91065 / 34 min (≈ E48 baseline, saddle lift only -0.00124 vs E79's -0.030) | single | **FALSIFIED 2026-05-05** — saddle DID lift cheap CD plateau, but ~25× less than on E25/E41's deep plateau. Plateau depth materially affects saddle's negative eigenvalues. Cheap plateau forfeits the saddle's value. |
| ~~E82~~ | Hybrid dispatcher: E79 path for `num_hard_macros ≤ 350`, E48 hybrid for larger | `experiments/E82_hybrid_dispatcher/` | n/a — never run on `--all` | n/a | **FALSIFIED 2026-05-05 (rule violation)** — `README.md` forbids "hardcoding solutions for specific benchmarks (must be general algorithm)". Per-bench-property dispatch is functionally equivalent. Successor E83 routes on observed elapsed time instead. |
| **E83** | **Clock-aware single algorithm** — same hyperparameters everywhere; Hessian phase adapts ONLY to remaining wall time (anytime-algorithm design, rule-compliant) | `experiments/E83_clock_aware/` | **--all 1.0859** (-2.80 % vs leaderboard, +0.41 % vs E48, +1.81 % vs E74); 17/17 valid; **17/17 fit 60-min cap on Windows but 5/17 within 1-min margin** | smoke→all | **MARGINAL 2026-05-06** — only candidate with hard wall enforcement. Phase 1+2 cap 37 min (CD 1500s, LNS 360s, SA 360s, K-joint 0); clock-aware Hessian (full ≥ 15 min remaining, minimal ≥ 5 min, skip otherwise). Wall margin too thin for EPYC slowdown (1.2-1.5×). Awaits E83 v2 with tighter budgets, OR explicit wall enforcement inside `_saddle_escape`. |

#### §Derisk supplementary work

| Task | Outcome |
|---|---|
| Throttled-CPU verification (`OMP_NUM_THREADS=1` on E79 ibm09) — §Derisk Mitigation #6 | **2026-05-05**: 1.4× single-thread slowdown vs `--jobs 4` sharing. Proxy stable (0.8275 vs 0.8250); wall 67 min vs 83 min (single-job had less contention). Not a clean EPYC predictor. |
| `plc_client_os.py` scientific-notation parser bug | **Fixed 2026-05-05**. Was crashing on ibm08/12/13 (-1.42109e-15 truncated). Cleared __pycache__ after edit. |
| Hardware-probe direction bug | **Fixed 2026-05-05**. Was `baseline/wall` (more budget on faster HW); now `wall/baseline` clamped [1.0, 1.5] (more budget on slower HW). |
| Tier 2 ORFS verification scoping (Task 18) | **Scoped 2026-05-05** at `analysis/tier2_orfs_scoping/notes.md`. Highlights ≥ 12 μm clearance check as cheapest pre-submission validation. |

#### Submission strategy as of 2026-05-06

The 60-min hard timeout (per `README.md`) means **none of E48/E74/E79 can be
submitted as-is on hard benches** — sequential phases blow the budget.
**E83 is the only candidate with provable cap fit** but its margin is
too thin for the EPYC slowdown.  Active decisions:

* Tighten E83 (v2) budgets — phase 1+2 cap to ~28 min, saddle to ~10 min,
  total ~38 min on M3-equivalent — and re-run `--all` to validate.
* Optional: explore E84 = DREAMPlace plateau (GPU, sub-minute) + Hessian
  saddle escape, leveraging the available RTX 6000 Ada 48GB on EPYC.
  E76 was the original DREAMPlace integration scoping (status: scoping).
* If neither hits cap reliably, ship E83 v2 with whatever proxy it
  delivers (likely ~1.09 `--all`) — still beats public leaderboard 1.1172.

### DPO line (all superseded by CD)

| ID | Hypothesis | Files | Best | Status | Lesson |
|---|---|---|---|---|---|
| dpo_v1 | First DPO version | `submissions/dpo/placer.py` | 1.4255 (--all) | superseded | First to beat RePlAce; topology not optimal |
| dpo_v2_moresteps | More gradient steps | (in placer.py) | 1.4107 (--all) | superseded | Diminishing returns |
| dpo_v3_final | Refined config | (in placer.py) | 1.4246 (--all) | superseded | Tuning hit the basin ceiling |
| best_of_sdf_dpo | Best-of SDF/DPO/seed per benchmark | `submissions/dpo/best_of_placer.py` | 1.4145 (--all) | superseded | Per-bench best-of helps marginally |
| best_of_v2 | Best-of with v2-steps | `submissions/dpo/best_of_v2_placer.py` | 1.3834 (--all) | superseded | DPO ceiling — basin lock |
| best_of_v2_seed{43-46} | Seed sensitivity | `submissions/dpo/best_of_v2_seed{N}.py` | 1.3790-1.3927 (--all) | **falsified (E5)** | All seeds within 1% on --all; ibm02/12 byte-identical |
| ablation_seed{43-46} | Seed sensitivity (DPO only) | `submissions/dpo/ablation_seed{N}.py` | 1.4237-1.4301 (--all) | **falsified** | Seed perturbation alone doesn't escape basin |
| ablation_no_cong | No congestion gradient | `submissions/dpo/ablation_no_cong.py` | 1.5092 (--all) | superseded | Congestion gradient adds 4% |
| ablation_no_dens | No density gradient | `submissions/dpo/ablation_no_dens.py` | 1.7342 (--all) | superseded | Density gradient adds 16% |
| ablation_phase1 | Phase-1 only | `submissions/dpo/ablation_phase1.py` | 1.4605 (--all) | superseded | Multi-phase continuation matters |
| ablation_random_init | Random init instead of SDF | `submissions/dpo/ablation_random_init.py` | 4.9415 (--all) | superseded | DPO needs a good init; random fails projection |
| cw_{050,075,100,125,150} | Congestion-weight sweep | `submissions/dpo/ablation_cw{N}.py` | 1.2081-1.2382 (fast) | superseded | cw=0.50 best on fast; doesn't generalize |
| v2_steps_restore | Restored v2 step count | `submissions/dpo/ablation_v2_steps.py` | 1.1749 (fast), 1.3888 (--all) | superseded | Not better than best_of_v2 |

### Falsified extension hypotheses (E5, E10, E11, E15)

| ID | Hypothesis | Files | Best | Status | Lesson |
|---|---|---|---|---|---|
| **E5** | Massively parallel DPO seeds (best-of-N at scale) | `submissions/dpo/batched_seeds_placer.py` (679L), `batched_seeds_b1.py` | 1.1638 (fast), 1.1698 (B=64 fast) | **FALSIFIED 2026-04-26** | B=64 wall = 8.9× B=1 (RUDY congestion kernel scales 27× on MPS). All 64 seeds collapse to same basin (sigma=0.04·canvas). Best-of-N within a single basin doesn't help. |
| **E15** | Pair-swap (swap two macros sharing ≥1 net) on top of CDAdaptive | `experiments/E15_pair_swap/code/cd_pair_swap.py`, manifest | 0.9414 (--fast) vs E16 baseline 0.9425 | **FALSIFIED 2026-04-28** | v2 finds 21–53 real swaps per benchmark on `--fast` but Δ = −0.0011 vs baseline (below noise). CD's plateau is robust to pair swaps too — connectivity-graph-promising candidates aren't the ones the proxy actually wants moved. Same lesson as SDF jitter and subset-CD destroy: swap targets that look right by graph topology stay inside CD's reachable set. |
| **E10** | Congestion-only refinement on DPO output | `submissions/dpo/congestion_refine_placer.py`, `_e10_sweep_{mild,aggressive}.py` | 1.1636 (fast) `mild`, 1.3788 (--all) | **MARGINAL/FALSIFIED** | Mild config gained 1.0% on fast set; only -0.33% on --all. **ibm02 got WORSE +2.3%** — basin-locked benchmarks regressed. |
| **E11** | DPO from diverse priors (SDF + Will + greedy + random) | `submissions/dpo/diverse_priors_placer.py`, `init_strategies.py`, `e11_sdf_only.py`, `e11_sdf_random.py` | 1.1659 (fast 2-prior), 1.3839 (--all) | **FALSIFIED 2026-04-27** | Fast-set -0.8% didn't scale to --all (FLAT +0.04%). ibm02/ibm12 got WORSE — alternative priors land in deeper basins for high-density benchmarks. |
| **E3 v1** | LNS rip-up-and-reinsert (full-canvas search) | (deleted from active tree in commit 44efd16; preserved at `writeup/archive/submissions/lns.py`, `writeup/archive/submissions/cd_lns_placer.py`) | 1.3846 (ibm17 single) | **FALSIFIED 2026-04-27** | Full-canvas 2244 grid candidates × ~100ms each = 200s+ per macro reinsertion. 1 LNS iteration in 428s. Final ibm17 = 1.3846 vs CDOnly 1.3830 (flat). |
| **E3 v2** | LNS with 5×5 local-window reinsert | (deleted from active tree in commit 44efd16; preserved at `writeup/archive/submissions/lns.py` with `window_radius=5`, and `writeup/archive/submissions/cd_lns_placer.py`) | 1.3824 (ibm17 single) | **FALSIFIED 2026-04-27** | 90× faster (15 iters in 600s) but final ibm17 = 1.3824 vs CDOnly 1.3830 = flat. Cost-based destroy selector saturates after 1-2 accepts; 5×5 window can't move clusters. Single-macro local LNS does NOT escape CD's local minimum. |

### Earlier polyhedra-era experiments (Phases 1-5)

These all topped out around 1.49 — proved the polyhedra-navigation approach
hit a ceiling.

| ID | Hypothesis | Best | Status |
|---|---|---|---|
| `phase3_50s`, `phase3_final` | Polyhedra navigation, 50s budget | 1.4921 (--all) | superseded |
| `phase3_45s`, `phase3_v2`, `phase3_sparse_verify` | LP-related variants | 1.2623-1.2680 (fast) | superseded |
| `phase4a_*` (v2-v7) | Coarse SA + LP-fine refinement | 1.2692 (fast best), 1.4921-1.4929 (--all) | superseded — SA topologies don't survive LP refinement |
| `polyhedra_phase2_*` | Phase-2 polyhedra LP refinements | 1.2588-1.2676 (fast), 1.4867 (--all best) | superseded |
| `polyhedra_lp_cap_v2`, `polyhedra_overlap_fix` | LP cap and overlap variants | 1.4890-1.4895 (--all) | superseded |
| `optimal_transport` (n=13) | OT-based reassignments | 1.1953 (fast best) | superseded — never moved --all needle |
| `cluster_screener` | Topology screening | 1.2728 (fast) | superseded (eval pipeline only) |
| `sdf_density` (n=13) | SDF init iterations | 1.2775 (fast best) | now used as init for CD |

---

## Strategic-update timeline

| Date | Event | Source |
|---|---|---|
| 2026-04-26 | E8 LP-HPWL diagnostic: proxy is **6% WL, 20% density, 74% congestion**. Pure HPWL CD caps at ~5% improvement; full-proxy CD is required. | `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md`, `scripts/lp_hpwl_lower_bound.py` |
| 2026-04-26 | E1 Incremental evaluator validated: 4657× speedup, bit-for-bit parity. | `writeup/cd_ibm10_results.md` |
| 2026-04-27 (early) | E2 ibm10 single-bench: full-proxy CD on incremental evaluator hits 1.0632 (40min budget) and 1.1039 (10min). Beats DPO 1.254 by 12-15%. | `writeup/cd_ibm10_results.md` |
| 2026-04-27 (early) | E5 falsified: batched seeds wall scales 8.9×, all collapse to same basin. | `submissions/dpo/batched_seeds_placer.py` |
| 2026-04-27 (early) | CDOnly --all: avg 1.1193, matches leaderboard 1.1172 within 0.18%, -23.2% vs RePlAce. | `submissions/cd_only/placer.py` |
| 2026-04-27 (mid) | E10 marginal, E11 flat: within-DPO refinements cap at 1-2%; basin lock structural. | `submissions/dpo/congestion_refine_placer.py`, `diverse_priors_placer.py` |
| 2026-04-27 (mid) | E3 LNS v1 + v2 falsified: single-macro local LNS doesn't escape CD's local minimum. | (deleted from active tree in commit 44efd16; preserved at `writeup/archive/submissions/lns.py`, `writeup/archive/submissions/cd_lns_placer.py`) |
| 2026-04-27 17:22 | E9 CDAdaptive --all: avg 1.1055 — beat leaderboard by -1.05% (now superseded). | `submissions/cd_adaptive/placer.py` |
| 2026-04-28 (am) | Three CD-escape mechanisms falsified: E3 LNS (single-macro, flat); SDF-jitter multi-init (0/8 improved, contractive); subset-CD destroy (0/24 improved, same per-axis fixed point). All reused CD's move type. | `experiments/E3_lns_v1/`, `experiments/E3_lns_v2/`, `analysis/multi_init_probe/`, `analysis/lns_escape_probe/` |
| 2026-04-28 15:57 | **E12 CDLNSGridBin --all: avg 1.0990 — BEATS leaderboard by -1.63%, beats E9 by -0.59%.** Grid-bin LNS uses a *different* move type ((col, row) cell centers, outside CD's per-axis breakpoint set). Promoted to champion (ADR-007). | `submissions/cd_lns_gridbin/placer.py`, `docs/decisions/007_cd_lns_gridbin_promotion.md` |

---

## Lessons for the writeup

1. **Decomposing the proxy was the unblock.** Until E8 showed congestion was 74%
   of cost, we were optimizing the wrong thing. WL-only CD (Miftari-style
   weighted median) caps at ~5%; full-proxy CD is mandatory.

2. **Infrastructure gates everything.** The 4657× incremental evaluator turned
   intractable per-move queries into the inner loop of CD. Without it, 600s/bench
   does too few sweeps to converge.

3. **Within-basin optimization caps at 1-2%.** Multi-seed DPO, congestion
   refinement, diverse priors — all gave 0.3-1% on fast set, near-zero on --all.
   The basin lock on ibm02/12 was unbreakable from inside DPO.

4. **CD changes the basin.** Single-macro coordinate moves with a fast evaluator
   reach a *different* fixed point than gradient descent. ibm02 dropped from
   DPO's 1.6888 (byte-identical across 4 seeds) to CD's 1.1534 — basin-changing,
   not within-basin tuning.

5. **LNS escape requires a different move type, not more time on the same one.**
   Single-axis destroy/reinsert (E3 v1/v2, subset-CD probe) reuses CD's per-axis
   breakpoint search and finds the same fixed point. Multi-init via SDF jitter
   (multi_init_probe) is contractive — perturbing strictly hurts. **Grid-bin
   LNS (E12) escapes** because its candidate set is `(grid_col × grid_row)` cell
   centers, outside CD's per-axis reachable set. This is the structural
   distinction between flat LNS and effective LNS, and it's what carried the
   champion from 1.1055 to 1.0990. The leaderboard's "Incremental CD+LNS"
   description was correct after all — earlier attempts had picked the wrong
   move type.

6. **Plateau detection > fixed budget.** ALL 17 benchmarks exited via plateau
   (none hit the 1hr cap) under default `(min_time_s=300, hard_cap_s=3600,
   patience=3, plateau_threshold=0.005)`. This transfers to the hidden NG45 test
   without per-benchmark tuning.

---

## What's next (post-1.0990)

| Lever | Expected | Effort | Notes |
|---|---|---|---|
| NG45 hidden-test datapoint for E12 | Confirm transfer | 1-2 hr per design | ADR-007 needs its own NG45 result before treating LNS-overlay transfer as confirmed |
| Drop cost-aware destroy ranking (use random) | flat (already shown on `--fast`) | 30 min | Saves ~10% wall per LNS sample at no quality cost; defer until post-deadline to avoid mid-deadline changes |
| Tighten LNS budget cap to 1200 s? | speculative | 1 day | If 600 s LNS still finds improvements, more time may help; verify before changing |
| Cluster-level LNS (joint reinsertion of K>1 macros) | speculative | 1 week | Different again from grid-bin; could compose on top of E12 |
| GPU-batched CD via conflict-graph coloring (E4) | 5-20× speedup | 3-4 days | Useful only if E12 wall risks the 1-hr-per-bench cap on NG45 |
| Submission writeup | Required by May 21 | ~1 week | See `writeup/` (other agent's responsibility); update for E12 |

Champion entry is secured ~23 days before the deadline.
