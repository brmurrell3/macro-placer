# Experiment Index

Last updated: 2026-04-28

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
| **CD + grid-bin LNS** | **`e12_gridbin_lns`** (CDLNSGridBin) | **1.0990** | **2026-04-28 15:57** | **CURRENT CHAMPION; beats leaderboard 1.1172 by -1.63%; ADR-007.** |

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
