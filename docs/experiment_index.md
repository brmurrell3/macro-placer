# Experiment Index

Last updated: 2026-04-27

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
| **CD-adaptive** | **`e9_adaptive_all`** (CDAdaptive) | **1.1055** | **2026-04-27 13:22** | **BEATS leaderboard 1.1172 by -1.05%** |

Lineage: each champion replaced the prior by a structural change, not parameter
tuning.

---

## Live experiments (current)

| ID | Hypothesis | Files | Best | Mode | Status |
|---|---|---|---|---|---|
| E1 | Incremental proxy evaluator | `macro_place/incremental_evaluator.py`, `test/test_incremental_evaluator.py`, `scripts/bench_incremental.py` | 4657× speedup, bit-for-bit parity | infrastructure | **VALIDATED** — load-bearing for E2/E9/E3 |
| E2 | Full-proxy CD on ibm10 (40 min) | `scripts/cd_ibm10_diagnostic.py` | 1.0632 (ibm10 single) | single | **VALIDATED** — sparked CDOnly |
| E9 | Adaptive per-bench plateau detection | `submissions/cd/cd_adaptive_placer.py` | **1.1055 (--all)** | all | **CHAMPION** |

---

## Falsified / superseded experiments (kept as evidence)

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

### Falsified extension hypotheses (E5, E10, E11)

| ID | Hypothesis | Files | Best | Status | Lesson |
|---|---|---|---|---|---|
| **E5** | Massively parallel DPO seeds (best-of-N at scale) | `submissions/dpo/batched_seeds_placer.py` (679L), `batched_seeds_b1.py` | 1.1638 (fast), 1.1698 (B=64 fast) | **FALSIFIED 2026-04-26** | B=64 wall = 8.9× B=1 (RUDY congestion kernel scales 27× on MPS). All 64 seeds collapse to same basin (sigma=0.04·canvas). Best-of-N within a single basin doesn't help. |
| **E10** | Congestion-only refinement on DPO output | `submissions/dpo/congestion_refine_placer.py`, `_e10_sweep_{mild,aggressive}.py` | 1.1636 (fast) `mild`, 1.3788 (--all) | **MARGINAL/FALSIFIED** | Mild config gained 1.0% on fast set; only -0.33% on --all. **ibm02 got WORSE +2.3%** — basin-locked benchmarks regressed. |
| **E11** | DPO from diverse priors (SDF + Will + greedy + random) | `submissions/dpo/diverse_priors_placer.py`, `init_strategies.py`, `e11_sdf_only.py`, `e11_sdf_random.py` | 1.1659 (fast 2-prior), 1.3839 (--all) | **FALSIFIED 2026-04-27** | Fast-set -0.8% didn't scale to --all (FLAT +0.04%). ibm02/ibm12 got WORSE — alternative priors land in deeper basins for high-density benchmarks. |
| **E3 v1** | LNS rip-up-and-reinsert (full-canvas search) | `submissions/cd/lns.py`, `submissions/cd/cd_lns_placer.py` | 1.3846 (ibm17 single) | **FALSIFIED 2026-04-27** | Full-canvas 2244 grid candidates × ~100ms each = 200s+ per macro reinsertion. 1 LNS iteration in 428s. Final ibm17 = 1.3846 vs CDOnly 1.3830 (flat). |
| **E3 v2** | LNS with 5×5 local-window reinsert | `submissions/cd/lns.py` (window_radius=5), `cd_lns_placer.py` | 1.3824 (ibm17 single) | **FALSIFIED 2026-04-27** | 90× faster (15 iters in 600s) but final ibm17 = 1.3824 vs CDOnly 1.3830 = flat. Cost-based destroy selector saturates after 1-2 accepts; 5×5 window can't move clusters. Single-macro local LNS does NOT escape CD's local minimum. |

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
| 2026-04-26 | E8 LP-HPWL diagnostic: proxy is **6% WL, 20% density, 74% congestion**. Pure HPWL CD caps at ~5% improvement; full-proxy CD is required. | `docs/lp_hpwl_diagnostic.md`, `scripts/lp_hpwl_lower_bound.py` |
| 2026-04-26 | E1 Incremental evaluator validated: 4657× speedup, bit-for-bit parity. | `writeup/cd_ibm10_results.md` |
| 2026-04-27 (early) | E2 ibm10 single-bench: full-proxy CD on incremental evaluator hits 1.0632 (40min budget) and 1.1039 (10min). Beats DPO 1.254 by 12-15%. | `writeup/cd_ibm10_results.md` |
| 2026-04-27 (early) | E5 falsified: batched seeds wall scales 8.9×, all collapse to same basin. | `submissions/dpo/batched_seeds_placer.py` |
| 2026-04-27 (early) | CDOnly --all: avg 1.1193, matches leaderboard 1.1172 within 0.18%, -23.2% vs RePlAce. | `submissions/cd/cd_only_placer.py` |
| 2026-04-27 (mid) | E10 marginal, E11 flat: within-DPO refinements cap at 1-2%; basin lock structural. | `submissions/dpo/congestion_refine_placer.py`, `diverse_priors_placer.py` |
| 2026-04-27 (mid) | E3 LNS v1 + v2 falsified: single-macro local LNS doesn't escape CD's local minimum. | `submissions/cd/lns.py`, `cd_lns_placer.py` |
| 2026-04-27 17:22 | **E9 CDAdaptive --all: avg 1.1055 — BEATS leaderboard by -1.05%.** | `submissions/cd/cd_adaptive_placer.py` |

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

5. **LNS as we built it is not the missing piece.** Single-macro destroy/reinsert
   (full-canvas or local-window) gives essentially zero gain on top of CD. Real
   LNS would need cluster-level joint reinsertion or randomized destroy
   strategies. The leaderboard's "Incremental CD+LNS" name is a red herring for
   our architecture — the gain we got from CDOnly→CDAdaptive (1.1193→1.1055,
   -1.23%) came from giving hard benchmarks more time, not from any LNS phase.

6. **Plateau detection > fixed budget.** ALL 17 benchmarks exited via plateau
   (none hit the 1hr cap) under default `(min_time_s=300, hard_cap_s=3600,
   patience=3, plateau_threshold=0.005)`. This transfers to the hidden NG45 test
   without per-benchmark tuning.

---

## What's next (post-1.1055)

| Lever | Expected | Effort | Notes |
|---|---|---|---|
| Tighten plateau threshold (`0.005` → `0.002`) | -0.5 to -1% | 1 hr | Hard benchmarks may still descend below current exit |
| Cluster-level LNS | Speculative | 1 week | Would need joint reinsertion + multi-macro evaluator |
| GPU-batched CD via conflict-graph coloring (E4) | 5-20× speedup | 3-4 days | Useful only if we exceed budget; currently we don't |
| NG45 hidden-test prep / robustness | Confirm | 1-2 days | Verify zero overlaps, plateau detection on commercial designs |
| Submission writeup | Required by May 21 | ~1 week | See `writeup/` (other agent's responsibility) |

Champion entry is secured 24 days before the deadline.
