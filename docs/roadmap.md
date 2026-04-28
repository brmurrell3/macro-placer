# Roadmap

Last updated: 2026-04-28
Competition deadline: May 21, 2026 (~23 days)

---

## Current Position

| Entry | Avg Proxy (--all) | vs Leaderboard 1.1172 | vs RePlAce 1.4578 |
|-------|-------------------|------------------------|--------------------|
| **CDLNSGridBin (E12, CHAMPION)** | **1.0990** | **-1.63%** | **-24.6%** |
| CDAdaptive (E9, prior, superseded) | 1.1055 | -1.05% | -24.2% |
| CDOnly (prior-prior, superseded) | 1.1193 | +0.18% | -23.2% |
| DPO best-of-v2 (prior, superseded) | 1.3834 | +23.8% | -5.1% |
| RePlAce baseline | 1.4578 | +30.5% | --- |
| SA baseline | 2.1251 | +90.2% | +45.8% |

- **Beats the leaderboard "Incremental CD+LNS" (vmallela 1.1172) by -1.63%** with zero overlaps on all 17 IBM benchmarks.
- Total runtime 28 256 s (7.85 hr) — within the 17-hr competition envelope (17 × 1 hr per-bench cap), with reduced margin vs CDAdaptive's 4.85 hr.
- Champion configuration: full-proxy coordinate descent on incremental evaluator with per-benchmark plateau detection (CD ≤ 3 000 s) followed by grid-bin LNS escape phase (LNS ≤ 600 s). ADR-007.

---

## Champion lineage

| Era | Champion | Best (--all) | Date | Replaced because |
|---|---|---|---|---|
| Pre-DPO | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Polyhedra | PolyhedraNavigation | 1.4921 | 2026-04-15 | Hit ceiling — congestion barrier structural |
| DPO v1 | DPO v1 | 1.4255 | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| DPO v2/v3/v2-steps | best_of_v2 | 1.3834 | 2026-04-26 | Within-DPO refinements cap at 1-2% |
| **CD-only** | CDOnly | 1.1193 | 2026-04-27 (am) | Fixed 600s budget left hard benchmarks mid-descent |
| **CD-adaptive** | CDAdaptive (E9, superseded) | 1.1055 | 2026-04-27 (pm) | Plateau-bound: every bench exited via plateau, none hit cap. Same move type — couldn't escape per-axis fixed point. |
| **CD + grid-bin LNS** | **CDLNSGridBin (E12)** | **1.0990** | **2026-04-28** | (current — ADR-007) |

Each transition was structural, not parameter tuning. See `docs/experiment_index.md` for the full catalog including failed attempts.

---

## Completed phases

### Phases 1-5 — Polyhedra navigation (Apr 1 - Apr 15)

Decomposed feasible region as union of convex polyhedra; navigated with SDF init + LP + surrogate-guided search. Hit a 1.49 ceiling. Phase 5's 22-experiment sweep proved the polyhedra surrogate was already well-calibrated — the ceiling was structural (the gradient of relaxed objective lost too much in legalization).

### Phase 6 — DPO (Apr 16 - Apr 26)

Differentiable proxy: gradient through smooth WL+density+congestion with annealed overlap penalty. Reached 1.3834 via best-of-v2. Multi-seed verification showed **basin lock** — `best_of_v2_seed{43,44,45,46}` all within 1% on --all, and ibm02/ibm12 byte-identical across seeds. DPO converges to a deterministic local minimum that gradient steps cannot escape.

Falsified DPO extensions:
- **E5 (batched seeds, 2026-04-26):** B=64 wall = 8.9× B=1; all seeds collapse to same basin under sigma=0.04·canvas perturbation.
- **E10 (congestion-only refinement, 2026-04-27):** Fast set -1.0%; --all -0.33% with ibm02 *worse* (+2.3%). Basin-locked benchmarks regressed.
- **E11 (diverse priors, 2026-04-27):** Fast -0.8%; --all FLAT (+0.04%). ibm02/ibm12 worse. Greedy/random priors land in deeper basins on hard benchmarks.

### Phase 7 — Coordinate descent on full proxy (Apr 26 - Apr 27)

The 2026-04-26 LP-HPWL diagnostic decomposed proxy as **6% WL, 20% density, 74% congestion**. Pure HPWL CD (Miftari-style weighted median) caps at ~5%; *full-proxy* CD captures all three components.

- **E1 (incremental evaluator, validated 2026-04-26):** 4657× speedup per move; bit-for-bit parity. Load-bearing for everything that follows.
- **E2 (CD on ibm10 single-bench, 2026-04-27 early):** 40-min budget hit 1.0632 vs DPO 1.254 (-15%). 10-min budget hit 1.1039 (~85% of value). Generalized to ibm02 (1.1534, broke DPO basin lock from 1.6888) and ibm18 (1.3929).
- **CDOnly --all (2026-04-27 03:50):** Productionized at 600s/bench. avg 1.1193, matched leaderboard within 0.18%.
- **E3 LNS (v1 full-canvas, v2 5×5 local-window, 2026-04-27):** Falsified. Single-macro destroy/reinsert produces flat ibm17 result regardless of speed. Cluster-level joint reinsertion would be needed for real gains.
- **E9 CDAdaptive --all (2026-04-27 17:22):** Per-benchmark plateau detection. avg **1.1055**, beat leaderboard 1.1172 by -1.05%.
- **E12 CDLNSGridBin --all (2026-04-28 15:57):** CD plateau + grid-bin LNS overlay (different move type — escapes CD's per-axis fixed point). avg **1.0990**, beat leaderboard by -1.63% and prior champion by -0.59%. Promoted 2026-04-28 (ADR-007).

---

## What's left

### Submission (required)

| Item | Owner | Status |
|---|---|---|
| Submission package (placer + reproducibility) | this branch | Ready — `submissions/cd_lns_gridbin/placer.py` is the entry |
| Writeup (innovation prize report) | parallel agent (`writeup/`) | In progress — must include process AND failures |
| NG45 hidden-test robustness check | tbd | Verify zero overlaps, plateau detection on commercial designs |

### Optional polish (24 days available)

| Lever | Expected | Effort | Notes |
|---|---|---|---|
| NG45 datapoint for E12 | Confirm transfer | 1-2 hr per design | ADR-007 needs its own NG45 result before treating LNS-overlay transfer as confirmed |
| Drop cost-aware destroy ranking | flat (already verified on `--fast`) | 30 min | Saves ~10 % wall per LNS sample at no quality cost; defer until post-deadline |
| Cluster-level LNS (joint reinsertion of K>1 macros) | Speculative | 1 week | Different again from grid-bin; could compose on top of E12 |
| **GPU-batched CD via conflict-graph coloring (E4)** | 5-20× speedup | 3-4 days | Useful only if E12 wall risks the 1-hr-per-bench cap on NG45; currently 7.85 hr / 17 hr envelope on IBM |
| Multi-seed validation | Confirm robustness | 1 day | E12 is deterministic via SDF init; seed sensitivity TBD |

Decision rule: submit current champion (1.0990, E12 CDLNSGridBin) as the safe baseline. Iterate optional polish only if time permits and risk is bounded.

---

## Files of record

| File | Purpose |
|---|---|
| `submissions/cd_lns_gridbin/placer.py` | **CHAMPION submission** (E12, 1.0990) |
| `submissions/cd_adaptive/placer.py` | Prior champion (CDAdaptive E9, 1.1055; superseded 2026-04-28) |
| `submissions/cd_only/placer.py` | Prior-prior champion (CDOnly fixed-budget) |
| `writeup/archive/submissions/lns.py` | E3 LNS module — falsified; deleted from active tree in commit 44efd16, preserved at this archive path |
| `writeup/archive/submissions/cd_lns_placer.py` | E3 placer — falsified; deleted from active tree in commit 44efd16, preserved at this archive path |
| `submissions/dpo/best_of_v2_placer.py` | DPO prior champion (basin-locked) |
| `submissions/dpo/{e10,e11,batched,...}*.py` | Falsified DPO extensions; kept for writeup |
| `submissions/dpo/ablation_*.py` | DPO ablation studies |
| `macro_place/incremental_evaluator.py` | E1 — 4657× speedup, load-bearing |
| `scripts/cd_ibm10_diagnostic.py` | E2 — sparked the CD line |
| `scripts/lp_hpwl_lower_bound.py` | E8 — proxy-decomposition diagnostic |
| `docs/results.md` | Current champion per-bench tables |
| `docs/experiment_index.md` | **Rigorous catalog of every experiment** |
| `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` | E8 details |
| `writeup/closing_the_gap.md` | E9 win narrative + E3/E4 design sketches |
| `writeup/cd_ibm10_results.md` | E2 details |
| `writeup/historical_results.md` | DPO/Polyhedra/Overnight per-bench data |
| `results/experiment_log.jsonl` | Source of truth for all numbers |

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden NG45 commercial designs differ from IBM (different macro-count regime) | Medium | Plateau detection transfers without per-bench tuning. Verify with `--ng45` run before submission. |
| Float drift between incremental evaluator and `compute_proxy_cost` over thousands of moves | Low (~0.5-1.5% on ibm02 observed) | Final VALID is the canonical value. Internal proxy is biased low ~1% on dense benchmarks but this doesn't change ranking; plateau detection still works. |
| Overlap regression on edge cases | Low | Hard validation in placer raises RuntimeError on overlap. Tested on all 17 IBM with zero overlaps. |
| Submission package missing reproducibility info | Low | `submissions/cd_lns_gridbin/placer.py` self-contained; reuses `IncrementalProxyEvaluator`, SDF init, and the CDAdaptive runner from already-tracked files. |
| Champion wall close to 17-hr envelope (7.85 hr used) | Low | Still ~9 hr headroom on `--all`. Hidden NG45 designs of similar scale should fit within 1 hr/bench. Monitor `--ng45` per-bench wall before submission. |

---

## See also

- [approach.md](approach.md) — current CD-on-incremental-evaluator architecture
- [results.md](results.md) — current champion (CDLNSGridBin) per-benchmark tables
- [experiment_index.md](experiment_index.md) — full catalog including falsified hypotheses
- [lp_hpwl_diagnostic.md](lp_hpwl_diagnostic.md) — the diagnostic that unblocked CD
- `writeup/historical_results.md` — DPO/Polyhedra/Overnight per-bench tables
- `writeup/closing_the_gap.md` — narrative of the leaderboard-beating run
- `writeup/cd_ibm10_results.md` — single-benchmark CD breakthrough
- `writeup/theory.md` — theoretical foundations (polyhedra decomposition, proxy decomposition)

---

## Proposed experiments (from experiments_overnight.md retirement)

Carried forward from the retired `experiments_overnight.md` working doc.
Each has: hypothesis, kill gate (cross-benchmark), generalization check,
expected wall. Path references have been updated to the post-restructure
layout (`experiments/E<NN>_*/code/...`).

### Tier 1 — overnight runnable, generalization-safe

#### E16. Tighter plateau threshold + extended cap (cheapest score win)
**Hypothesis:** every hard benchmark exited CDAdaptive with last-3-sweep
deltas 0.001-0.005 — below the 0.005 threshold but each ≈0.001 of proxy.
Tightening to threshold=0.001, cap=2hr captures those tail sweeps.
**Algorithm change:** ONE global hyperparameter swap, applied to ALL
benchmarks. No per-bench tuning.
**Kill gate:** if the new threshold doesn't drop avg --all proxy by ≥0.005,
kill. (NB: applies to all 17 benches, not "hard ones only".)
**Generalization check:** run on the 4 public NG45 designs (ariane133/136,
mempool_tile, nvdla) — proxy on those should also drop or hold steady, not
regress.
**Wall:** ~12 hr (--all run with extended cap).
**Status:** completed; landed at 1.1025 (−0.0030, formally marginal vs the
0.005 kill gate). See `experiments/E16_tight_threshold/code/cd_adaptive_e16.py`
and `writeup/evidence.md` §7.6.

#### E12. Full-canvas grid-bin LNS (the actual LNS recipe we never tested)
**Hypothesis:** prior LNS failures used subset-CD reinsertion, which finds
the *same* per-axis fixed point. The actual recipe (E3 in roadmap) is
grid-bin search: for each destroyed macro, evaluate proxy at every (grid_col
× grid_row) center, pick min. This is a different move type CD can't reach.
**Algorithm:** Adaptive: destroy size = 5% of movable hard macros, capped at
30. Choose destroy via cost-aware ranking. For each destroyed macro,
enumerate (col×row) candidates ~50-2500 per benchmark. Pick globally best
position via incremental evaluator. Iterate until N samples or wall budget.
**Kill gate:** if 0/8 grid-bin LNS samples improve baseline by ≥0.5% on
**the median benchmark** (NOT the worst, NOT the easiest — pick the median
to avoid overfit), kill.
**Generalization check:** if it passes on median, validate on `--fast` and
one NG45 design before --all.
**Wall:** ~2 hr build + 4 hr validation = 6 hr.
**Status:** **GRADUATED 2026-04-28.** `--all` final 1.0990 — promoted to
champion (ADR-007), supersedes E9 CDAdaptive. Code now at
`submissions/cd_lns_gridbin/placer.py`; experiment record at
`experiments/E12_grid_bin_lns/manifest.md`.
**Note on prior art:** vmallela's #1 method is named "Incremental CD+LNS" —
this might be exactly their recipe. TAISPlAce did "ALNS + Thompson Sampling"
and only hit 1.4321, suggesting LNS execution detail matters a lot.

#### E14. SA polish on CDAdaptive output
**Hypothesis:** SA can accept worse moves probabilistically — explicit
tunneling through CD's fixed point. Different mechanism than LNS (which is
structural).
**Algorithm:** After CDAdaptive converges, run SA: single-macro moves drawn
from per-axis breakpoint set (same move space as CD), Metropolis acceptance
with global temperature schedule (T₀ → T_f over N moves). All
hyperparameters global.
**Kill gate:** SA polish on `--fast` shows no improvement over CDAdaptive
baseline → kill.
**Generalization check:** validate on NG45 ariane133.
**Wall:** ~3 hr (build + --fast eval).
**Why higher than E15:** SA is a known-strong method; this is the version
that didn't lock the ibm02 basin in our prior DPO experiments.

#### E15. 2-macro pair swap moves
**Hypothesis:** CD plateaus when each macro is at its single-macro fixed
point. Swapping two strongly-coupled macros (sharing nets) is a coordinated
move outside CD's reachable set.
**Algorithm:** Adaptive: candidate pairs = those sharing ≥2 nets (or top-K
by net adjacency, K = 5% of pairs, capped). For each candidate pair, query
proxy with positions swapped via incremental evaluator; accept if improving.
**Kill gate:** zero accepted swaps on `--fast` → CD's fixed point is stable
to swaps too → kill.
**Generalization check:** test on NG45 ariane133.
**Wall:** ~3 hr.
**Status:** v1 was buggy (`min_shared_nets=2` filter found zero candidates —
most macro pairs share exactly 1 net). v2 ran on `--fast` finding 21–53
real swaps per benchmark and landing at 0.9414 (essentially flat vs E16
baseline 0.9425). See `experiments/E15_pair_swap/code/cd_pair_swap.py`.
**Note on prior art:** the leaderboard has "MacroBio (Two-Opt Swap)" as
pending. Either result is data.

#### E23. NG45 sanity test (do this FIRST)
**Hypothesis:** before any new optimization, validate CDAdaptive runs
end-to-end on NG45 designs and produces ORFS-flowable placements. This is
**defensive** — Cezar got DQ-risked from a self-reported / verified
discrepancy. We need to confirm no analogous problem in our pipeline.
**Algorithm:** No change. Just run.
**Kill gate:** N/A — diagnostic only. Failure → fix bugs.
**Wall:** ~30 min.
**Output:** confirmation + any latent bugs surfaced.

### Tier 2 — medium-EV, build-if-Tier-1-mixed

#### E13. Congestion-region spatial-cluster LNS
Targeted destroy of macros in high-congestion regions. Adaptive criterion:
cells with congestion > median + 1σ. Different from E12 (which destroys
cost-aware random); E13 destroys spatial neighbors, releasing joint
constraints.
**Wall:** ~3 hr after E12 framework exists.

#### E17. Random-uniform init + CD (true diversification)
Replace SDF with random uniform legal placement. Test if SDF basin is
sometimes the wrong basin.
**Generalization check:** if random init is uniformly worse → expected (SDF
is a smart prior). If random init beats SDF on some benchmarks, that's
signal of basin diversity → run best-of-N from random + SDF mixed inits.
**Wall:** ~5 hr.

#### E18. DPO best_of_v2 → CDAdaptive polish (E6 redux)
Use DPO best_of_v2 (1.3134 avg) as init for CDAdaptive instead of SDF. Per
E11: alternative inits sometimes find different basins, but at risk of
worse outcomes on some benches.
**Wall:** ~6 hr full --all if --fast passes.

### Tier 3 — speculative / parking lot

#### E19. RUDY → Steiner-tree congestion
Different proxy modeling. **Risk:** if Steiner is more accurate but the
contest's TILOS evaluator uses RUDY (it does), our proxy diverges from the
scoring proxy. We optimize for a thing the contest doesn't measure. **Skip
unless Tier 2 needs it.**

#### E20. ILP polish on local windows
~3 days to build, uncertain payoff. Only if Tier 1 lacks improvement and
we have time.

#### E21. GPU-batched breakpoint search
Per Proof A, CD is plateau-bound, not budget-bound on IBM. Speed-up
doesn't directly help score on IBM. **Re-live for NG45 if hidden designs
are larger** and budget actually binds.

#### E22. Conflict-graph colored CD
Same caveat as E21. Only useful at scale.

---

## Deprioritized hypotheses

Carried forward from the retired `experiments_overnight.md`.

- **Per-benchmark hyperparameter tuning** — explicitly forbidden by
  competition rules
- **More multi-init CD via SDF jitter** — falsified, 0/8 improved,
  structurally explained (SDF is contractive). See
  `analysis/multi_init_probe/README.md` and ADR-005.
- **More random-destroy LNS via subset-CD** — falsified across 24 samples,
  two benchmarks. See `analysis/lns_escape_probe/README.md`.
- **Within-DPO refinement** (E10/E11 variants) — basin-locked per memory; CD
  already breaks that lock
- **GPU best-of-N seeds** (E5) — already falsified
- **Polyhedra navigation** — superseded
- **Reinforcement learning / GNN priors** — too long to build given May 21
  deadline; could be revisited if we want to chase Innovation Award narrative

---

## Open follow-ups

Carried forward from the retired `findings.md` and `experiments_overnight.md`.

- **NG45 sanity test** — we don't have the public NG45 designs locally;
  obtain or simulate "NG45-like" designs.
- **ORFS local integration** — multi-day project; the actual path to the
  $20K Tier-2 grand prize. Should be its own multi-day track.
- **Verify zero overlaps preserved through legalization on E12 final
  placement** — should be enforced by `compute_overlap_metrics` raise in
  placer; verify against the full --all result.
- **Tier 2 robustness check on E12** — verify behavior generalizes beyond
  IBM. Need NG45 access for this.
- **Multi-seed variance report** — produce a writeup figure showing
  CDAdaptive's variance across seeds on each benchmark (innovation-award
  material).
- **Adversarial / hidden-design stress test** — synthesize random
  "NG45-like" benchmarks (random sizes, densities, topologies) and verify
  CDAdaptive doesn't catastrophically fail on edge cases.
- **Tier 2 OpenROAD flow integration** — running ORFS locally to iterate on
  WNS/TNS/Area instead of proxy.
