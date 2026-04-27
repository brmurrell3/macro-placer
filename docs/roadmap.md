# Roadmap

Last updated: 2026-04-27
Competition deadline: May 21, 2026 (~24 days)

---

## Current Position

| Entry | Avg Proxy (--all) | vs Leaderboard 1.1172 | vs RePlAce 1.4578 |
|-------|-------------------|------------------------|--------------------|
| **CDAdaptive (E9, CHAMPION)** | **1.1055** | **-1.05%** | **-24.2%** |
| CDOnly (prior, superseded) | 1.1193 | +0.18% | -23.2% |
| DPO best-of-v2 (prior, superseded) | 1.3834 | +23.8% | -5.1% |
| RePlAce baseline | 1.4578 | +30.5% | --- |
| SA baseline | 2.1251 | +90.2% | +45.8% |

- **Beats the leaderboard "Incremental CD+LNS" (vmallela 1.1172) by -1.05%** with zero overlaps on all 17 IBM benchmarks.
- Total runtime 17480s (4.85 hr) — well within compute budget. ALL 17 benchmarks exited via plateau detection; none hit the 1hr cap.
- Champion configuration: full-proxy coordinate descent on incremental evaluator with per-benchmark plateau detection.

---

## Champion lineage

| Era | Champion | Best (--all) | Date | Replaced because |
|---|---|---|---|---|
| Pre-DPO | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Polyhedra | PolyhedraNavigation | 1.4921 | 2026-04-15 | Hit ceiling — congestion barrier structural |
| DPO v1 | DPO v1 | 1.4255 | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| DPO v2/v3/v2-steps | best_of_v2 | 1.3834 | 2026-04-26 | Within-DPO refinements cap at 1-2% |
| **CD-only** | CDOnly | 1.1193 | 2026-04-27 (am) | Fixed 600s budget left hard benchmarks mid-descent |
| **CD-adaptive** | **CDAdaptive (E9)** | **1.1055** | **2026-04-27 (pm)** | (current) |

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

---

## What's left

### Submission (required)

| Item | Owner | Status |
|---|---|---|
| Submission package (placer + reproducibility) | this branch | Ready — `submissions/cd/cd_adaptive_placer.py` is the entry |
| Writeup (innovation prize report) | parallel agent (`writeup/`) | In progress — must include process AND failures |
| NG45 hidden-test robustness check | tbd | Verify zero overlaps, plateau detection on commercial designs |

### Optional polish (24 days available)

| Lever | Expected | Effort | Notes |
|---|---|---|---|
| Tighten plateau threshold (`0.005` → `0.002`) | -0.5 to -1% | 1 hr | Hard benchmarks may still descend below current exit |
| Cluster-level LNS (joint reinsertion) | Speculative | 1 week | E3 single-macro versions falsified — would need new architecture |
| **GPU-batched CD via conflict-graph coloring (E4)** | 5-20× speedup | 3-4 days | Useful only if we exceed budget; currently we're at 4.85 hr and hidden cap is 17 hr |
| Multi-seed validation | Confirm robustness | 1 day | E9 is deterministic via SDF init; seed sensitivity TBD |

Decision rule: submit current champion (1.1055) as the safe baseline. Iterate optional polish only if time permits and risk is bounded.

---

## Files of record

| File | Purpose |
|---|---|
| `submissions/cd/cd_adaptive_placer.py` | **CHAMPION submission** |
| `submissions/cd/cd_only_placer.py` | Prior champion (CDOnly fixed-budget) |
| `submissions/cd/lns.py` | E3 LNS module — falsified; kept as evidence |
| `submissions/cd/cd_lns_placer.py` | E3 placer — falsified; kept as evidence |
| `submissions/dpo/best_of_v2_placer.py` | DPO prior champion (basin-locked) |
| `submissions/dpo/{e10,e11,batched,...}*.py` | Falsified DPO extensions; kept for writeup |
| `submissions/dpo/ablation_*.py` | DPO ablation studies |
| `macro_place/incremental_evaluator.py` | E1 — 4657× speedup, load-bearing |
| `scripts/cd_ibm10_diagnostic.py` | E2 — sparked the CD line |
| `scripts/lp_hpwl_lower_bound.py` | E8 — proxy-decomposition diagnostic |
| `docs/results.md` | Full per-bench result tables |
| `docs/experiment_index.md` | **Rigorous catalog of every experiment** |
| `docs/closing_the_gap.md` | E9 win narrative + falsification record |
| `docs/lp_hpwl_diagnostic.md` | E8 details |
| `docs/cd_ibm10_results.md` | E2 details |
| `results/experiment_log.jsonl` | Source of truth for all numbers |

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden NG45 commercial designs differ from IBM (different macro-count regime) | Medium | Plateau detection transfers without per-bench tuning. Verify with `--ng45` run before submission. |
| Float drift between incremental evaluator and `compute_proxy_cost` over thousands of moves | Low (~0.5-1.5% on ibm02 observed) | Final VALID is the canonical value. Internal proxy is biased low ~1% on dense benchmarks but this doesn't change ranking; plateau detection still works. |
| Overlap regression on edge cases | Low | Hard validation in placer raises RuntimeError on overlap. Tested on all 17 IBM with zero overlaps. |
| Submission package missing reproducibility info | Low | `cd_adaptive_placer.py` self-contained; reuses `IncrementalProxyEvaluator` and SDF init from already-tracked files. |

---

## See also

- [approach.md](approach.md) — current CD-on-incremental-evaluator architecture
- [results.md](results.md) — per-benchmark champion tables (CDAdaptive, CDOnly, DPO, RePlAce)
- [closing_the_gap.md](closing_the_gap.md) — narrative of the leaderboard-beating run
- [experiment_index.md](experiment_index.md) — full catalog including falsified hypotheses
- [theory.md](theory.md) — theoretical foundations (polyhedra decomposition, proxy decomposition)
- [lp_hpwl_diagnostic.md](lp_hpwl_diagnostic.md) — the diagnostic that unblocked CD
- [cd_ibm10_results.md](cd_ibm10_results.md) — single-benchmark CD breakthrough
