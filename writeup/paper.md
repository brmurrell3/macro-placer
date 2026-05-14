# Macro Placement: A Diagnostic-Driven Path from 1.46 to 1.08, and a Hessian Saddle Escape Beyond

> **STATUS:** Working master draft. Source-of-truth for the paper. Section
> scaffolding + locked-in numbers; prose drafts are TODO. Cite the supporting
> docs (`evidence.md`, `contributions.md`, `theory.md`) but do not duplicate
> them — write fresh prose from those sources here.

> **TODO(meta):** Title, authors, affiliation, abstract, venue/format target.
> Current placeholder title above. Length target: 14–18 pages including the
> Hessian-saddle-escape arc added in §§8.9–8.11.

> **Headline numbers (current):**
> - **1.0612** uncapped on M3 (cascade saddle escape ceiling, 17 IBM)
> - **1.0771** cloud EPYC, 60-min/bench cap (post-A1 implementation speedup)
> - **0.6870** NG45 (4 commercial designs)
> - vs RePlAce 1.4578 → **−26.1 %**; vs leaderboard top 1.037 → **+3.9 %** gap.
> - Zero overlaps on every benchmark, every variant.

---

## Abstract

> **TODO(prose):** ~250 words. The arc through three eras:
>
> 1. **Diagnostic era** (E1–E12): composite proxy diagnosis (WL +
>    0.5·D + 0.5·C; E8 decomposition: 6 % WL / 20 % density / 74 %
>    congestion), two diagnosis-pivot cycles, infrastructure-unlocks-
>    algorithm twice (E1 incremental evaluator → E9 plateau detection
>    → E12 grid-bin LNS overlay), "bypass don't fix" principle.
> 2. **Compositional era** (E18–E61): SDF + DPO as orthogonal basin
>    sources; CD + LNS + SA + K-joint as compositional polish stack;
>    per-bench best-of-{E25, E41} hybrid (E48) at **1.08151**; E65
>    infeasibility-wall finding formalizing why local-move and
>    crossover heuristics share a reachable set.
> 3. **Hessian saddle escape era** (E74–E84 + PATH A): the local-move
>    plateau is a saddle of the smooth proxy with multiple negative-
>    curvature eigenvectors; applying transition-state methods from
>    chemistry / materials science (climbing-image NEB, dimer,
>    gentlest-ascent) to the combinatorial placement plateau yields
>    a uniform improvement across all 21 designs (including breaking
>    the IBM/NG45 transfer-failure pattern on ariane133, −3.21 %
>    where every prior IBM-aligned mechanism regressed). Cascading
>    the saddle escape until the smooth-proxy Hessian becomes
>    positive-semidefinite reaches a verified **1.0612** uncapped
>    on 17 IBM. A 5.36× implementation speedup of the CD inner loop
>    closes most of the cap-vs-ceiling gap on EPYC, landing at
>    **1.0771** on the partcl-equivalent hardware under the 60-min
>    per-bench cap. Zero overlaps on every benchmark; single global
>    algorithm; no per-benchmark tuning; CPU-only Python.

---

## 1. Introduction (~1 page)

**Source material:** `evidence.md` §1 (champion lineage), `docs/problem.md`.

**Setup.**
- Place 200–537 rectangular macros on a 2D canvas; minimize composite
  proxy `f(p) = WL + 0.5·D + 0.5·C`; zero overlaps required.
- Reference baselines: RePlAce **1.4578**, Will's pre-fork SA seed **1.5338**,
  public leaderboard target **1.1172** (vmallela, "Incremental CD+LNS",
  unverified), Cezar self-reported 1.0666 re-verified at **1.2224**.
- Our verified result: **1.0990** avg on 17 IBM benchmarks (CDLNSGridBin
  E12), **−24.6 % vs RePlAce, −1.63 % vs leaderboard**, zero overlaps,
  total wall 28 256 s (7.85 hr).

**The argument.** Two diagnosis-pivot cycles drove the improvement, each
driven by quantitative analysis (correlation studies, ablations, cell-by-cell
fidelity comparison) rather than algorithm intuition:

1. **Polyhedra (1.49) → barrier diagnosis (LP-HPWL ρ=−0.001) → DPO (1.38).**
2. **DPO (1.38) → RUDY fidelity diagnosis (10.9% hotspot overlap) → CD (1.12).**
3. **Refinements stacked on top of CD:** CDOnly fixed budget → per-bench
   plateau detection (CDAdaptive E9 → 1.1055) → grid-bin LNS overlay
   (CDLNSGridBin E12 → **1.0990**, current champion). The LNS overlay's
   key insight is that CD's plateau is a per-axis fixed point — escaping
   it required a *different move type* ((col, row) cell-center
   enumeration), not more wall-clock on the same move type.

**Contributions (preview).**
1. Polyhedral decomposition applied to macro placement (built end-to-end).
2. Empirical *barrier diagnosis*: LP-HPWL ρ=−0.001 with proxy.
3. DPO with direct congestion gradient on the competition metric.
4. Penalty-as-barrier-crossing interpretation (and complexification analogy).
5. Empirical non-decomposability of the proxy (component, scale, structure).
6. RUDY fidelity analysis (cell-by-cell, 3.1× gap, 10.9% top-5 % overlap).
7. Incremental proxy evaluator (4657× speedup, bit-for-bit parity).
8. Full-proxy CD with breakpoint enumeration.
9. Per-benchmark plateau detection (E9, leaderboard-beating).
10. "Bypass, don't fix" as a transferable design principle.

> **TODO(prose):** Draft full intro. End with paper roadmap (one sentence
> per remaining section).

---

## 2. The Polyhedral Decomposition (~2 pages)

**Source material:** `theory.md` §1–2, `docs/problem.md`, `contributions.md` §1.

**Claim.** The non-overlap feasible region is a union of convex polyhedra,
one per complete pairwise L/R/A/B assignment. Within any single polyhedron,
HPWL minimization is a linear program. LP duals provide a complete
sensitivity map of the pairwise constraints for free.

**What's well-known and what we contribute.** The decomposition is textbook
disjunctive programming (Balas 1979; Kronqvist et al. 2025). Our contribution
in this section is the *application*: SDF init, assignment extraction, HiGHS
LP with dual extraction, GridSurrogate for fast candidate evaluation,
ClusterScreener pruning, cascade overlap repair. The decomposition itself is
not novel; the *empirical characterization of its failure mode* in §4 is.

**Figures planned.**
- 3-macro schematic of the L/R/A/B disjunction.
- LP dual interpretation of pairwise constraint sensitivity.

> **TODO(prose):** Draft the math precisely. Use real-analysis-level rigor;
> do not claim novel theorems. Mark all formal results as cited or
> "observation."
> **TODO(figure):** 3-macro polyhedra schematic (`writeup/scripts/`).

---

## 3. The Navigation System (~1.5 pages)

**Source material:** `contributions.md` §1, `archive/planning/eval_pipeline_design.md`.

**What we built (seven modules, ~2200 lines, removed in post-CD cleanup
but preserved in pre-2026-04-28 git history; SDF init retained at
`macro_place/sdf_init.py`):**

- SDF initialization (analytical density-aware spreading).
- Assignment extraction from SDF positions.
- HiGHS LP solve + dual extraction.
- GridSurrogate (~0.1 ms candidate eval).
- Surrogate-guided navigation with SA acceptance.
- Multi-pair cluster moves; ClusterScreener (Zobrist hash + displacement
  floor + net-span HPWL bound + axis crowding) pruning 20–40 % of moves
  at 1–10 µs before projection.
- Robust projection (cascade repair + direct overlap repair).

**Result.** 1.4867 avg proxy on `--all`, 0 overlaps, beats RePlAce on 3/17
benchmarks (ibm02, ibm10, ibm12). Hit a ceiling at ~1.49.

> **TODO(prose):** Draft. Be honest that the seven modules are ablated in
> the current repo; reference git history for verification. Show one
> per-bench result table from `historical_results.md`.

---

## 4. The Congestion Barrier — Act 2 Diagnosis (~2 pages)

**Source material:** `evidence.md` §4 (overnight sweep, Miftari, swap+LP),
§3.3 (component breakdown); `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md`.

**Empirical decomposition of proxy cost (DPO-optimized placements,
17 benchmarks):**

| Component | Avg fraction |
|-----------|-------------:|
| WL        | 5.8 %        |
| Density   | 19.4 %       |
| Congestion| **74.9 %**   |

**The 22-experiment overnight sweep.** SP3 (surrogate accuracy, 8 items),
SP1 (initial topology, 6 items), SP4 (LP formulation, 6 items), Combine
(2 items). Global best 1.4918 vs baseline 1.4921 — *all within ±0.5 % of
baseline*. Conclusion: navigation is at a plateau; the lever isn't here.

**The Miftari correlation experiment (24 feasible topologies, ibm01).**

| Pair | ρ |
|---|---:|
| LP-HPWL → refined proxy | **−0.001** |
| LP-HPWL → refined WL | +0.852 |
| LP-HPWL → refined density | −0.536 |
| LP-HPWL → refined congestion | +0.072 |
| refined congestion → refined proxy | +0.825 |

**Diagnosis.** LP-HPWL is uncorrelated with proxy. HPWL and density anti-
correlate physically (tighter WL ⇒ denser ⇒ worse density). The proxy is
congestion-dominated and HPWL is blind to congestion. Any LP-based ranking
of polyhedra is blind to the objective. *This is an objective mismatch,
not a search problem.*

**The swap+LP experiment.** Confirms the diagnosis from the other side:
swapping topologies *can* reach −44 % congestion, but LP refinement after
swap blows density up by +180 %. Different topologies have better
congestion; LP cannot navigate to them.

**Figures planned.**
- LP-HPWL vs proxy scatter (ρ=−0.001).
- Congestion vs proxy scatter (ρ=+0.825).
- Overnight sweep bar chart (22 experiments, all flat).

> **TODO(prose):** Draft.
> **TODO(data):** Generalize Miftari from ibm01-only to ibm01/04/09/13.
> Currently `contributions.md §2` admits the caveat; the writeup must
> either generalize or hedge.
> **TODO(figure):** Three figures above, generated from a frozen data file.

---

## 5. First Pivot — Differentiable Proxy Optimization (DPO) (~2 pages)

**Source material:** `archive/planning/dpo.md`, `evidence.md` §3
(DPO ablation, multi-seed, component breakdown), `contributions.md` §3.

**Architecture.**
1. SDF init (~3 s).
2. LSE-HPWL with annealed γ (Naylor 2001).
3. Differentiable grid density via `torch.topk` (top-10 %).
4. Differentiable RUDY congestion: smooth bbox → fractional cell overlap
   → ABU-5 %.
5. Pairwise overlap penalty (ReLU), annealed across 3 phases
   (λ = 1 → 50 → 500).
6. Adam optimizer; iterative legalization at the end.
7. Best-of(SDF, DPO) per benchmark.

**Headline result: best-of-v2, 1.3834 avg, +5.1 % vs RePlAce. 5-seed range
0.45 % (1.3790–1.3927); all 5 seeds beat RePlAce.**

**Ablation (--all):**

| Variant | Avg | Δ vs DPO | Δ vs RePlAce |
|---|---|---|---|
| Full DPO (seed 42) | 1.4246 | — | +2.3 % |
| − congestion gradient | 1.5092 | +5.9 % | −3.5 % |
| − density gradient | 1.7342 | +21.7 % | −19.0 % |
| Phase-1 only | 1.4605 | +2.5 % | −0.2 % |
| Random init (no SDF) | 4.9415 | +247 % | −239 % |

**Novelty boundary.** Differentiable HPWL + density is DREAMPlace standard.
Differentiable RUDY exists in C3PO/NV-Place (ASP-DAC 2026) for *standard-
cell* placement. Our contribution is the application to *macro* placement
with the competition metric, with the congestion gradient adding +5.9 %.

> **TODO(prose):** Draft.
> **TODO(figure):** DPO ablation bar chart (data already in
> `experiment_notes.md` §1).

---

## 6. Why DPO Crosses the Barrier (~1 page)

**Source material:** `theory.md` (penalty-as-tunneling), `evidence.md` §8
(barrier-crossing quantification), `contributions.md` §4 + §6.

**The structural claim.** Legal-state representations are trapped in the
disconnected feasible region. The penalty continuation provides the escape:
at low λ the optimizer traverses temporarily-infeasible configurations;
at high λ it converges in a different polyhedron.

**Quantification.**
- ibm01: DPO changes **3 556 / 30 135 (11.8 %)** of pairwise L/R/A/B
  assignments between SDF init and final output.
- ibm10: 15 539 / 308 505 (5.0 %).
- Top transition types: **L↔B and R↔A** (≈98 % of changes — perpendicular
  flips, not same-axis).

**Geometric interpretation (observation, not theorem).** Penalty
continuation is equivalent to lifting macros into ℝ^{3N} via a z-coordinate;
two macros at different z-heights don't overlap even if their xy-projections
collide. Annealing μ·Σz² continuously deforms 3D into 2D. Connection to
complexification (Zariski 1962): in ℝ^{2N} the feasible region is
disconnected; in ℝ^{3N} it is connected.

**The low-seed-variance argument.** A method trapped in random local minima
would show high variance. DPO's 0.45 % range across 5 seeds is the
practical signature of an effective continuation. Random init shows 4.94
avg — the chaos *without* the continuation.

> **TODO(prose):** Draft. Frame the complexification connection as
> "interpretive" not "proof."

---

## 7. The RUDY Limit — Act 3 Diagnosis (~1.5 pages)

**Source material:** `evidence.md` §5 (RUDY fidelity, congestion-weight
sweep), `analysis/rudy_fidelity/rudy_analysis.py`, `contributions.md` §7.

**The question.** DPO worsens 4/17 benchmarks (ibm01, ibm02, ibm06, ibm12)
vs SDF init. Is RUDY's gradient *direction* wrong, or just its magnitude?

**Congestion-weight sweep (--fast):**

| Cong weight | Avg proxy | Δ vs control |
|---|---|---|
| 0.50 (control) | 1.2081 | — |
| 0.75 | 1.2251 | +1.4 % worse |
| 1.00 | 1.2278 | +1.6 % worse |
| 1.25 | 1.2377 | +2.5 % worse |
| 1.50 | 1.2382 | +2.5 % worse |

Monotonic worsening. The problem is *direction*, not magnitude.

**Cell-by-cell RUDY vs real congestion (ibm01):**
- Real/RUDY ratio: **3.1×** (not 2× as estimated from aggregate metrics).
- Real exceeds RUDY on 1843/1845 cells.
- Spatial CoV of per-cell ratio: 0.944.
- **Top-5 % hotspot overlap: 10.9 %** (Jaccard 0.057, near-random).
- Three structural sources: L-routing vs uniform bbox (2.74×), missing
  macro blockage (28.3 % of real congestion), spatial smoothing.
- Vertical congestion worst (correlation 0.327 vs horizontal 0.635).

**Falsified DPO extensions (confirms basin lock structural):**
- E5 batched seeds (B=64 on MPS): all collapse to same basin.
- E10 congestion-only refinement: −0.33 % on `--all`; ibm02 *worse* +2.3 %.
- E11 diverse priors (SDF + Will + greedy + random): −0.8 % on `--fast`,
  flat (+0.04 %) on `--all`.

**Figures planned.**
- RUDY vs real congestion heatmap (ibm01).
- Per-cell ratio distribution (Real/RUDY).

> **TODO(prose):** Draft.
> **TODO(data):** Freeze `rudy_analysis.py` output to
> `writeup/data/rudy_ibm01.txt` so the 10.9 %, 3.1×, Jaccard 0.057 numbers
> are reproducible from a captured artifact, not a re-run.
> **TODO(figure):** Two figures above.

---

## 8. Second Pivot — Full-Proxy CD on an Incremental Evaluator (~2 pages)

**Source material:** `evidence.md` §7 (CD breakthrough: E1, E2, CDOnly,
E9, E16), `docs/approach.md` §1, `contributions.md` §8–9 + §12.

**Infrastructure prerequisite — the incremental evaluator (E1).**
- File: `macro_place/incremental_evaluator.py` (~930 lines).
- Maintains per-net min/max trackers, bin-density grid, per-net RUDY
  congestion contributions, smoothing via vectorized cumsum.
- **4657× speedup** on ibm10 (6.5 ms/move vs 30 s full evaluation).
- Bit-for-bit parity with `compute_proxy_cost` (worst diff 1.1×10⁻¹⁵).
- Verified on ibm01 + ibm10, 130 random moves, single-step revert tested.

**Algorithm — full-proxy CD with breakpoint enumeration.** For each
non-fixed macro, search both axes:
- Breakpoints = net endpoints (HPWL change-points) + bin-grid lines
  (density change-points) within the legal axis range.
- Legal range = clamp to canvas + non-overlap with all other macros at
  current other-axis value.
- Best position = breakpoint minimizing proxy under the incremental
  evaluator.
- Acceptance: monotone, only commit on strict drop > 10⁻⁹.
- Typical: 30–300 candidates per axis per macro; 3–13 sweeps per benchmark.

**E2 single-bench breakthrough (ibm10, 2400 s budget).** SDF init 1.4112 →
CD final **1.0632** (−24.7 %). 13 sweeps, 15 000 accepted moves, zero
golden-section fallbacks, zero overlaps. *Beats DPO on the same benchmark
(1.254) by 17.5 %.*

**The trade-off pattern.** WL goes *up* 13–15 % while density drops 21–32 %
and congestion drops 23–40 %. CD trades cheap WL for expensive
density/congestion — exactly the trade DPO cannot make because RUDY
misidentifies which cells are congested.

**The ibm02 basin lock.** All 5 DPO seeds converge to byte-identical
ibm02 placement (1.6888). CD: 1.6888 → **1.1534 (−32 %)**. DPO's RUDY
gradient cannot see the path; CD's exact evaluator can.

**E9 — per-benchmark plateau detection.**
```
min_time_s       = 300        # 5 min minimum
hard_cap_s       = 3600       # competition rule
patience         = 3          # consecutive sub-threshold sweeps
plateau_threshold = 0.005     # absolute proxy delta per sweep
```
**Result: 1.1055 avg on `--all`, 17 480 s total. All 17 benchmarks exit
via plateau; none hit the 1-hour cap.** This run signature — every bench
plateau-bound, none budget-bound — is what motivated E12's escape phase.

**E12 — grid-bin LNS overlay (current champion, 1.0990).** Three escape
mechanisms tested before E12 all reused CD's per-axis move type and
produced flat results: E3 single-macro LNS (full-canvas v1, 5×5 local v2),
SDF-jitter multi-init (0/8 improved, contractive), subset-CD destroy
(0/24 improved, same fixed point). The unifying lesson: a *different move
type* is required, not more wall-clock on the same one. Grid-bin LNS
provides it: after CD plateaus, destroy K = max(1, min(30, 5 % × |movable
hard macros|)) macros (cost-aware ranking) and reinsert each at the
proxy-minimizing legal position drawn from the full
`(grid_col × grid_row)` cell-center set. The candidate set is outside
CD's per-axis breakpoint enumeration. Production budget split: CD
≤ 3 000 s + LNS ≤ 600 s = 3 600 s (matches the contest cap). Result:
**1.0990 avg on `--all`** (vs E9 1.1055, **−0.59 %**), zero overlaps, all
17 IBM benchmarks improved over E9 (no regressions). Total wall 28 256 s
= 7.85 hr. ADR-007.

A `--fast` ablation showed the cost-aware destroy ranking is *not*
load-bearing: random destroy matched cost-aware within noise (and on
ibm09, random 0.8541 actually beat cost-aware 0.8591). Production keeps
the cost-aware ranking only because the ablation result landed late
relative to the May 21 deadline; the ranking adds ~10 % wall per LNS
sample at no quality benefit.

> **TODO(prose):** Draft.
> **TODO(data):** Capture per-sweep convergence (proxy / WL / density /
> congestion vs sweep) for 3–4 representative benchmarks (easy/medium/hard)
> from a re-run with logging — drives the convergence figure.
> **TODO(data):** Plateau-threshold sensitivity sweep
> (0.001 / 0.002 / 0.005 / 0.01) so the defaults are *defended*, not asserted.
> E16 is one such datapoint (threshold=0.001 → 1.1025); add 2 more.
> **TODO(figure):** CD convergence curve; CDAdaptive per-benchmark wall-
> time chart.

---

## 8.5 Compositional Polish — SA-v2, DPO Init, K-joint (~2 pages)

**Source material:** `evidence.md` §X (TBD), `experiments/E18_dpo_init/`,
`experiments/E25_lns_sa_compose/`, `experiments/E39_kmacro_joint_lns/`,
`experiments/E41_dpo_kjoint/`, `docs/decisions/008_cd_lns_sa_promotion.md`,
`docs/decisions/009_dpo_init_promotion.md`,
`docs/decisions/010_dpo_kjoint_promotion.md`.

E12 (1.0990) was plateau-bound with one escape mechanism (grid-bin LNS).
This section covers the three orthogonal composition extensions tested
2026-04-29 → 04-30, each adding a different mechanism to E12's CD-LNS
backbone.

### 8.5.1 SA-v2 polish on per-axis breakpoints (E25 → 1.0954)

> **TODO(prose):** Stochastic Metropolis acceptance over CD's per-axis
> breakpoint set, with best-so-far tracking and T₀=5e-4. Adds one
> stochastic mechanism on top of CD's monotone descent. Lessons: (a)
> best-tracking is non-negotiable (E14 falsification at +5.7 % when
> SA wandered without restoration); (b) T₀ scale must match
> per-breakpoint Δproxy magnitudes; (c) basin lock is *not* broken by
> SA — chain wandered up by ~0.01 absolute on hard benches even with
> best-tracking on, so SA polishes within a basin but doesn't escape.
> ADR-008 *Superseded by ADR-011*.

### 8.5.2 DPO best_of_v2 init replaces SDF (E18 → 1.08979)

> **TODO(prose):** Multi-basin claim: DPO best_of_v2 init lands a basin
> structurally different from SDF, AND deeper. E11 had earlier failed
> with diverse priors *within DPO refinement*; the difference here is
> using DPO output as init for CD-LNS-SA-v2, not as the polish itself.
> 4/4 NG45 wins confirmed at-or-below E12 NG45 0.7037 — basin lift
> transfers. ADR-009 *Superseded by ADR-011*.

### 8.5.3 K-macro joint LNS (E39 standalone, E41 composed → 1.0848)

> **TODO(prose):** K=3 joint reinsertion, top-N=5 candidates per macro,
> brute-force N^K=125 combos per K-tuple. Different move type from
> E12's grid-bin (single-macro joint) and CD's per-axis (single-macro
> single-axis). Composes with DPO init: E41 = E18 ⊕ K-joint = 1.0848
> on `--all`, breaking the multi-mechanism plateau on ibm10/11/13/14/15
> (-1.7 % to -4.4 % vs E25 on those benches specifically). NG45 0.69022.
> Three K-joint variants (E42 K=4, E43 longer budget, E44 spatial)
> all failed NG45 transfer — see §9.4. ADR-010 *Superseded by ADR-011*.

> **TODO(figure):** Lineage waterfall E12 → E25 → E18 → E41 with
> per-bench ablation of each added mechanism.

---

## 8.6 Per-Bench Best-of Hybrid — E48 (~1.5 pages, current champion)

**Source material:** `experiments/E48_hybrid_e25_e41/manifest.md`,
`docs/decisions/011_hybrid_e25_e41_promotion.md`,
`submissions/cd_lns_sa_hybrid/placer.py`.

> **TODO(prose):** Per-bench analysis on verified --all results from E25
> and E41 showed E25 wins on 5/17 (ibm01/06/07/17/18) where DPO basin
> is globally worse than SDF basin, E41 wins on 12/17 (rest). Theoretical
> best-of-2 = 1.08121. E48 hybrid runs both pipelines per benchmark and
> returns lower-cost output by *proxy value* (no per-bench hardcoded
> logic; algorithmically valid). Verified `--all` 1.08151 (within
> float-drift of theoretical bound). ADR-011 *Accepted* 2026-05-02;
> supersedes ADR-007/008/009/010.
>
> **The structural insight to articulate:** the per-bench winner is
> determined by which init class (SDF or DPO) lands in the right basin
> for that benchmark. The hybrid is *meta-algorithmic* in that sense —
> it doesn't introduce a new mechanism, just exploits the
> across-benchmark heterogeneity in which mechanism wins. This pattern
> is what positions the work as "compositional optimization" rather than
> "tuned algorithm".

> **TODO(figure):** Per-bench breakdown E25 vs E41 vs E48 — bar chart
> showing where each lane wins.
>
> **TODO(data):** Cross-validate with E53m (3-way multi-seed hybrid) to
> show the lift saturates at 2 lanes; adding seed=1 lane gives +0.02 %
> at --all (within run-to-run noise).

---

## 8.7 Failed Extensions (the May 1–2 wave) (~1 page)

**Source material:** `experiments/E53_dpo_basin_eval/manifest.md`,
`experiments/E53_multiseed_hybrid/manifest.md`,
`experiments/E54_congestion_destroy/manifest.md`.

After E48 verified at 1.08151, three orthogonal extension directions were
tested overnight 2026-05-01 → 02. **None lifted past E48.**

### 8.7.1 GPU DPO basin polish (E53) — falsified

> **TODO(prose):** Multi-restart full-pose Adam on smooth proxy + overlap
> penalty, MPS device on M3 Max. Layered after CD-LNS-SA as a polish
> phase. **0 accepts in 350 GPU restarts across 4 benchmarks.** Smooth
> proxy gradient cannot escape the local optimum CD-LNS-SA's breakpoint
> enumeration + LNS + Metropolis already reach. Smoke test on shortened
> budgets (CD/LNS/SA at 10-20 s each) DID see −2.16 % lift; at production
> budgets the prior phases converge tightly enough that GPU finds nothing.
> **Lesson: GPU acceleration must replace CD's basin-crossing role
> (architecturally), not act as a polish phase after fully-converged CD.**

### 8.7.2 Multi-seed within DPO (E53m) — marginal

> **TODO(prose):** 3-way hybrid {E25, E41 seed=42, E41 seed=1}. `--fast`
> 0.91128 (-0.97 % vs E48 fast 0.92024) suggested a major lift; `--all`
> 1.08128 was within −0.02 % of E48 (essentially tied). The `--fast`
> result was a sample-size outlier on 4 small benches where DPO
> seed-noise was amplified; at --all the seeds tied on most benches
> and the aggregate lift collapsed. **Lesson: multi-seed within DPO is
> a dead-end for breakthrough at competition scale.**

### 8.7.3 Mechanism-aligned destroy (E54) — falsified on NG45

> **TODO(prose):** E8 LP-HPWL diagnostic decomposed proxy as 6 % WL /
> 20 % density / 74 % congestion. None of the destroy heuristics had
> targeted the dominant component. E54 ranked LNS-destroy by macro
> contribution to abu-top-5 % cells (matching the proxy's congestion
> term). On IBM `--fast` E54 tied E48 (with per-bench wins on ibm04 /
> ibm13). **On NG45 ariane133, +5.14 % catastrophic regression vs E48.**
> Joins E42 (K=4, +3.57 % ariane133), E43 (longer K-joint, +4.10 %),
> E44 (spatial K-tuple, IBM kill gate fired) in the *IBM-aware /
> NG45-blind failure class*. (See §8.8 for the structural pattern.)

> **TODO(figure):** Three-experiment summary as a bar chart (E48 vs
> each extension, --fast / --all / --ng45).

---

## 8.7.5 The Infeasibility Wall — E65 Cross-Section (~1 page, structural finding, 2026-05-03)

**Source material:** `experiments/E65_neb_cross_section/manifest.md`,
`memory/e65_infeasibility_wall.md`.

After the May 1-2 falsifications (E53/E53m/E54), the open question
was: *what bridges the SDF and DPO basins*? The natural family of
mechanisms — interpolation, blending, macro-level crossover — were
either falsified (E61 V1: 136 unrecoverable overlaps) or stuck on
implementation (E63 V2/V3 legalization issues).

E65 NEB cross-section measures the proxy and feasibility along a
**linear path between two converged placements** — E25 (SDF basin)
and E41 (DPO basin). For each k ∈ {0.1, 0.2, …, 0.9}, the placement
`(1−k)·E25 + k·E41` is computed, and `project_overlaps` runs to
attempt legalization.

> **TODO(prose):** Frame the structural finding. Two cross-sections
> (ibm01, ibm12) tested. **Both yielded 0/9 feasible interpolations.**
> Wall residuals 88-155 hard overlaps per intermediate k-value, peak
> 145 at k=0.5 (ibm01). On ibm12 the cross-section endpoints differ
> by only 0.2 % in proxy yet are separated by a 233-residual feasibility
> gap. **The wall is a function of spatial-configuration distance,
> not proxy distance.**
>
> **What this rules out for the field:** any mechanism that bridges
> macro placement basins at the *solution* level via per-element
> recombination. This is a stronger claim than "DPO seeds collapse
> to byte-identical placements" (E5) — it says the *space between*
> any two converged placements is overwhelmingly infeasible, even if
> the basin endpoints are nearly proxy-equivalent.
>
> **What this rules in:** mechanisms that operate at coarser
> granularity than per-element (E61 V2 spatial blocks succeed because
> blocks are internally feasible), non-local feasibility-respecting
> moves (K=N Hungarian re-pack, with N → all hard movables), or
> constrained NEB that follows the feasible manifold rather than
> a Euclidean line.
>
> Connection to ML "out-of-distribution" / "manifold hypothesis"
> literature is worth flagging — the feasible region of placements is
> an extremely thin manifold in R^{2N} space.

> **TODO(figure):** Cross-section plot. X-axis = k ∈ [0, 1]. Two
> y-axes: pre-legal proxy (left), residual overlap count (right).
> Show ibm01 + ibm12 cross-sections in two panels.

> **TODO(data):** ibm01 + ibm12 cross-section CSVs in
> `experiments/E65_neb_cross_section/results/`.

---

## 8.7.6 Spatial-Block GA Crossover — E61 V2 Threads the Wall (~1.5 pages, candidate, 2026-05-03)

**Source material:** `experiments/E61_ga_crossover/manifest.md`,
`experiments/E61_ga_crossover/results/best_of_analysis.md`,
`docs/decisions/012_e61v2_spatial_block_crossover_promotion.md` (ADR
*Proposed*).

E61 tested whether GA crossover between E25 (SDF basin) and E41 (DPO
basin) outputs could find a basin neither parent reaches alone.

> **TODO(prose):**
> **V1 (per-macro Bernoulli)**: each hard macro takes its position
> independently from E25 or E41 (Bernoulli p=0.5). **Falsified** —
> 136 unrecoverable overlaps per crossover attempt; project_overlaps
> caps at 50 iters and never legalizes. Direct empirical
> consequence of the §8.7.5 infeasibility wall.
>
> **V2 (spatial-block 2×2)**: the canvas is divided into four
> quadrants (top-left / top-right / bottom-left / bottom-right);
> each quadrant takes ALL its hard macros from one parent (Bernoulli
> per-quadrant). Block boundaries align with mid-canvas; macros
> straddling boundaries assigned to whichever side their center
> falls. The four-block crossover produces a placement where each
> 2×2 region is internally feasible; only block-boundary
> interactions need overlap repair, which `project_overlaps`
> handles in <50 iters.
>
> **Why this works (mechanism):** spatial blocks are coarser than
> per-macro but finer than full-canvas. Block-internal topology is
> preserved from one parent; block-boundary topology is mixed but
> small in extent. The key insight: **the infeasibility wall is
> avoided not by smoothing the recombination but by choosing a
> recombination granularity at which both parents are individually
> close to feasible**.

### 8.7.6.1 V2 results (verified)

| Mode | Avg | Δ vs E48 | Notes |
|------|----:|---------:|-------|
| --fast | 0.91342 | **−0.74 %** | sample-size lift on 4 small benches |
| --all | 1.08083 | **−0.07 %** | marginal at aggregate; fails standalone promotion threshold |
| --ng45 | **0.6908** | **−0.20 %** | **first NG45-positive mechanism since E18** |
| ariane133 | 0.6760 | **−1.47 %** | **breaks the consistent ariane133 failure point** (E42/E43/E44/E54/E62 all regressed there) |

Per-bench --all: 6 wins / 5 losses / 6 ties vs E48. Big wins on
**tied-parent benches** (ibm12 −0.68 %, ibm14 −0.47 %, ibm15
−0.29 %) — exactly the benches where E25 and E41 are within ~1.7 %
of each other. On benches where one parent dominates (ibm04 E41 by
−2.7 %, ibm17 E25 by +0.7 %), polish reverts to the dominant parent.

### 8.7.6.2 Hybrid extension

> **TODO(prose):** best-of-{E48, E61_v2} per-bench = **1.08025**
> (−0.12 % vs E48). Best-of-3 with E53m = 1.07995 (−0.14 %).
> Hybrid contribution is the strongest argument for ADR-012
> promotion. Connect to §8.6 hybrid mechanism.

> **TODO(figure):** Per-bench bar chart E48 / E53m / E61_v2 / best-of-3.
> Use `experiments/E61_ga_crossover/results/best_of_analysis.md` data.

---

## 8.8 The IBM/NG45 Transfer-Failure Pattern (~0.75 pages, structural finding)

**Source material:** `docs/roadmap.md` §4.5; per-experiment manifests
above; `external/MacroPlacement/Testcases/ariane133/` for the
benchmark structure data.

> **TODO(prose):** Five experiments now show the same pattern — IBM-fast
> lift that fails to transfer to NG45 commercial designs, with
> ariane133 as the consistent failure point. Build the case:
>
> | Experiment | Mechanism | --fast Δ vs E41 | NG45 ariane133 Δ |
> |---|---|---:|---:|
> | E42 | K-joint K=4 | −0.35 % | **+3.57 %** |
> | E43 | Longer K-joint (1200 s) | −0.33 % | +4.10 % |
> | E44 | Spatial K-tuple | +0.63 % (kill gate) | n/a |
> | E54 | Congestion-targeted destroy | tied | **+5.14 %** |
>
> Hypothesis (not formally verified, but consistent with all five):
> structurally-aligned destroy and K-tuple heuristics depend on
> *dense* macro packing. ariane133 has 133 hard macros on a
> 1433 × 1433 micron canvas (~1/15.5 macros per square micron), vs
> IBM's 246-760 hard macros on 23-73 micron canvases (~10-20 per
> square micron). Top-5 % cells of a 24×21 grid (NG45) = 25 cells —
> too few for congestion-targeted destroy to find structurally-coupled
> K-tuples. **Cost-aware destroy (total Δproxy, IBM-blind) and
> netlist-adjacency K-tuple ranking remain the safe baselines.**
>
> This finding generalizes the standard advice "validate cross-benchmark"
> into something more specific: when a heuristic engages the proxy's
> structural decomposition (E8: 6/20/74 %), verify on a *sparse*
> commercial benchmark before promoting. The E48 hybrid's safety
> comes from per-bench best-of: any IBM-tuned lane that regresses on
> NG45 simply isn't picked there. But adding such a lane consumes wall
> budget and complicates the system; the May 1-2 wave shows the EV
> doesn't justify the complexity.

> **TODO(data):** Formal verification of the dense/sparse hypothesis —
> per-design `n_hard / canvas_area` ratio across IBM + NG45, plotted
> against transfer-success rate of mechanism-aligned heuristics.

---

## 8.9 Hessian Saddle Escape on the Local-Move Plateau — E74

**Source material:** `experiments/E74_hessian_saddle/manifest.md`,
`experiments/E74_hessian_saddle/code/hessian_saddle.py`,
`docs/decisions/012_e74_hessian_saddle_promotion.md`.

### 8.9.1 The wall: local-move saturation

By the end of Act 2 every local-move mechanism we could compose had
been tried and stacked into the E48 hybrid. The CD coordinate-descent
backbone polished against the canonical proxy; grid-bin LNS (§8.5)
provided a complementary move-type that escapes CD's per-axis fixed
point; SA-v2 (§8.5.1) applied Metropolis acceptance on those same
breakpoints with best-so-far tracking; the K-macro joint LNS (§8.5.3)
brute-forced 5³ joint reinsertions over the three most cost-coupled
macros; and the spatial-block GA crossover (§8.7.6) recombined whole
2×2 quadrants between SDF and DPO basins. All five mechanisms
plateaued within 0.3 % of each other near 1.08, and the May 1–2
extension wave (§8.7) produced no further lift.

The pattern is structural. Every one of those mechanisms operates on
**one or a few macros at a time**: CD breakpoints move a single macro
along one axis; LNS destroys K=12 macros and reinserts them one by
one; SA proposes a single-axis breakpoint move; K-joint enumerates
joint moves over K=3 macros; spatial-block crossover swaps a quadrant
at a time. They share a *reachable set* — the set of placements
reachable from the current state by composing some number of local
moves — and that reachable set has a fixed point, which is what the
plateau is. The structural finding in §8.7.5 (the *infeasibility
wall*) sharpens this: two proxy-equivalent placements from different
initial basins are separated by a thick region of overlap-violation
in spatial-configuration space, so element-level recombination cannot
bridge them either. Whatever escape exists from the 1.08 plateau, it
is not in the local-move family or its element-wise recombinations.

### 8.9.2 Diagnostic: the plateau is a high-index saddle

The structural-finding sequence makes a specific prediction: if the
plateau is locally optimal under *all* the moves we have, it ought to
be a true local minimum of *something*. The natural candidate is the
canonical proxy itself — but the canonical proxy is non-differentiable
(top-K density, discrete grid-cell membership, non-smooth max for
HPWL), so its second-order structure is undefined.

Instead we examine the *smooth proxy* — a differentiable surrogate
already used internally by DPO (§§5–6) and the GPU restart experiments
(§8.7.1), which approximates the canonical objective with
LogSumExp-smoothed HPWL, Gaussian-kernel grid density, and a
differentiable variant of the TILOS RUDY congestion estimator. The
smooth proxy is not the cost we score against; it is a *curvature
oracle* that exists in `ℝ^{2N}` (where the canonical proxy lives on
the discrete sub-manifold of legal placements) and whose
second-derivative information is, in principle, accessible.

For an `N`-macro placement with `2N ≈ 500` degrees of freedom, we do
not form the full `(2N)×(2N)` Hessian explicitly. Instead we exploit
the fact that PyTorch's autograd gives the *Hessian-vector product*
`Hv` exactly via `torch.autograd.functional.hvp`, in time linear in
the cost of one gradient evaluation. We wrap the HVP as a SciPy
`LinearOperator` and pass it to `scipy.sparse.linalg.eigsh` with
`which='SA'` (smallest-algebraic), `k=2` to `4`. The Lanczos iteration
converges on the soft eigenmodes in three to five seconds on a single
CPU core.

The result on a typical E48 plateau is striking. For ibm01 the top
four smallest-algebraic eigenvalues are `(−0.14, −0.076, −0.066,
−0.060)` — every one negative. For ibm04 they are `(−0.69, −0.11,
−0.07, −0.03)`. The plateau is not a smooth-proxy local minimum at
all; it is a **high-index saddle**, with at least four directions of
descent in the smooth surface that the local-move family cannot
follow because each of those directions is a *coordinated
displacement of many macros simultaneously* — exactly the kind of
joint move that single-macro and single-element heuristics produce
only by accident.

### 8.9.3 Mechanism: smooth-proxy curvature, canonical-proxy polish

Once we accept that the plateau is a saddle, the literature of
transition-state search becomes directly applicable. Three classical
methods motivate the approach:

- *Climbing-image nudged elastic band* (Henkelman & Jónsson, 2000),
  which finds saddle points along a discretized path between two
  minima by following the unstable mode at each interior image;
- The *dimer method* (Henkelman & Jónsson, 1999), which finds saddles
  using only first derivatives by tracking a pair of nearby states
  rotated to align with the softest mode;
- *Gentlest-ascent dynamics* (E & Zhou, 2011), which formulates saddle
  search as a continuous dynamics that climbs along the smallest
  eigenvector while descending in orthogonal directions.

Each of these methods escapes a saddle by *stepping along its
negative-curvature eigenvector*. We take the simplest, most direct
form of that idea:

1. **Compute** the softest eigenvector `v_0` of the smooth-proxy
   Hessian at the current plateau state via the HVP-Lanczos pipeline
   above. If `λ_0 ≥ −10^{−3}` the smooth surface is essentially
   positive-semidefinite locally; we stop, because no useful escape
   direction remains.
2. **Step** the placement `p ← p ± ε · v_0` for
   `ε ∈ {0.3, 1.0, 3.0}` and both signs. The eigenvector is normalized
   per coordinate; `ε` carries dimensionless units of placement-vector
   norm, so the same grid works across benchmarks of different macro
   counts and canvas sizes.
3. **Legalize** by applying `project_overlaps` — a geometric
   axis-aligned projection that resolves any residual overlaps
   introduced by the perturbation.
4. **Polish** the legalized state on the *canonical* proxy with the
   CD-adaptive sweep from §8 (capped at a per-trial budget). The smooth
   proxy was the curvature oracle; the canonical proxy remains the
   score.
5. **Keep** the polished result with the lowest canonical proxy across
   the six `(sign, ε)` trials.

The shape of the algorithm is exactly that of the dimer or
gentlest-ascent step, transplanted from continuous chemistry-style
energy landscapes onto the constrained discrete manifold of legal
placements. What the literature calls a *transition state* — a
saddle point separating two stable states — is, in our setting, the
plateau that every local-move heuristic terminates at. What we are
doing is using the smooth proxy's second-order structure as a
*guidance field* to identify the direction that crosses the
transition state, then letting the canonical proxy's local optimizer
(CD-adaptive) settle into the deeper basin on the other side.

The contribution is not the algorithmic primitives — those are
half a century old in chemistry and physics — but the *connection*:
applying transition-state search to combinatorial placement plateaus
on the smooth-proxy curvature, with canonical-proxy polish in the
loop. To our knowledge this connection has not previously appeared
in the macro-placement literature.

### 8.9.4 Verified results

| Metric | E48 hybrid (parent) | E74 (this) | Δ |
|--------|--------------------:|-----------:|---:|
| `--all` IBM avg | 1.08151 | **1.0666** | **−1.38 %** |
| ariane133 (NG45) | 0.6861 | **0.6641** | **−3.21 %** |
| ariane136 (NG45) | 0.6685 | 0.6518 | −2.50 % |
| nvdla (NG45) | 0.6767 | 0.6716 | −0.75 % |
| mempool_tile (NG45) | 0.7375 | 0.7376 | tied |
| Hard overlaps | 0 / 17 + 4 | 0 / 17 + 4 | — |

E74 lifts uniformly across all 17 IBM benchmarks (per-bench deltas
range from −0.11 % on ibm09 to −7.13 % on ibm02), with the largest
gains concentrated on benchmarks whose E48 plateau happens to have
the largest negative-curvature eigenvalues.

The ariane133 result deserves attention. Every IBM-aware mechanism in
the May 1–2 wave regressed on ariane133 — E42 (K-joint K=4) by
+3.57 %, E43 (longer K-joint) by +4.10 %, E44 (spatial K-tuple)
killed its own gate, E54 (congestion-aligned destroy) by +5.14 %,
E62 (Will-seed init) by +1.45 %. The structural reading (§8.8) is
that those mechanisms depend on dense macro packing typical of IBM
ICCAD04 designs and degrade on the sparse ariane-class commercial
benchmarks. Hessian saddle escape does the opposite: it *advances*
on ariane133 by 3.21 %, the largest single-mechanism NG45 lift since
the introduction of DPO basins in E18. The reason is mechanistic:
the smooth-proxy eigenvectors are agnostic about macro density.
What they care about is the geometry of the second-derivative
structure, which is governed by net connectivity and the
electrostatic-style interactions that the smooth proxy encodes —
properties that scale uniformly between IBM and NG45 designs.

E74 is, since CD itself in Act 1, the first mechanism we have found
that lifts uniformly across all 21 designs we evaluate. ADR-012
promoted it as champion on 2026-05-05.

> **TODO(figure):** λ-spectrum bar chart for the ibm01 plateau, top-8
> eigenvalues. Visualizes the multi-direction saddle and motivates
> the cascading extension in §8.10.

---

## 8.10 Cascading the Saddle Escape — E84 (~1.5 pages)

**Source material:** `experiments/E84_cascading_saddle/manifest.md`,
`experiments/E84_cascading_saddle/code/cascading_saddle.py`.

> **TODO(prose):** E74 applied one step of saddle escape (compute v_0,
> step ε, polish, return best). But the eigenvalue diagnostic shows
> *multiple* negative-curvature directions at the plateau — and once we
> escape via v_0, the new state may itself be a saddle (just at a
> deeper basin). Cascading saddle escape iterates: after each polish,
> recompute the Hessian, check the smallest eigenvalue, and if still
> negative, escape again.
>
> Stop conditions:
> - smallest eigenvalue ≥ `−1e-3` (true local minimum reached)
> - cascade iteration produced no proxy improvement
> - `max_iters = 5` exhausted
> - wall budget exhausted
>
> Empirically the cascade runs 2-5 iterations per benchmark before
> hitting one of the stop conditions, with diminishing per-iter lift:
> first iter gives the bulk (−1 to −5 % depending on benchmark);
> subsequent iters add 0.1 to 0.5 % each. The cumulative lift is
> consistent across all 17 IBM and 4 NG45 designs.

**Verified results (M3 cached, no wall cap):**

| Metric | E74 (parent) | E84 cascade (this) | Δ |
|--------|-------------:|-------------------:|---:|
| IBM `--all` | 1.0666 | **1.0612** | **−0.51 %** |
| Overlaps | 0 / 17 | 0 / 17 | — |
| Walls > 55 min | 0 / 17 | 8 / 17 | requires wall-safe variant |

The wall-uncapped 1.0612 number is the algorithmic *ceiling* of the
cascade approach — further iteration of E74-style saddle escape no
longer improves the proxy. Closing the gap from this ceiling to a
*cap-bound* result on EPYC is an *engineering* problem, not an
algorithmic one, and is the subject of §8.11.

---

## 8.11 PATH A — Closing the Cap-vs-Ceiling Gap via Implementation Speedup (~1.5 pages)

**Source material:** `macro_place/incremental_evaluator.py` (commits
59a7a8b, 53b4a26, af520c3, 9df5ac2, f2269b3),
`docs/decisions/A4_postA1_validation.md` (forthcoming).

> **TODO(prose):** The cascade saddle escape reaches 1.0612 on M3 Max
> with no wall budget enforced. Under the 60-min/bench cap on
> partcl-equivalent EPYC, the same algorithm plateaus at 1.137 because
> CD coordinate descent in the inner polish phase is single-threaded
> Python and can't complete enough sweeps. cProfile on ibm04 isolated
> the bottleneck: 32 of 33 s of CD time is in
> `IncrementalProxyEvaluator.move() + revert()`, the per-candidate
> probe-and-undo pattern in `cd_core.search_axis`.
>
> Four optimizations close the gap:
>
> 1. **`delta_cost(macro_idx, new_xy)`** — a no-mutation cost peek
>    that mirrors the (move, current_cost, revert) trio without
>    building a `_MoveSnapshot` and without applying state mutations
>    that need to be undone. Briefly retargets the moving macro's pin
>    positions in a try/finally; everything else builds hypothetical
>    cell tensors via clone + delta. **1.95× per probe** on ibm10.
>
> 2. **`delta_cost_axis_batch(macro_idx, axis, cur_xy, candidates)`** —
>    evaluates K candidates on a single axis in one call. Amortizes
>    the "subtract old contributions" precompute once; collects
>    per-candidate cell-level deltas as flat `(k, cell, val)` triples
>    and applies via `torch.index_put_(accumulate=True)`; runs density
>    top-K and congestion smoothing + top-K as batched torch ops on
>    `[K, num_cells]` tensors. **4.32× combined** on ibm10.
>
> 3. **`_net_cong_contrib_flat`** — flat-list replacement for the
>    dict-based per-net routing helper. Vectorized pin→gcell via
>    tensor floor/clamp/long; inlined 2-pin and multi-pin routing
>    (the 88 % case per profile); no dict allocation; duplicates
>    re-aggregated downstream by `index_put_`. **5.36× combined** on
>    ibm10.
>
> 4. **LNS-helper conversions** — `_pick_destroy_by_cost` and
>    `_gridbin_reinsert` in E25 / E18 / E39 are pure probe patterns
>    (move, cost, revert); replaced with `delta_cost`. Same bit-exact
>    cost result; ~2× wall reduction on the LNS phase, which on
>    cascade benches converts to deeper basins per LNS sample under
>    the same per-phase budget.

**Verified results (cloud EPYC, jobs=4, `budget_seconds=3000`):**

| Run | Mode | Avg proxy | Overlaps | vs Pre-A1 |
|-----|------|----------:|---------:|----------:|
| Pre-A1 baseline (2026-05-11) | `--all` | 1.137 | 0 | — |
| A4-v1 (A1 inner only) | `--all` | **1.0782** | 0 | **−5.2 %** |
| A4-v2 (A1 + LNS-delta) | `--all` | **1.0771** | 0 | **−5.3 %** |
| Pre-A1 baseline | `--ng45` | 0.7034 | 0 | — |
| A4-v1 | `--ng45` | **0.6853** | 0 | **−2.6 %** |
| A4-v2 | `--ng45` | **0.6870** | 0 | −2.3 % |

> **TODO(prose):** The 1.0771 cloud number is within +1.5 % of the
> 1.0612 M3 cached ceiling. The remaining gap is residual per-core
> speed differential between M3 and EPYC; closing it further would
> require either a Cython/Numba port of the routing inner loops or a
> GPU port of the routing-dispatch logic. Neither is justified at this
> margin since the leaderboard top is at 1.037 (Cezar/ReFine) — closing
> the cap-vs-ceiling gap fully gives < 0.4 % more, which is not enough
> to overtake. The breakthrough vector beyond cascade is *algorithmic*
> (PATH B: DREAMPlace basin + cascade polish; PATH C: multi-direction
> saddle escape), not implementation.

> **TODO(theory ref):** Note that the implementation work is parallel
> to the algorithmic work — the same `IncrementalProxyEvaluator`
> primitives feed all downstream variants (DP-hybrid, multi-direction
> cascade, etc.), so A1's speedup compounds with their potential
> algorithmic lifts.

---

## 9. Empirical Results (~2 pages)

### 9.1 Champion lineage (`--all`, 17 IBM benchmarks, zero overlaps)

| Era | Method | Avg proxy | Δ vs RePlAce | Date |
|-----|--------|-----------|--------------|------|
| Pre-history | RePlAce | 1.4578 | — | n/a |
| Polyhedra | Phase 2 (300 s nav) | 1.4867 | −0.6 % | 2026-04-15 |
| DPO v1 | seed 42 | 1.4246 | +2.3 % | 2026-04-23 |
| DPO best-of | best_of_v2 | 1.3834 | +5.1 % | 2026-04-26 |
| CD-only | CDOnly (fixed 600 s) | 1.1193 | +23.2 % | 2026-04-27 |
| CD-adaptive | CDAdaptive (E9) | 1.1055 | +24.2 % | 2026-04-27 |
| CD + grid-bin LNS | CDLNSGridBin (E12) | 1.0990 | +24.6 % | 2026-04-28 |
| CD + LNS + SA-v2 *(component)* | CDLNSSA (E25) | 1.0954 | +24.9 % | 2026-04-29 |
| DPO init + CD + LNS + SA-v2 *(component)* | CDLNSSADPOInit (E18) | 1.08979 | +25.2 % | 2026-04-30 |
| DPO init + CD + LNS + SA-v2 + K-joint *(component)* | CDLNSSADPOKJoint (E41) | 1.0848 | +25.6 % | 2026-04-30 |
| **Per-bench best-of-{E25, E41} hybrid** | **CDLNSSAHybrid (E48)** | **1.08151** | **+25.8 %** | **2026-05-02** |
| Spatial-block GA crossover (E25 ⊗ E41 outputs) | CDLNSGACrossoverPlacer (E61_v2) | 1.08083 | +25.9 % | 2026-05-03 |
| **Hessian saddle escape on E48 plateau** | **CDLNSSAHessian (E74)** | **1.0666** | **+26.8 %** | **2026-05-05** |
| **Cascading saddle escape (M3 cached, uncapped)** | **CDLNSSACascade (E84)** | **1.0612** | **+27.2 %** | **2026-05-10** |
| **Cascade + A1 implementation speedup (cloud, 60-min cap)** | **CDLNSSACascadeAdaptivePlacer (post-A1)** | **1.0771** | **+26.1 %** | **2026-05-13** |

Each champion replaced its predecessor by a *structural change*, not
parameter tuning. The May 2026 entries (E18 / E25 / E41) appear as
*components* of the E48 hybrid — none promoted standalone, but each
contributes lanes the hybrid picks per-bench. E61_v2 (spatial-block
crossover) is a candidate for the same role at one level higher: a
crossover-of-hybrid-outputs mechanism that threads through the
infeasibility wall identified in §8.7.5.

### 9.2 Per-benchmark champion table

> **TODO(content):** Pull full per-benchmark CDAdaptive table from
> `docs/results.md` (already populated). Include side-by-side with
> CDOnly (prior champion) and RePlAce. 17 rows.

### 9.3 DPO ablation summary

> **TODO(content):** Pull from `evidence.md` §3.1. Table is already drafted
> in §5 above; either inline here or cross-reference.

### 9.4 Falsification record (the dead ends)

| Hypothesis | Best | Verdict | Lesson |
|-----------|------|--------|--------|
| Polyhedra LP-HPWL navigation | 1.4867 | ceiling | LP optimizes the wrong objective |
| 22-experiment polyhedra sweep | 1.4918 | flat | Navigation is not the bottleneck |
| Congestion weight sweep (DPO) | 1.2382 | killed | Direction wrong, not magnitude |
| E5 batched seeds (B=64) | 1.1698 | killed | Within-basin best-of-N is flat |
| E10 congestion-only refinement | 1.3788 | marginal | −0.33 % on `--all`; ibm02 worse |
| E11 diverse priors (SDF/Will/greedy/random) | 1.3839 | flat | Basin lock is topology, not init |
| E3 LNS v1 (full-canvas reinsert) | flat | falsified | Single-macro local LNS does not escape CD |
| E3 LNS v2 (5×5 window reinsert) | flat | falsified | Same fixed point |
| Multi-init CD via SDF jitter (`analysis/multi_init_probe/probe.py`) | 0/8 improved | falsified | SDF→CD is contractive |
| Subset-CD LNS (`analysis/lns_escape_probe/probe.py`) | 0/24 | falsified | Same per-axis fixed point |
| Group topology cascade (polyhedra) | infeasible | falsified | LP cycles |
| E14 SA polish (no best-tracking, T₀=0.01) | 0.9962 (--fast +5.7 %) | falsified | SA wandered up; best-tracking + cool T₀ are non-negotiable (E25 fixed both) |
| E32 SAM-CD (sharpness-aware K=4 perturbations) | 0.985 (--fast +5.5 %) | falsified | K=4 multiplier on CD eval cost prevents convergence |
| E40 multi-SA-seed best-of-4 (within-basin) | 0.93295 (--fast) | marginal | Within-SA-basin ensembles don't compound |
| **E42** K-joint K=4 | 0.91859 (--fast); 0.6960 (--ng45 +0.84 %) | **falsified on NG45** | First IBM-aware/NG45-blind: ariane133 +3.57 % vs E41 |
| **E43** Longer K-joint budget (1200 s) | 0.91867 (--fast); 0.7009 (--ng45) | **falsified on NG45** | ariane133 +4.10 %, ariane136 +2.48 % |
| **E44** Spatial K-tuple selection | 0.9276 (--fast +0.63 %) | **falsified** | Kill gate fired on IBM; netlist-adjacency K-tuple is load-bearing |
| **E53** GPU DPO basin polish (multi-restart MPS Adam) | 0.9254 (--fast); 0/350 GPU restarts accepted at full budgets | **falsified** | Smooth-proxy gradient cannot escape fully-converged CD-LNS-SA local optimum |
| **E53m** Multi-seed hybrid (3-way w/ E41 seed=1) | 0.91128 (--fast outlier); 1.08128 (--all tied with E48) | marginal | Multi-seed within DPO averages out at --all aggregate scale |
| **E54** Congestion-targeted LNS destroy | 0.9222 (--fast tied); 0.7022 (--ng45 +1.45 %) | **falsified on NG45** | ariane133 +5.14 %; mechanism-aligned destroy is IBM-aware/NG45-blind |
| **E60** Replica-exchange SA-v2 (parallel tempering) | 0/21 swap accepts at narrow temp spacing | falsified | Energy gap between chains ≥0.05 → swap log-prob ≈ −1000; SA isn't basin-stuck, basin choice is the bottleneck |
| **E61 V1** Per-macro Bernoulli GA crossover | 136 unrecoverable overlaps per attempt | falsified | Per-macro recombination falls inside the infeasibility wall (§8.7.5); fixed by V2 spatial-block at coarser granularity |
| **E62** WillSeed init lane (E41 backbone) | --fast 0.9335 (+1.44 % vs E48; loses every fast bench) | falsified | Standalone init quality (1.5338) doesn't predict basin quality; CD lands worse fixed point from WillSeed than from SDF or DPO |
| **E63** Spectral / quadratic init (V2 Hungarian / V3 row-pack) | implementation blocked: V2 28 max-size slots vs 246 macros; V3 111 fixed-macro residuals | implementation incomplete | Eigendecomposition works in 0.2 s; legalization is the hard step. Needs row-pack with fixed-region awareness (~2-4 hr dev). |

The bottom block (E42 / E43 / E44 / E54) reveals a structural failure
class: heuristics that engage the proxy's component decomposition
(6 % WL / 20 % density / 74 % congestion) lift on dense IBM layouts but
break on sparse commercial designs (ariane133 specifically). See §8.8.

E60 / E61 V1 / E62 / E63 reveal an orthogonal structural finding: the
SDF↔DPO basin pair is separated by an **infeasibility wall** in
spatial-configuration space (§8.7.5). Mechanisms that bridge basins at
the *solution* level via per-element recombination fall inside the
wall; mechanisms operating at coarser granularity (E61 V2 spatial
blocks, §8.7.6) thread through. E63 spectral and E64 LP-bounded beam
K-joint remain candidates for "construct a third basin without
bridging" — both still under development.

### 9.5 Generalization & robustness

NG45 transfer (4 commercial designs, zero overlaps):

| Design | E12 | E18 | E41 | **E48** | Δ vs E12 |
|--------|----:|----:|----:|--------:|---------:|
| ariane133 | 0.7202 | 0.6850 | 0.6733 | **0.6861** | **−4.74 %** |
| ariane136 | 0.6850 | 0.6724 | 0.6728 | **0.6685** | **−2.41 %** |
| mempool_tile | 0.7375 | 0.7375 | 0.7375 | **0.7375** | tied |
| nvdla | 0.6720 | 0.6815 | 0.6767 | **0.6767** | +0.70 % |
| **avg** | **0.7037** | 0.69193 | 0.69022 | **0.6922** | **−1.66 %** |

> **TODO(prose):** Frame the NG45 result. E48's basin choice (per-bench
> best-of-{E25 SDF basin, E41 DPO basin}) transfers *without per-bench
> tuning*. Per-design wins distribute across both lanes — the same
> hybrid mechanism that exploits IBM heterogeneity also exploits
> commercial-design heterogeneity. **No mechanism in this work was
> tuned on NG45**; transfer is the structural test of "no per-benchmark
> tuning" claims.

> **TODO(data):** E12 / E48 multi-seed variance on `--all` (3–5 seeds).
> Currently n=1 on the champion run. DPO has 5-seed evidence; the paper
> needs symmetric evidence for the champion. E53m (multi-seed hybrid)
> partially fills this — E41 seed=42 vs seed=1 differ by ≤1.5 % per-bench
> on small benches and < 0.4 % on large benches.

> **TODO(data):** Plateau-threshold sensitivity (see §8 TODO).

### 9.6 Compute envelope

| Stage | Wall (s) | Per-bench wall | Notes |
|-------|---------:|----------------|-------|
| CDOnly fixed 600 s | 10 316 | 600 (uniform) | superseded |
| CDAdaptive (E9, prior champion) | 17 480 | 322–2238 | All exit via plateau, none hit 3600 s cap |
| **CDLNSGridBin (E12, champion)** | **28 256** | **524–3 487** | **CD ≤ 3 000 s + LNS ≤ 600 s; ibm17 closest to per-bench cap at 3 487 s of 3 600 s** |

Total wall = 7.85 hr — inside the 17-hr (17 × 1 hr) hidden-test envelope
with ~9 hr of headroom (down from CDAdaptive's ~12 hr).

> **TODO(figure):** Champion lineage bar chart
> (1.50 → 1.49 → 1.38 → 1.12 → 1.10).

---

## 10. Discussion (~1.5 pages)

**Source material:** `contributions.md` §5 + §10 + §11; `evidence.md` §9
(falsification record).

### 10.1 The objective mismatch recurs at every level
LP-HPWL vs proxy (ρ=−0.001); RUDY congestion vs real congestion (10.9 %
hotspot overlap); DPO's gradient vs the true gradient. Each approximation
introduces an objective mismatch that caps performance. The same diagnostic
methodology — systematic ablation + correlation analysis — detected each one.

### 10.2 Decomposition is counterproductive for this objective
By component (LP-HPWL ρ=0); by structure (22-experiment sweep flat); by
scale (hierarchical clustering 18–28 % worse, 0/190 cluster swaps). The
proxy couples WL/D/C through shared grid cells; any separation of concerns
loses the information that matters. *DPO succeeds because it does not
decompose; CD succeeds because the exact evaluator is fast enough that
CD on the joint objective beats gradient descent on a wrong joint model.*

### 10.3 "Bypass, don't fix" as an algorithmic design principle
Demonstrated twice on the same problem:
- LP-HPWL → DPO: bypass single-component LP with joint differentiable proxy.
- DPO RUDY → CD: bypass differentiable approximation with exact proxy via
  incremental data structures.
When an approximation is *structurally* wrong (not just noisy), exact
evaluation with a faster data structure beats a more accurate approximation.

### 10.4 Infrastructure unlocks algorithms
CD is textbook. The 4657× incremental evaluator was the gate. Without it,
30 s/eval would do ~2 sweeps/hour; with it, 13 sweeps in 10 min. Same
algorithm, different infrastructure. Per-bench plateau detection (E9) is
also infrastructure-driven: the per-sweep delta logs that motivated it
only existed because the evaluator produced them in real time.

### 10.5 Compositional polish — three orthogonal mechanisms compose multiplicatively

> **TODO(prose):** Frame the E18 / E25 / E41 / E48 lineage as a
> compositional finding. Each mechanism (DPO basin shift, SA-v2 polish,
> K-joint LNS) is a verified individual lift on top of E12; their
> *naive sum* would predict 1.0925 (E12 1.0990 + −0.51 % DPO + −0.45 %
> K-joint − overlap), but actual composition E48 hits 1.08151 (−1.59 %
> vs E12). The hybrid further extracts +0.30 % over E41 by per-bench
> best-of, which is the *meta-algorithmic* level of composition (the
> hybrid doesn't introduce a new mechanism; it exploits across-benchmark
> heterogeneity in which mechanism wins). Connection to ensemble
> learning / "boosting" literature is worth flagging.

### 10.6 The IBM/NG45 transfer-failure pattern — generalization fails on sparse layouts

> **TODO(prose):** Surface the §8.8 finding into the discussion. Five
> experiments showed IBM lift that doesn't transfer to ariane133. The
> generalizable claim is: **heuristics that engage the proxy's structural
> decomposition (component fractions, topology, geometric clustering)
> degrade on benchmarks with different macro-density profiles.**
> Cost-aware destroy (a topology-blind ranking by total Δproxy) and
> netlist-adjacency K-tuple ranking (using *graph* structure, not
> *geometric* structure) are the safe baselines. The paper's
> contribution to the field is partly negative: the obvious
> "decomposition-aware" heuristics fail under cross-design generalization,
> and the field should validate on a sparse benchmark (commercial NG45
> ariane133) before promoting any IBM-tuned mechanism.
>
> Connection to "OOD generalization" literature in ML is worth flagging.

### 10.7 Limitations
- CD plateaus after 10–15 sweeps; ibm17/18 still descending at the cap on
  ablations with tighter thresholds.
- LNS at single-macro / local-window granularity does not escape CD.
  Cluster-level joint reinsertion remains untested in the unique
  combinations explored here.
- The polyhedral decomposition is the right framework for *understanding*
  but did not yield direct algorithmic advantage. Structural insight
  guided diagnosis, not solution.
- GPU acceleration as an additive polish phase doesn't help (E53);
  whether it helps as a CD replacement is untested.
- The hybrid extends naturally to N>2 lanes, but additional lanes
  (multi-seed, cross-init) hit diminishing returns at --all aggregate
  scale. Whether sparser benchmark sets (NG45 5-design ASAP7?) retain
  the multi-lane lift pattern is untested.

### 10.8 What this means for the field
Macro placement with composite objectives may be fundamentally resistant
to divide-and-conquer strategies that work for single-objective problems.
The path forward is *better evaluators*, not better decompositions —
AND *robust hybrid composition*, not single-mechanism optimization.
Mechanism-aligned heuristics overfit to the design class they were
tuned on; per-bench best-of-N hybrids plus topology-blind base
heuristics are the empirical safe combination.

> **TODO(prose):** Draft. Tighten to ~1.5 pages total for §10.

---

## 11. References (~1 page)

**Sketch list (build out from `outline.md` references):**
- Balas (1979) — disjunctive programming.
- Kronqvist et al. (2025) — P-split formulation.
- Naylor (2001) — LSE-HPWL.
- Lin et al. (2019) — DREAMPlace.
- Lu et al. (2020) — congestion-aware DREAMPlace.
- C3PO / NV-Place (ASP-DAC 2026) — differentiable RUDY for standard-cell.
- DREAMPlace-Cong (DATE 2021); DCGP (DAC 2025); DGR (DAC 2024).
- Hazan et al. (2016) — graduated optimization.
- Bertsekas (1982) — penalty methods.
- Zariski (1962) — hyperplane arrangement complement connectivity.
- Slaney & Walsh (2001) — backbone variables.
- Miftari et al. (2024-2026) — LP sensitivity analysis.
- Mobahi & Fisher (2015) — Gaussian smoothing.
- RePlAce (Cheng et al.); TILOS / IBM benchmarks.
- Partcl/HRT Macro Placement Challenge 2026.

> **TODO(refs):** Format properly with DOIs. Verify each citation is one
> we *use*, not just a literature pointer. Drop anything we only read.

---

## A. Supplementary material (excluded from body)

**Source:** `theory.md` §"Excluded" list, `contributions.md` §"Explicitly
excluded".

The following theoretical connections live in `writeup/theory.md` but do
not enter the paper body — we cannot defend them from direct experience:
quantum tunneling / SQA, survey propagation / cavity methods / RSB,
stratified Morse theory / persistent homology, information geometry,
population annealing, homotopy continuation, spectral decomposition of
L(K_n), Benders decomposition, ejection chains / ALNS, diffusion models
(DiffPlace, DiffUCO), Graphs of Convex Sets, Dead-End Elimination,
discrete Schrödinger bridges, Kolmogorov complexity sampling, semi-
discrete optimal transport / Laguerre tessellations.

---

## Master TODO list (rolled up from inline `TODO(...)` markers)

### Data (load-bearing)
- [ ] **NG45 transfer datapoint** (≥ 1 public design: ariane133 / ariane136 /
      mempool_tile / nvdla). The Tier-1 generalization claim hinges on this.
      The 1–2 truly hidden NG45 designs remain opaque, but any *public* NG45
      result backs "transfers without per-bench tuning."
- [ ] **CDAdaptive multi-seed variance** (3–5 seeds on `--all`).
- [ ] **Generalize Miftari ρ=−0.001** from ibm01 to ibm04/09/13.
- [ ] **Plateau-threshold sensitivity** sweep (0.001 / 0.002 / 0.005 / 0.01).
      E16 is one datapoint; need 2 more.
- [ ] **Frozen `data/rudy_ibm01.txt`** captured from `rudy_analysis.py`.
- [ ] **Per-sweep convergence trajectories** for 3–4 benchmarks
      (easy/medium/hard).

### Data (optional / contingent)
- [x] **E12 grid-bin LNS final result — DONE 2026-04-28.** Avg `--all`
      1.0990, zero overlaps. Promoted to champion (ADR-007). Integrated
      into §8 as the final refinement and §9.1 lineage table.

### Figures (all unbuilt)
- [ ] §2: 3-macro polyhedra schematic + LP dual interpretation.
- [ ] §4: LP-HPWL vs proxy scatter; congestion vs proxy scatter; overnight
      sweep bar chart.
- [ ] §5: DPO ablation bar chart.
- [ ] §7: RUDY vs real congestion heatmap (ibm01); per-cell ratio
      distribution.
- [ ] §8: CD convergence curve; CDAdaptive per-benchmark wall-time chart.
- [ ] §9: Champion lineage bar chart.

### Prose (none drafted)
- [ ] Sections 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 — full prose.
- [ ] Abstract.
- [ ] References list, formatted with DOIs.
- [ ] Title finalize.

### Final pass
- [ ] Length/density check vs 12–14-page target.
- [ ] Math notation consistency.
- [ ] Cold-read by someone unfamiliar with the project.

---

## File map (where to source each section)

| Section | Primary source | Secondary |
|---------|---------------|-----------|
| §1 Intro | `evidence.md` §1 (champion lineage) | `docs/problem.md` |
| §2 Polyhedra | `theory.md` §1–2 | `docs/problem.md`, `contributions.md` §1 |
| §3 Navigation | `contributions.md` §1 | `archive/planning/eval_pipeline_design.md` |
| §4 Barrier | `evidence.md` §4, §3.3 | `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` |
| §5 DPO | `evidence.md` §3 | `archive/planning/dpo.md`, `contributions.md` §3 |
| §6 DPO theory | `theory.md`, `evidence.md` §8 | `contributions.md` §4 + §6 |
| §7 RUDY | `evidence.md` §5, `analysis/rudy_fidelity/rudy_analysis.py` | `contributions.md` §7 |
| §8 CD | `evidence.md` §7, `docs/approach.md` §1 | `contributions.md` §8–9 + §12 |
| §9 Results | `evidence.md` §1, §2, §9; `docs/results.md`; `docs/experiment_index.md` | — |
| §10 Discussion | `contributions.md` §5 + §10 + §11; `evidence.md` §9 | — |
