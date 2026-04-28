# Macro Placement: A Diagnostic-Driven Path from 1.46 to 1.10
<!-- TODO(meta): the headline 1.46 -> 1.10 still rounds correctly to 1.0990; consider whether to update to "1.46 to 1.10 (1.0990)" once the abstract is drafted. -->


> **STATUS:** Working master draft. Source-of-truth for the paper. Section
> scaffolding + locked-in numbers; prose drafts are TODO. Cite the supporting
> docs (`evidence.md`, `contributions.md`, `theory.md`) but do not duplicate
> them — write fresh prose from those sources here.

> **TODO(meta):** Title, authors, affiliation, abstract, venue/format target.
> Current placeholder title above. Length target: 12–14 pages.

---

## Abstract

> **TODO(prose):** ~200 words. Hits: composite proxy (WL + 0.5·D + 0.5·C),
> two diagnosis-pivot cycles, "bypass don't fix" principle, infrastructure-
> unlocks-algorithm twice (E1 and E9), grid-bin LNS overlay (E12) escapes
> the CD plateau via a *different move type*, final **1.0990** avg on 17
> IBM benchmarks (−24.6 % vs RePlAce 1.4578, −1.63 % vs leaderboard
> 1.1172), zero overlaps, within 1-hour-per-bench compute envelope.

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

## 9. Empirical Results (~2 pages)

### 9.1 Champion lineage (`--all`, 17 IBM benchmarks, zero overlaps)

| Era | Method | Avg proxy | Δ vs RePlAce | Date |
|-----|--------|-----------|--------------|------|
| Pre-history | RePlAce | 1.4578 | — | n/a |
| Polyhedra | Phase 2 (300 s nav) | 1.4867 | −0.6 % | 2026-04-15 |
| DPO v1 | seed 42 | 1.4246 | +2.3 % | 2026-04-23 |
| DPO best-of | best_of_v2 | 1.3834 | +5.1 % | 2026-04-26 |
| CD-only | CDOnly (fixed 600 s) | 1.1193 | +23.2 % | 2026-04-27 |
| CD-adaptive | CDAdaptive (E9, prior champion) | 1.1055 | +24.2 % | 2026-04-27 |
| **CD + grid-bin LNS** | **CDLNSGridBin (E12)** | **1.0990** | **+24.6 %** | **2026-04-28** |

Each champion replaced its predecessor by a *structural change*, not
parameter tuning.

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

### 9.5 Generalization & robustness

> **TODO(data, important):** NG45 transfer table. At minimum one public
> NG45 design (ariane133 / ariane136 / mempool_tile / nvdla). Without
> this, the "transfers without per-bench tuning" claim is unbacked.

> **TODO(data):** CDAdaptive multi-seed variance on `--all` (3–5 seeds).
> Currently n=1 on the champion run. DPO has 5-seed evidence; the paper
> needs symmetric evidence for the champion.

> **TODO(data):** Plateau-threshold sensitivity (see §8 TODO).

> ~~**TODO(data, optional):** E12 grid-bin LNS. If it lands ≤ 1.10 with
> zero overlaps it changes the closing chapter; if not, it lives as
> another falsified-or-marginal datapoint.~~ **DONE 2026-04-28** —
> E12 landed at 1.0990 (zero overlaps); promoted to champion (ADR-007).
> Closing chapter updated in §8 and §9.1.

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

### 10.5 Limitations
- CD plateaus after 10–15 sweeps; ibm17/18 still descending at the cap on
  ablations with tighter thresholds.
- LNS at single-macro / local-window granularity does not escape CD.
  Cluster-level joint reinsertion remains untested.
- The polyhedral decomposition is the right framework for *understanding*
  but did not yield direct algorithmic advantage. Structural insight
  guided diagnosis, not solution.

### 10.6 What this means for the field
Macro placement with composite objectives may be fundamentally resistant
to divide-and-conquer strategies that work for single-objective problems.
The path forward is *better evaluators*, not better decompositions.

> **TODO(prose):** Draft. Tighten to ~1.5 pages.

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
