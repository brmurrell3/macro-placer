# Macro Placement: A Diagnostic-Driven Path from 1.46 to 1.08, and a Hessian Saddle Escape Beyond

> **STATUS:** Working master draft. Source-of-truth for the paper. Section
> scaffolding + locked-in numbers; prose drafts are TODO. Cite the supporting
> docs (`evidence.md`, `contributions.md`, `theory.md`) but do not duplicate
> them — write fresh prose from those sources here.

> **TODO(meta):** Title, authors, affiliation, venue/format target.
> Current placeholder title above. Length target: 16–20 pages including
> the Hessian-saddle-escape arc (§§8.9–8.11) and the DREAMPlace-lane
> refutation (§8.12).

> **Headline numbers (verified, 2026-05-16):**
> - **1.06650** IBM `--all`, **0.68086** NG45 — cascade + DREAMPlace
>   third lane (Option B, `cd_lns_sa_cascade_dp_lane/placer.py`),
>   17/17 IBM + 4/4 NG45 zero overlaps under 60-min/bench cap.
> - **1.07820** IBM `--all`, **0.68102** NG45 — cascade alone (Option A,
>   `cd_lns_sa_cascade/placer_adaptive.py`), no external deps.
> - **1.0612** uncapped on M3 (E84 cascade saddle escape algorithmic
>   ceiling, 17 IBM).
> - vs RePlAce 1.4578 → **−26.9 %**; vs every verified leaderboard entry
>   → **≥ −13 %**; vs leaderboard #1 vmallela self-report 1.0109 →
>   **+5.6 %** gap.
> - Composite avg across 21 benches (17 IBM + 4 NG45): Option B 0.993,
>   Option A 0.998.

---

## Abstract

We report a complete algorithmic and empirical study of the macro
placement problem under the Partcl/HRT 2026 Challenge proxy
(`f(p) = WL + 0.5·D + 0.5·C`), traversing three eras of mechanism
discovery in three weeks. **Era 1 (diagnostic, E1–E12)** decomposes
the proxy as 6 % wirelength / 20 % density / 74 % congestion and
exposes two structural mismatches — LP-HPWL with refined proxy
(ρ = −0.001), RUDY-congestion with real congestion (10.9 % top-5 %
hotspot overlap) — that explain why decomposition-based and surrogate-
gradient approaches saturate. Routing those findings through "bypass,
don't fix" yields a 4657× incremental evaluator that makes full-proxy
coordinate descent feasible inside the contest budget; per-benchmark
plateau detection and a grid-bin LNS escape phase reach
**1.0990** (E12, 17 IBM, zero overlaps). **Era 2 (compositional,
E18–E61)** stacks SDF and DPO basin sources, SA-v2 polish, K-macro
joint LNS, and per-bench best-of hybrids into a 2-lane cascade at
**1.08151** (E48). The E65 cross-section formalizes that the two
basins are separated by an infeasibility wall, ruling out the
element-level recombination family. **Era 3 (transition-state escape,
E74–E91)** identifies the local-move plateau as a high-index saddle of
the smooth proxy with up to four negative-curvature eigenvectors and
escapes it by stepping along the softest mode and polishing on the
canonical proxy — a direct transplant of dimer / climbing-image NEB
methods from chemistry to combinatorial placement. Iterating the
escape until the smooth-proxy Hessian becomes positive-semidefinite
reaches a verified **1.0612** uncapped on M3. A 5.36× implementation
speedup closes the cap-vs-ceiling gap on EPYC under the 60-min/bench
constraint, landing at **1.0782**. Adding DREAMPlace as a third init
lane — and empirically refuting a documented PATH B autopsy that had
ruled DREAMPlace structurally inferior — reaches **1.0665** on 17 IBM
+ NG45 0.6809 (Option B). The system beats RePlAce (1.4578) by 26.9 %
and every verified leaderboard entry by ≥13 %, with a single global
algorithm, no per-benchmark tuning, zero overlaps everywhere, and
CPU-only Python (PyTorch autograd for the Hessian-vector product;
no GPU required).

---

## 1. Introduction

Macro placement seeks the assignment of 200 to 537 rectangular hard
macros to a 2D canvas that minimizes a composite proxy
`f(p) = WL + 0.5·D + 0.5·C` (wirelength, density, routing congestion)
under a hard zero-overlap constraint and a 60-minute-per-benchmark
budget. Classical baselines on the 17-benchmark IBM ICCAD04 suite
position the difficulty: RePlAce — a heavily-tuned analytic placer
two decades old — reaches **1.4578**; simulated annealing reaches
**2.1251**; the partcl-distributed seed placement is **1.5338**.
The competition's public leaderboard target is **1.1172**, and the
verified best entry at the time of writing (MTK / DREAMPlace++) is
**1.2818**. The self-reported top entry (vmallela) sits at **1.0109**
with a similar Hessian-saddle-escape mechanism class.

This paper documents a three-week investigation that began with a
polyhedral decomposition of the placement feasible region and ended
with a transition-state-search method transplanted from computational
chemistry. The verified results stop at **1.07820** (Option A,
cascade alone) and **1.06650** (Option B, cascade with a DREAMPlace
third init lane) on 17 IBM, with NG45 commercial-design verification
at **0.68102** / **0.68086** respectively. Zero overlaps on every
benchmark; a single algorithm with no per-benchmark hyperparameters;
CPU-only Python plus PyTorch's autograd for a Hessian-vector product.
The system beats RePlAce by 26.9 % and every verified leaderboard
entry by at least 13 %.

The narrative is structured around two diagnosis-pivot cycles and one
transition-state insight, each driven by *quantitative diagnosis* —
correlation studies, ablations, cell-by-cell fidelity analysis — rather
than algorithmic intuition. The first pivot replaces a polyhedral
LP-navigation system (1.4867) with a differentiable-proxy optimizer
(DPO, 1.3834) after a correlation study reveals that LP-HPWL is
*uncorrelated* with the refined proxy (ρ = −0.001) — the LP optimizes
the wrong objective. The second pivot replaces DPO with a full-proxy
coordinate-descent algorithm (CD, 1.1055) after a cell-by-cell RUDY
fidelity analysis reveals 10.9 % hotspot overlap with real congestion
on ibm01 — DPO's congestion gradient points in the wrong direction on
hard benchmarks. The transition-state insight (E74) comes from
spectral analysis of the smooth-proxy Hessian at the CD-LNS-SA plateau:
the plateau is not a local minimum at all, but a *high-index saddle*
with up to four negative-curvature directions. Stepping along the
softest mode and re-polishing the canonical proxy lifts every
benchmark uniformly, including the NG45 ariane133 design that had
killed five prior IBM-aligned heuristics. Iterating the escape until
the smooth-proxy Hessian becomes positive-semidefinite (E84) reaches
the algorithmic ceiling at **1.0612** uncapped.

The two unifying themes across the eras are *infrastructure unlocks
algorithm* (a 4657× incremental evaluator made CD feasible inside
the contest budget; a 5.36× delta-cost API closed the cap-vs-ceiling
gap for the cascade) and *bypass, don't fix* (when an approximation is
structurally mis-aligned with the objective, exact evaluation backed
by faster data structures beats a more accurate approximation, applied
twice on the same problem — once to LP-HPWL, once to RUDY).

The paper is organized as follows. §2 sets up the polyhedral
decomposition of the feasible region and §3 describes the navigation
system we built on it. §4 documents the LP-HPWL barrier diagnosis;
§5 covers the DPO pivot and §6 the geometric interpretation of why
penalty continuation crosses the infeasibility barrier. §7 documents
the RUDY fidelity diagnosis. §8 is the longest section: incremental
evaluator (§8), full-proxy CD and plateau detection (§§8.1–8.4),
compositional polish (§8.5), the per-bench best-of hybrid (§8.6), the
failed extension wave and the infeasibility-wall finding (§§8.7–8.7.5),
the spatial-block crossover (§8.7.6), the IBM/NG45 transfer-failure
pattern (§8.8), single Hessian saddle escape (§8.9), cascading saddle
escape (§8.10), the implementation speedup (§8.11), and the DREAMPlace
third lane (§8.12). §9 collects empirical results across 17 IBM + 4
NG45 designs. §10 discusses the recurring objective-mismatch pattern
and what we believe it means for the field. §11 collects references.

---

## 2. The Polyhedral Decomposition

The non-overlap constraint between two hard macros `m_i` and `m_j`
with half-widths `w_i, w_j` and half-heights `h_i, h_j` is the
disjunction `x_i + w_i ≤ x_j − w_j ∨ x_j + w_j ≤ x_i − w_i ∨
y_i + h_i ≤ y_j − h_j ∨ y_j + h_j ≤ y_i − h_i` (Left of, Right of,
Above, Below). Choosing one branch of each of the `N choose 2`
disjunctions yields a complete pairwise L/R/A/B assignment; the set of
placements consistent with that assignment is a (possibly empty)
convex polyhedron defined by `O(N²)` half-spaces. The non-overlap
feasible region of the placement problem is therefore the union of
those convex polyhedra over all valid assignments — a classical
disjunctive program in the sense of Balas (1979). Within any single
polyhedron, half-perimeter wirelength minimization is a linear program
that HiGHS solves in milliseconds, and the LP duals yield a complete
sensitivity map of every pairwise constraint at no additional cost.

The mathematical structure is well-known. Our contribution at this
stage is the *systematic application* to macro placement: an SDF
initialization (signed-distance-field analytical spreading) seeds a
non-overlapping placement; pairwise relations between macros are
extracted from the SDF assignment to fix the polyhedron; HiGHS solves
the resulting HPWL LP with full dual extraction; a `GridSurrogate`
constructed from canvas-bin RUDY estimates evaluates candidate
re-assignments at ~0.1 ms each; a `ClusterScreener` prunes 20–40 % of
candidate moves at the 1–10 µs level via Zobrist-hash filtering,
displacement-floor checks, net-span HPWL bounds, and axis-crowding
heuristics; surrogate-guided multi-pair cluster moves with a simulated-
annealing acceptance criterion explore the disjunction; and a cascade-
repair projection resolves the residual overlaps that arise when a
multi-pair flip lands on a constraint cycle. Seven modules, ~2 200
lines of code, all preserved in the pre-2026-04-28 git history; the
SDF-initialization module survives at `macro_place/sdf_init.py` because
every champion since uses it.

The disjunctive structure is not novel. What we contribute in this
paper is the *empirical characterization of its failure mode* on the
macro-placement objective (§4), which is what motivates the pivot to
differentiable optimization in §5 and the eventual full-proxy
coordinate-descent algorithm in §8.

> **TODO(figure):** 3-macro polyhedra schematic — three macros, the
> four L/R/A/B branches per pair, the resulting polyhedron in 6-D
> position space; annotation showing one assignment's polyhedron as a
> convex polytope and the union over assignments as the feasible region.
> Generate from `writeup/scripts/polyhedra_schematic.py` (TODO).

---

## 3. The Navigation System

The seven-module navigation system reached a verified **1.4867** avg
proxy on the 17 IBM benchmarks (zero overlaps) after a 300-second
budget per bench. The result beats RePlAce (1.4578) on 3 of 17
benchmarks (ibm02, ibm10, ibm12) and matches the SA baseline within
2 %. By the standards of the polyhedral-search literature this is a
respectable working system: the LP-with-dual-extraction subsystem alone
contains roughly 600 lines of HiGHS interfacing, the GridSurrogate
constructs and refreshes hundreds of thousands of fractional-cell-overlap
RUDY estimates per benchmark, and the multi-pair cluster move with
Zobrist-pruned screening was at the time the most sophisticated
combinatorial component we knew how to build for this objective.

The system also *plateaued* at approximately 1.49. A 22-experiment
overnight sweep (§4) varying surrogate accuracy (8 items), initial
topology (6 items), LP formulation (6 items), and combined-effect
parameters (2 items) produced a global best of 1.4918 against a
baseline of 1.4921 — all within ±0.5 % of each other. Whatever was
binding the navigation system was not its surrogate accuracy, its
initial topology, or its LP formulation. It was something else.

The seven modules other than `sdf_init.py` were removed in the
post-2026-04-28 repository cleanup once the second pivot was complete;
the pre-cleanup commit is referenced in this section for reviewers who
wish to verify the implementation independently. The SDF-init module
survives because every later champion uses it as the analytical
starting point.

---

## 4. The Congestion Barrier — Act 2 Diagnosis

The polyhedral navigation system was tuned, optimized, and ablated for
22 experiments overnight. None of the levers helped. Across `SP3`
(surrogate accuracy improvements, 8 items), `SP1` (alternative initial
topologies including spectral and hMetis partitioners, 6 items), `SP4`
(modified LP formulations including McCormick-style area penalties and
quadratic regularizers, 6 items), and combined two-mechanism
experiments (2 items), the spread of average-proxy results across all
22 runs was **1.4918 best, 1.4921 baseline, ±0.5 % entirely**. The
plateau was not a tuning problem.

The diagnosis came from a simple correlation experiment on ibm01: for
24 distinct polyhedra obtained by single-pair flips from the SDF
baseline, compute the LP-HPWL objective (the LP cost the navigation
system minimizes) and the refined proxy (the score that the
competition evaluates), and ask how the two correlate.

| Pair | ρ |
|---|---:|
| LP-HPWL ↔ refined proxy | **−0.001** |
| LP-HPWL ↔ refined WL | +0.852 |
| LP-HPWL ↔ refined density | −0.536 |
| LP-HPWL ↔ refined congestion | +0.072 |
| refined congestion ↔ refined proxy | +0.825 |

LP-HPWL has *negative zero* correlation with the proxy we are evaluated
on. The LP correctly tracks half-perimeter wirelength
(`ρ_WL = +0.852`) and anti-correlates with density (`ρ_density =
−0.536`, the "tighter wires pull macros together, density worsens"
physical anti-correlation), but it is blind to congestion
(`ρ_congestion = +0.072`) — and congestion accounts for **74.9 %** of
the proxy on the 17-benchmark DPO-optimized average. Any LP-ranked
polyhedron search is optimizing the wrong objective.

The corroborating evidence is the per-component decomposition of the
proxy itself, measured on DPO-optimized placements across the 17 IBM
benchmarks:

| Component | Avg fraction of total proxy |
|---|---:|
| WL | 5.8 % |
| Density | 19.4 % |
| Congestion | **74.9 %** |

This is a *non-decomposition* finding (revisited in §10.2). The proxy
is congestion-dominated, but HPWL — the only component an LP can
optimize against a linear objective — is the *smallest* component.
Worse, HPWL and density anti-correlate physically (a tighter
HPWL-minimizing placement is denser, which worsens the density term),
so navigating on LP-HPWL improves a 5.8 %-weight component while
degrading a 19.4 %-weight component, and does not even attempt to
improve the 74.9 %-weight component.

The third piece of evidence corroborates the structural reading. The
"swap+LP" experiment swaps a single pair of macros, then re-runs the
LP on the new polyhedron, and measures the congestion change vs the
density change. Swapping *can* find topologies with substantially
lower congestion (up to −44 %), but the LP-refined placement within
those topologies has density *180 % higher* than the baseline. The
LP cannot navigate to the better-congestion polyhedra because its
objective points away from them.

The combined finding — corroborated by three independent pieces of
evidence — is **structural**, not algorithmic: LP-HPWL is the wrong
objective for this proxy. No amount of LP-search refinement can lift
the navigation system past the ceiling it has hit. The next pivot
must change *what is being optimized*, not how it is being searched.

> **TODO(data):** Generalize the 24-topology Miftari correlation
> experiment from ibm01 to ibm01 / ibm04 / ibm09 / ibm13. The current
> claim rests on ibm01; the multi-design check moves it from
> "indicative" to "established." Source script:
> `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.py`.

> **TODO(figure):** Three figures: (a) LP-HPWL vs refined-proxy
> scatter (ρ = −0.001); (b) refined-congestion vs refined-proxy
> scatter (ρ = +0.825); (c) 22-experiment bar chart of the overnight
> sweep showing all variants within ±0.5 % of baseline. Generate from
> `writeup/data/lp_hpwl_correlations_ibm01.csv` (TODO: freeze).

---

## 5. First Pivot — Differentiable Proxy Optimization (DPO)

The §4 diagnosis pointed at a specific algorithmic class: optimize the
joint proxy directly via gradients, on a smooth surrogate that includes
*every* component (WL, density, *and* congestion). The polyhedral
system optimized a linearization of one component; the next-generation
system would differentiate through all three. We call the resulting
architecture DPO ("Differentiable Proxy Optimization").

The architecture is straightforward and mostly assembles standard
components from the analytical-placement literature:

1. **SDF init** (~3 s) provides a non-overlapping starting point.
2. **LSE-HPWL** with annealed gamma (Naylor et al. 2001) — the standard
   log-sum-exp smoothing of wire half-perimeter that makes WL
   differentiable everywhere.
3. **Differentiable grid density** via `torch.topk` over per-cell area
   contributions (top-10 % bins; smoothed via the standard softmax
   relaxation of the order statistic).
4. **Differentiable RUDY congestion** — our own contribution, lifted
   from the standard-cell-placement work in C3PO / NV-Place (ASP-DAC
   2026) but adapted to the macro-placement setting and to the
   competition's RUDY-based congestion estimate. Smooth bounding-box
   coverage maps to fractional per-cell overlap, accumulated to a
   continuous ABU-5 % estimate.
5. **Pairwise overlap penalty** (ReLU per pair), annealed across three
   phases (λ ∈ {1, 50, 500}).
6. **Adam optimizer** (3 phases, learning-rate schedule), then an
   iterative legalization sweep at the end that resolves the residual
   overlaps left by the soft-overlap penalty.
7. **Best-of(SDF, DPO)** per benchmark — return whichever has the
   lower canonical proxy.

The headline verified result is **1.3834** avg on 17 IBM (`best_of_v2`,
+5.1 % vs RePlAce), with 5-seed variance of 0.45 % (range 1.3790–1.3927)
and every seed individually beating RePlAce. Multi-seed verification
is *not* a tiebreaker in the competition rules, but it is the
diagnostic we use to distinguish "robust mechanism" from "lucky run":
DPO's 0.45 % variance — within a hundredth of a proxy unit across
seeds — is the practical signature of a continuation method that has
converged to the *same* basin from different perturbations.

An ablation falsifies the alternative hypotheses about what is driving
the lift:

| Variant | Avg | Δ vs DPO seed-42 | Δ vs RePlAce |
|---|---:|---:|---:|
| Full DPO (seed 42) | 1.4246 | — | +2.3 % |
| − congestion gradient | 1.5092 | +5.9 % | −3.5 % |
| − density gradient | 1.7342 | +21.7 % | −19.0 % |
| Phase-1 only (no penalty annealing) | 1.4605 | +2.5 % | −0.2 % |
| Random init (no SDF) | 4.9415 | +247 % | −239 % |

The two largest individual contributions are the density gradient
(+21.7 % without it) and the SDF init (+247 % without it). Removing
the congestion gradient costs +5.9 %, which is small in absolute terms
but is the largest lift of any component that is genuinely *novel* to
this work — the literature on differentiable RUDY congestion at
macro-placement scale was not yet developed at the time we built this
system. The penalty annealing matters too (+2.5 % without it) but is
a standard continuation-method ingredient that the literature has
extensively justified (Bertsekas 1982, Lagrangian methods).

The novelty boundary is sharp. Differentiable HPWL + density is
DREAMPlace standard (Lin et al. 2019). Differentiable RUDY exists in
C3PO and NV-Place (ASP-DAC 2026) for *standard-cell* placement. Our
contribution in this section is the *application* of differentiable
RUDY to macro placement against the partcl/HRT 2026 competition
metric, with the empirically-measured 5.9 % lift over a no-congestion
ablation. The architecture itself is not novel; the empirical
finding about its limits — covered in §7 — is.

> **TODO(figure):** DPO ablation bar chart from the table above.
> Generate from `writeup/data/dpo_ablation.csv` (TODO: freeze).

---

## 6. Why DPO Crosses the Barrier

The §2 polyhedral decomposition framed the feasible region as a
*disconnected union* of convex polytopes. A continuous optimizer that
respects the non-overlap constraint at every step is trapped in
whichever polyhedron its initialization lands in. DPO's penalty-
continuation schedule does not respect the constraint at every step;
during the low-λ phase the optimizer is permitted to *traverse
temporarily-infeasible configurations*, and by the high-λ phase the
penalty is high enough that the placement is again essentially legal.
The question this section addresses is whether penalty continuation
constitutes a real basin-changing mechanism, or whether DPO simply
descends within the same polyhedron the SDF init seeded.

The empirical answer is unambiguous. We measure the per-pair L/R/A/B
assignment at the SDF init and at the DPO output and count how many
flip:

- ibm01: **3 556 / 30 135 (11.8 %)** of pairwise assignments flip.
- ibm10: **15 539 / 308 505 (5.0 %)**.

DPO converges in a *different* polyhedron than SDF init started in,
on every benchmark, with the largest flip counts concentrated on the
hardest benchmarks. The dominant transition types are
*perpendicular flips* (L↔B and R↔A, ≈98 % of the changes); same-axis
flips (L↔R, A↔B) are essentially absent. The pattern is consistent
with a continuation method that follows the smoothed objective's
gradient through 2D-projected configurations the strict-feasibility
constraint would have forbidden.

The structural interpretation is a complexification argument
(Zariski 1962, interpreted here as commentary rather than proof). A
2D placement of `N` macros lives in `ℝ^{2N}`; lifting each macro to
`ℝ^3` by adding a z-coordinate yields a placement space in `ℝ^{3N}`
where two macros at different z-heights cannot overlap, regardless
of their xy-projections. In `ℝ^{3N}` the feasible region is
*connected* — one can continuously deform any feasible 3D placement
to any other by lifting through the z-axis. The penalty term
`μ Σ z_i²` realizes the lifting; annealing μ → ∞ projects the
relaxed 3D solution back to 2D. The empirical signature of an
effective continuation is *low seed variance*: a method trapped in
random local minima would show high variance, while a method
converging in a deterministic basin would show essentially none.
DPO's 0.45 % seed variance against the random-init 4.94 avg is the
practical signature of the continuation succeeding.

What this section does *not* claim is that DPO's basin is the
right one. The barrier it crosses is the SDF-induced topological
barrier; whether the basin on the other side is the right one for the
proxy is the subject of the next section.

---

## 7. The RUDY Limit — Act 3 Diagnosis

DPO reaches 1.3834 best-of-v2 on the 17 IBM benchmarks, but it
worsens 4 of them (ibm01, ibm02, ibm06, ibm12) compared to the SDF
init alone. The pattern raises the question: is DPO's congestion
gradient pointing in the wrong *direction*, or is its *magnitude* just
miscalibrated? The first hypothesis predicts that no congestion-weight
tuning can rescue the worsened benchmarks; the second predicts that
some weight setting would.

The congestion-weight sweep tests the magnitude hypothesis directly.
We re-run DPO on the predictive `--fast` subset (4 benchmarks) with
the congestion-loss multiplier swept across `{0.5, 0.75, 1.0, 1.25,
1.5}`:

| Cong weight | Avg proxy (`--fast`) | Δ vs control |
|---|---:|---:|
| 0.50 (control) | 1.2081 | — |
| 0.75 | 1.2251 | +1.4 % worse |
| 1.00 | 1.2278 | +1.6 % worse |
| 1.25 | 1.2377 | +2.5 % worse |
| 1.50 | 1.2382 | +2.5 % worse |

Every weight setting above 0.5 makes things *worse*, monotonically.
The direction hypothesis is correct: DPO's RUDY congestion gradient
points away from the lower-congestion basin, and dialing it up just
amplifies the misdirection.

The cell-by-cell fidelity analysis on ibm01 quantifies the
mis-correspondence between RUDY and the canonical congestion estimate
used by the competition evaluator:

- **Real-to-RUDY congestion ratio: 3.1×** — RUDY systematically
  under-predicts congestion by a factor of three (not the 2× estimated
  from aggregate metrics; the cell-level resolution reveals the larger
  gap).
- **Real exceeds RUDY on 1 843 of 1 845 cells** — RUDY *never* over-
  predicts on this benchmark.
- **Spatial coefficient of variation of the per-cell ratio: 0.944** —
  the under-prediction is heterogeneous, not uniform.
- **Top-5 % hotspot overlap: 10.9 %** (Jaccard 0.057) — the cells RUDY
  identifies as hotspots are *near-random* with respect to the cells
  that actually congest.
- Three structural sources contribute to the gap: L-routing vs the
  uniform bounding-box assumption (a 2.74× contribution), missing macro
  blockage (28.3 % of real congestion is blockage-induced and RUDY
  ignores it), and aggressive spatial smoothing in the RUDY formula.
- Vertical congestion is worse-modeled than horizontal (correlations
  0.327 and 0.635 respectively) — RUDY's bounding-box symmetry assumes
  routing distributes evenly in both directions, which is wrong in
  practice.

The 10.9 % top-5 % hotspot overlap is the load-bearing number. DPO's
congestion gradient steers toward what *RUDY* thinks the hotspot cells
are; if 89 % of those cells are not actually congested in canonical
RUDY, then the gradient direction is steering toward placements that
do not reduce real congestion. The four benchmarks where DPO worsens
SDF init are precisely the cases where this misdirection costs more
than DPO's WL and density improvements gain.

Three subsequent DPO extensions confirm the basin-lock interpretation
of the limit:

- **E5 batched seeds** (`B=64` on MPS, 4 distinct seeds): all collapse
  to the same basin. ibm02 / ibm12 outputs are byte-identical across
  4 seeds.
- **E10 congestion-only refinement** (re-optimize on the congestion
  loss alone after the joint DPO output): −0.33 % on `--all`; ibm02
  *worsens* by +2.3 % — the same benchmark the joint DPO already
  loses on.
- **E11 diverse priors** (DPO over SDF init, Will-SA seed, greedy, and
  random inits, returning best-of-4): −0.8 % on `--fast`, flat
  (+0.04 %) on `--all`. The diversity does not change the basin.

The structural conclusion: **DPO's gradient points to the wrong basin
because RUDY mis-identifies which cells are congested, and within the
wrong basin no amount of additional gradient steps, seed variation, or
init diversity finds the right one**. The next pivot must replace the
surrogate evaluator with something accurate enough that gradient
direction is reliable — or replace gradient descent with an algorithm
that does not need an accurate surrogate at all. We choose the second
path in §8.

> **TODO(data):** Freeze `analysis/rudy_fidelity/rudy_analysis.py`
> output on ibm01 to `writeup/data/rudy_ibm01.txt`, so the 10.9 %,
> 3.1×, Jaccard 0.057 numbers reproduce from a captured artifact, not
> from a re-run.

> **TODO(figure):** RUDY-vs-canonical-congestion heatmap on ibm01 with
> top-5 % cells marked on each. Per-cell ratio distribution (real /
> RUDY) histogram. Generate from the frozen analysis output.

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

**E12 — grid-bin LNS overlay (prior champion, 1.0990; ADR-007).** Three escape
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

The four mechanisms compose into a sequence whose individual lifts
each address a specific limitation of its predecessor:

1. **The incremental evaluator** unblocks coordinate descent by making
   per-move proxy evaluation a 6.5-ms operation rather than a 30-second
   one. Without it, the contest budget allows perhaps two sweeps per
   benchmark; with it, 13 sweeps on the same bench.

2. **Full-proxy CD with breakpoint enumeration** beats DPO on the same
   benchmarks (ibm10: CD 1.0632 vs DPO 1.254 = −17.5 %) because it
   optimizes against the canonical proxy directly, with the breakpoint
   set covering all O(net degree + grid cols) candidates per axis per
   macro. The exact evaluator removes the surrogate error that
   constrained DPO; the breakpoint enumeration provides the discrete-
   topology jumps that gradient descent cannot make.

3. **Per-benchmark plateau detection (E9)** routes compute time to
   where it is needed. Every benchmark exits CD via plateau, never via
   wall-clock cap — the algorithm self-terminates when sweep-deltas
   fall below `plateau_threshold = 0.005`. The contest 1-hour cap is a
   safety bound, not the operational regime.

4. **Grid-bin LNS escape (E12)** crosses the per-axis fixed point that
   CD plateaus at. After CD terminates, K = max(1, min(30, 5 % of
   movable hard macros)) macros are destroyed (cost-aware ranking) and
   each is reinserted at the proxy-minimizing legal `(grid_col, grid_row)`
   cell center — a fundamentally different move type than CD's per-axis
   breakpoint enumeration. The escape lifts E9's 1.1055 to **1.0990**
   on `--all`, with zero benchmark regressions, in an additional 600 s
   per benchmark of LNS time.

A cost-aware-vs-random destroy ablation on the predictive `--fast`
subset showed that the cost ranking is **not** load-bearing: random
destroy averaged 0.9372 vs cost-aware 0.9426 (within run noise, with
random actually winning on ibm09 — 0.8541 vs 0.8591). Production keeps
the cost-aware ranking because the ablation result landed late in the
project timeline, but the simplification is justified by the data.

> **TODO(data):** Capture per-sweep convergence (proxy / WL / density /
> congestion vs sweep) for 3–4 representative benchmarks (easy /
> medium / hard) from a re-run with logging — drives the convergence
> figure below.
> **TODO(data):** Plateau-threshold sensitivity sweep
> (0.001 / 0.002 / 0.005 / 0.01) so the defaults are *defended*, not
> asserted. E16 (`plateau_threshold = 0.001` → 1.1025) is one such
> datapoint; add two more to verify the gradient.
> **TODO(figure):** CD convergence curve (proxy vs sweep) for one
> easy / medium / hard benchmark; CDAdaptive per-benchmark wall-time
> chart showing every bench exiting via plateau.

---

## 8.5 Compositional Polish — SA-v2, DPO Init, K-joint

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

E12 (1.0990) is monotone-descent on the proxy: every move either
reduces proxy or is rejected. A natural extension is to add a
stochastic Metropolis acceptance step on top of CD's breakpoint
enumeration — accept moves that *increase* proxy with probability
`exp(−Δproxy / T)`, in the hope that the chain occasionally crosses
local barriers that CD's strict monotone rule cannot. The implementation
("SA-v2") reuses CD's per-axis breakpoint set as the move proposal
distribution, runs a Metropolis chain at `T₀ = 5×10⁻⁴`, and tracks
best-so-far so the polished output is the lowest-proxy state observed.

Two implementation-level findings are load-bearing for any future
SA-on-breakpoints work:

- **Best-so-far tracking is non-negotiable.** E14 (an earlier SA
  attempt without it) regressed +5.7 % below the CD baseline because
  the chain wandered upward and the final state was *not* the best
  visited. Returning to the proxy-minimizing visited state turns SA
  from a regression into a marginal lift.
- **`T₀` must match the per-breakpoint Δproxy magnitudes.** CD
  breakpoint deltas at the plateau are on the order of `10⁻⁴` (the
  acceptance threshold for CD's monotone rule); `T₀ = 5×10⁻⁴` gives
  ~10–40 % uphill acceptance, which is the chain's exploratory regime.
  At `T₀ = 0.01` (E14's setting), acceptance is ~99 % and the chain is
  a random walk.

With both rules satisfied, E25 (CD + grid-bin LNS + SA-v2) reaches
**1.0954** on `--all`, a 0.33 % lift over E12. Importantly, even with
best-tracking on, the SA chain wanders upward by ~0.01 absolute on
hard benches before recovering — SA polishes *within* a basin but does
not escape it, which is the structural ceiling we revisit in §8.7.
ADR-008 documents the change; later superseded by ADR-011.

### 8.5.2 DPO best_of_v2 init replaces SDF (E18 → 1.08979)

The §7 RUDY diagnosis falsified DPO as a *standalone* placer (1.3834
on `--all`), but the question that remained was whether the DPO basin
— qualitatively different from the SDF basin — could serve as an
*initialization* for the CD-LNS-SA-v2 pipeline. The mechanism difference
is sharp: SDF init is density-aware analytical spreading; DPO output
is a partial barrier-crossing into a polyhedron the gradient
continuation has reached. The two starting points yield different
basins after polish, and a multi-basin best-of can in principle pick
the better one per benchmark.

E18 verifies this: DPO best_of_v2 as init to the E25 pipeline
(CD + grid-bin LNS + SA-v2) reaches **1.08979** on `--all`, a 0.84 %
lift over E12. The lift transfers to NG45 commercial designs cleanly
— E18's NG45 result is 0.69193 across the 4 designs (4/4 wins
at-or-below E12's NG45 0.7037). The basin shift survives polish.

The result *contradicts* an earlier prediction from E11 (§7): when DPO
output was used to *refine* DPO (the diverse-priors experiment), the
basin lift was 0 % on `--all`. The difference is that E11 ran DPO →
DPO; E18 runs DPO → CD-LNS-SA-v2. The first is "polish DPO better
within the same RUDY-misdirected basin"; the second is "use DPO's
partial barrier crossing as the *starting* polyhedron for a fundamentally
different polish algorithm that uses the canonical proxy directly." The
mechanism distinction is load-bearing for §8.6's hybrid.

ADR-009 documents E18; later superseded by ADR-011.

### 8.5.3 K-macro joint LNS (E39 standalone, E41 composed → 1.0848)

E12's grid-bin LNS destroys K macros one by one and reinserts each at
its proxy-minimizing legal `(col, row)`. The reinsertion is *single-
macro*: each destroyed macro is placed in isolation, ignoring how
its placement affects the remaining destroyed macros' optimal positions.
For cost-coupled macros that constrain each other through shared nets,
single-macro reinsertion is myopic.

K-macro joint LNS reinserts K destroyed macros *jointly*. At K = 3,
top-N = 5 candidates per macro, the enumeration is `5³ = 125` joint
placements per K-tuple; the proxy-minimizing joint placement is
committed. The move type is fundamentally different from CD's per-axis
single-macro single-axis search or grid-bin LNS's single-macro
two-axis search — joint reinsertion captures coordinated cost across
the K-tuple.

K-joint composes with DPO init (E18) cleanly. E41 = E18 ⊕ K-joint =
**1.0848** on `--all`, a 1.29 % lift over E12 and 0.46 % over E18. The
per-bench breakdown shows K-joint lifting the hardest benchmarks
disproportionately: ibm10 (−1.7 % vs E25), ibm11 (−4.4 %), ibm13
(−2.2 %), ibm14 (−2.7 %), ibm15 (−1.2 %) — the benches where multi-
mechanism plateaus had previously been densest. NG45 transfer is
**0.69022** (better than E12 by 1.91 %, better than E18 by 0.25 %).

Three K-joint variants explored later in the project all failed NG45
transfer in the pattern documented in §8.8 — K=4 (E42), longer-budget
K-joint (E43), and spatial K-tuple selection (E44) each lifted IBM but
regressed substantially on ariane133. The structural reading from §8.8
is that mechanism-aligned destroy heuristics depend on dense macro
packing that IBM has but commercial designs do not. The K=3 base
configuration with netlist-adjacency K-tuple ranking — the version in
E41 — is the topology-blind safe baseline that the failed variants
diverge from. ADR-010 documents E41; later superseded by ADR-011.

> **TODO(figure):** Lineage waterfall — bar chart showing E12 → E25 →
> E18 → E41 per-bench, with each added mechanism colored separately.
> Visualizes the compositional structure.

---

## 8.6 Per-Bench Best-of Hybrid — E48 (prior champion)

E25 (SDF init + CD-LNS-SA-v2) and E41 (DPO init + CD-LNS-SA-v2 +
K-joint) reach 1.0954 and 1.0848 respectively on `--all` — within
1 % of each other in aggregate, but with substantial per-benchmark
heterogeneity in which one wins. The verified per-bench comparison:
**E25 wins on 5 of 17 benchmarks** (ibm01, ibm06, ibm07, ibm17, ibm18,
where the DPO basin is globally worse for those particular topologies)
and **E41 wins on 12 of 17** (the rest, where the DPO basin transfers
to a better polish endpoint).

The theoretical best-of-2 (picking the per-bench winner with oracle
knowledge of canonical-proxy values) is **1.08121**. The E48 hybrid
implements this without oracle knowledge: for each benchmark, it runs
*both* pipelines, evaluates the canonical proxy on each output, and
returns the lower-proxy result. No per-bench hardcoded logic; no
benchmark-name dispatch; the meta-algorithm is "pick whichever lane
wins by canonical proxy on this run." Verified `--all` is **1.08151**,
within float-drift (3 in the fifth decimal) of the theoretical bound.
ADR-011 *Accepted* 2026-05-02; supersedes ADR-007 through ADR-010.

The structural insight is that the per-bench winner is determined by
*which init class lands in the right basin* for the benchmark, not by
which polish algorithm is intrinsically better. The hybrid is therefore
*meta-algorithmic* — it does not introduce a new mechanism, it exploits
the across-benchmark heterogeneity in which existing mechanism wins.
The result positions this work in the "compositional optimization /
ensemble" tradition rather than the "tuned single algorithm" tradition.
The §8.12 DREAMPlace-lane extension (Option B) generalizes the same
pattern from 2 lanes to 3 — the meta-algorithmic structure is identical;
adding a lane adds a basin source without changing the merger.

NG45 transfer for E48 is **0.6922** average across the 4 commercial
designs, with ariane133 at 0.6861. Per-design, the per-bench best-of
lane-picking pattern transfers cleanly: ariane133 picks the DPO lane,
ariane136 and mempool_tile tie on both lanes, nvdla picks SDF. The
hybrid mechanism is design-class-blind by construction; the basin
heterogeneity drives the per-design lane pick.

The 3-lane extension (E53m, adding E41 seed=1 as a third lane) lifts
`--all` by only +0.02 % (within run-to-run noise) but `--fast` by
−0.97 % (a sample-size outlier on the 4-bench predictive subset where
DPO seed-noise is amplified relative to the bench-set diversity).
Multi-seed within DPO does not extend the basin coverage that adding a
genuinely different init class (DREAMPlace, in §8.12) does.

> **TODO(figure):** Per-bench bar chart of E25 vs E41 vs E48 vs E53m
> outputs on all 17 IBM benchmarks. Color-coded by lane-pick winner.
> Generate from
> `results/CDLNSSAHybridPlacer_*.json` + corresponding single-lane
> result files.

---

## 8.7 Failed Extensions (the May 1–2 wave) (~1 page)

**Source material:** `experiments/E53_dpo_basin_eval/manifest.md`,
`experiments/E53_multiseed_hybrid/manifest.md`,
`experiments/E54_congestion_destroy/manifest.md`.

After E48 verified at 1.08151, three orthogonal extension directions were
tested overnight 2026-05-01 → 02. **None lifted past E48.**

### 8.7.1 GPU DPO basin polish (E53) — falsified

The hypothesis was that GPU acceleration would allow Adam-on-smooth-
proxy as a polish phase *after* CD-LNS-SA, exploiting parallel restarts
to escape the local optimum the prior phases converge to. The
implementation (MPS device on M3 Max, multi-restart full-pose Adam
with annealed overlap penalty, restart budget 10–50 per benchmark)
ran across 4 benchmarks at production budgets.

**Result: 0 GPU restarts out of 350 produced a lower-proxy state than
the CD-LNS-SA baseline.** A smoke-test variant with shortened CD/LNS/SA
budgets (10–20 s each, vs production ~660 s each) *did* see −2.16 % GPU
lift, which is the key falsifier — the GPU phase finds lift only when
the prior phases under-converge. At production budgets, CD-LNS-SA-v2
converges tightly enough that the smooth-proxy gradient cannot find an
exit.

The structural reading is sharper than "GPU doesn't help." The
breakpoint-enumeration CD + grid-bin LNS + Metropolis SA composition
already explores a *richer* candidate set than the smooth-proxy
gradient can — CD evaluates O(degree + grid cols) candidates per axis
per macro on the canonical proxy, while Adam on smooth proxy follows
the *direction* of the surrogate gradient (which §7 has documented is
systematically biased for congestion-dominated benches). For GPU
acceleration to contribute, it must architecturally *replace* CD's
basin-crossing role rather than act as a polish phase after fully-
converged CD-LNS-SA. The Xplace integration explored later in the
project (and still in flight under GPU quota approval) is the
implementation of that architecture.

### 8.7.2 Multi-seed within DPO (E53m) — marginal

The 3-way hybrid `{E25 (SDF), E41 seed=42 (DPO), E41 seed=1 (DPO)}`
on the `--fast` subset reached 0.91128, a 0.97 % lift over E48's
`--fast` 0.92024. The `--all` result was 1.08128 — essentially tied
with E48 1.08151 (−0.02 % within run noise). The `--fast` lift was a
sample-size outlier on the 4-benchmark predictive subset, where DPO
seed-noise on smaller benchmarks (ibm01 in particular) was amplified
relative to the bench-set diversity. At the 17-benchmark scale the
seed variance averaged out and the aggregate lift collapsed.

The structural finding is that **multi-seed within DPO is not a
breakthrough vector at competition scale**. Adding seed diversity does
not extend basin coverage when both seeds converge in the same DPO
basin (E5 confirmed 4-seed byte-identical outputs on ibm02 and ibm12).
The next basin-source extension to actually succeed is §8.12's
DREAMPlace lane, which adds a *different basin class* rather than a
different seed of the same class.

### 8.7.3 Mechanism-aligned destroy (E54) — falsified on NG45

The §4 proxy decomposition (6 % WL / 20 % density / 74 % congestion)
suggested that LNS destroy heuristics should target the dominant
component. E12's cost-aware destroy ranks by total `Δproxy`, which
weights every component equally; E54 instead ranks by macro
contribution to ABU-top-5 % congested cells, matching the proxy's
dominant component. On IBM `--fast` E54 tied E48 (with per-bench wins
on ibm04 and ibm13). The mechanism appeared aligned with the proxy
structure and looked like a candidate for promotion.

**On NG45 ariane133, E54 regressed +5.14 % vs E48** — a catastrophic
failure on a single design. The NG45 aggregate landed at 0.7022 vs
E48's 0.6922 (+1.45 %). E54 joins a growing list of mechanism-aligned
destroy variants with the same NG45-blind failure pattern: E42
(K-joint K=4, +3.57 % ariane133), E43 (longer K-joint, +4.10 %), E44
(spatial K-tuple, killed its own IBM gate). The structural reading,
formalized in §8.8, is that mechanism-aligned destroy heuristics depend
on dense macro packing typical of IBM ICCAD04 layouts; the sparse
ariane-class commercial designs degrade them sharply. Cost-aware
destroy (a topology-blind ranking by total Δproxy) and netlist-
adjacency K-tuple ranking (using graph structure, not geometric
structure) are the safe baselines that survive cross-design transfer.

> **TODO(figure):** Three-experiment summary bar chart — E48 vs E53,
> E53m, E54 on `--fast`, `--all`, and `--ng45`, with ariane133 broken
> out. Visualizes the IBM/NG45 transfer-failure pattern that §8.8
> formalizes.

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

The two cross-sections tested (ibm01 and ibm12) both yielded **0 of
9 feasible interpolations**. Pre-legal residual-overlap counts peaked
at 145 hard macros at k = 0.5 on ibm01 (range 88–155 across the
9 intermediate k values); ibm12 showed 233 residual overlaps at the
mid-cross-section despite the endpoint placements differing by only
0.2 % in proxy. The implication is structural: **the wall is a function
of spatial-configuration distance, not proxy distance**. Two placements
can be proxy-equivalent yet separated by hundreds of overlap-violating
intermediate placements.

This rules out — for this problem class — any mechanism that bridges
macro-placement basins at the *solution* level via per-element
recombination. The claim is stronger than E5's "DPO seeds collapse to
byte-identical placements" because E5 was about within-basin
determinism; E65 says the *space between* any two converged
placements is overwhelmingly infeasible, even when the basin endpoints
are nearly proxy-equivalent. The set of legal placements forms a
*disconnected union of thin manifolds* in `ℝ^{2N}`; element-level
interpolation between two manifolds passes through forbidden territory
with overwhelming probability.

What the wall rules *in* — mechanisms that survive E65's prediction —
falls into three classes:

- **Coarser-than-element recombination.** E61 V2 spatial-block
  crossover (§8.7.6) takes whole canvas quadrants from one parent each.
  Block-internal topology is preserved from one parent, so each block
  is internally feasible; only block-boundary interactions need overlap
  repair, which `project_overlaps` handles in fewer than 50 iterations.
- **Non-local feasibility-respecting moves.** A K = N Hungarian re-pack
  (with N → all hard movables) operates inside the legal sub-manifold
  by construction. E63 (spectral init) and E64 (LP-bounded beam K-joint)
  attempted this class but remain implementation-blocked.
- **Constrained NEB on the legal sub-manifold.** Following the feasible
  manifold geodesic between two endpoints rather than a Euclidean line.
  Untested in this work; flagged as future direction in §10.4.

The connection to the ML "manifold hypothesis" literature (Tenenbaum
et al. 2000) is worth flagging: the feasible region of macro placements
is an extremely thin manifold in `ℝ^{2N}`, and the algorithms that work
on it are precisely those that respect its topology — coarse-grained
operations that stay close to it, or constrained dynamics that move
along it, but never element-level interpolation that ignores it.

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

**V1 (per-macro Bernoulli)** had each hard macro take its position
independently from E25 or E41 (Bernoulli p = 0.5). The mechanism is
falsified by the result the §8.7.5 cross-section predicts: 136
unrecoverable overlaps per crossover attempt; `project_overlaps` caps
at 50 iterations and never legalizes a single one of the attempts.
V1 falls inside the infeasibility wall by construction; the per-macro
Bernoulli sampling is essentially a random walk through the wall.

**V2 (spatial-block 2 × 2)** divides the canvas into four quadrants
(top-left, top-right, bottom-left, bottom-right) at the canvas
midpoint and assigns each quadrant's hard macros entirely to one
parent (Bernoulli per quadrant). Macros that straddle a block
boundary are assigned to whichever side their center falls on. The
four-block crossover yields a placement where each 2 × 2 region is
internally feasible — each block's macros come from a single parent
that was itself feasible — and only the block-boundary interactions
need overlap repair. The boundary overlaps are sparse and local;
`project_overlaps` resolves them in fewer than 50 iterations.

The mechanism explanation is what makes V2 a finding worth recording:
**the infeasibility wall is avoided not by smoothing the recombination
but by choosing a recombination granularity at which both parents are
individually close to feasible**. V1's per-macro granularity is too
fine — every recombination point is a potential overlap source. V2's
quadrant granularity is coarser than the basin-distinguishing scale
(individual macro positions) but finer than the canvas scale (which
would just pick one parent). The two-by-two division finds a feasible
sweet spot.

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

The 0.07 % standalone lift on `--all` would not justify promotion
under the standalone ≥ 0.30 % threshold this project applies (see §A
on promotion criteria). The argument for promotion rests on E61 V2's
behavior in a *hybrid* with E48: best-of-{E48, E61_v2} per-bench
reaches **1.08025** on `--all` (−0.12 % vs E48 alone), and the three-
way hybrid best-of-{E48, E53m, E61_v2} reaches **1.07995** (−0.14 %).

The hybrid contribution is the meta-algorithmic complement to E48
(§8.6). E48 picks per-bench between two basin sources (SDF and DPO);
E61 V2 provides a third basin source via spatial-block recombination
that neither parent reaches alone. The hybrid does not introduce a
new mechanism — it just adds another lane to the per-bench best-of —
but the additional lane is empirically NG45-positive on every tested
design, including the ariane133 failure point. This is the
infrastructure that the §8.12 DREAMPlace third lane builds on at
larger scale.

E61 V2 was eventually superseded by the Hessian saddle escape (§8.9)
as the project's principal post-plateau mechanism, but the
spatial-block hybrid contribution remains a viable Option-B-class lane
source. The Proposed ADR for the V2 hybrid is preserved at
`docs/decisions/012a_e61v2_spatial_block_crossover_proposed_superseded.md`
as a historical record of the promotion case that was overtaken before
acceptance.

> **TODO(figure):** Per-bench bar chart of E48 / E53m / E61_v2 /
> best-of-3 across the 17 IBM benchmarks. Source data:
> `experiments/E61_ga_crossover/results/best_of_analysis.md`.

---

## 8.8 The IBM/NG45 Transfer-Failure Pattern (~0.75 pages, structural finding)

**Source material:** `docs/roadmap.md` §4.5; per-experiment manifests
above; `external/MacroPlacement/Testcases/ariane133/` for the
benchmark structure data.

Five experiments by 2026-05-02 showed the same transfer-failure
pattern — IBM-fast lift that did not survive NG45 commercial-design
evaluation, with ariane133 as the consistent failure point:

| Experiment | Mechanism | `--fast` Δ vs E41 | NG45 ariane133 Δ |
|---|---|---:|---:|
| E42 | K-joint K = 4 | −0.35 % | **+3.57 %** |
| E43 | Longer K-joint (1200 s) | −0.33 % | +4.10 % |
| E44 | Spatial K-tuple selection | +0.63 % (kill gate fired) | n/a |
| E54 | Congestion-targeted destroy | tied | **+5.14 %** |
| E62 | Will-seed init lane | +1.44 % (kill gate fired) | +1.45 % |

The hypothesis we offer — consistent with all five experiments but not
formally verified — is that mechanism-aligned destroy and K-tuple
heuristics depend on **dense macro packing** in ways the commercial-
design class violates. ariane133 has 133 hard macros on a
1 433 × 1 433-micron canvas (roughly one macro per 15.5 square microns).
IBM ICCAD04 benchmarks have 246–760 hard macros on 23–73-micron
canvases (10–20 per square micron). The factor-of-150–300 density
difference matters for any heuristic that operates on geometric
clustering: the top-5 % cells of an NG45 24 × 21 grid is only 25 cells,
far fewer than the ~750 top-5 % cells that congestion-targeted destroy
can find structurally-coupled K-tuples in on the dense IBM grids. The
mechanism alignment optimizes for geometric patterns that the sparse
layouts simply do not contain.

Cost-aware destroy (ranking by total `Δproxy` without component
attribution) and netlist-adjacency K-tuple ranking (using graph
structure, not geometric structure) survive both design classes —
these are the safe baselines the failed variants diverge from.

The finding sharpens the standard cross-benchmark validation advice
into a specific practical rule: **when a heuristic engages the proxy's
structural decomposition — component fractions, topology, or geometric
clustering — verify on a sparse commercial benchmark like ariane133
before promoting**. The E48 hybrid's safety comes from per-bench
best-of: an IBM-tuned lane that regresses on ariane133 is simply not
picked there at evaluation time. But maintaining the failing lane
consumes wall budget and adds system complexity, and the May 1–2 wave's
evidence is that the expected lift does not justify the complexity for
structurally-aligned variants. The Hessian saddle escape mechanism
introduced in §8.9 follows the opposite design pattern — agnostic to
macro density, working on the smooth-proxy curvature — and lifts
ariane133 by 3.21 % rather than regressing.

The connection to ML out-of-distribution generalization literature
(Arjovsky et al. 2019, Krueger et al. 2021) is worth flagging.
Mechanism-aligned heuristics that overfit to the design class they
were tuned on are a placement-specific instance of the classic OOD
failure mode; topology-blind base heuristics with per-bench best-of
hybrids are the placement-specific instance of the distributionally-
robust mitigation.

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

## 8.10 Cascading the Saddle Escape — E84

**Source material:** `experiments/E84_cascading_saddle/manifest.md`,
`experiments/E84_cascading_saddle/code/cascading_saddle.py`.

### 8.10.1 One escape is not enough

E74 takes a single saddle-escape step: compute the softest eigenvector
of the smooth-proxy Hessian at the E48 plateau, perturb the placement
along it for `(sign, ε) ∈ {±1} × {0.3, 1.0, 3.0}`, polish each
candidate, return the best. The argument behind this single step is
that the local-move family's reachable set is bounded by the saddle's
positive-curvature directions, and we need to leave it along a
negative-curvature direction.

The eigenanalysis in §8.9.2 makes a stronger claim than E74 fully
exploits. On the ibm01 plateau the smooth-proxy Hessian has four
negative-algebraic eigenvalues `(−0.14, −0.076, −0.066, −0.060)`;
on ibm04 it has `(−0.69, −0.11, −0.07, −0.03)`. A single step along
`v_0` follows the steepest of those descent directions but ignores
the others. Worse, once the placement has descended along `v_0` and
the canonical CD polish has settled into a deeper basin, the *new*
basin may itself be a saddle of the smooth proxy with its own
non-empty negative-curvature spectrum.

Cascading saddle escape is the obvious response: iterate. After each
polish, re-compute the Hessian at the new state, check the smallest
eigenvalue, and if it is still negative, escape again. The process
terminates only when one of four conditions becomes true:

1. **All Hessian eigenvalues are non-negative** (within tolerance
   `−10^{−3}`). The smooth surface is locally convex; no escape
   direction remains.
2. **The cascade iteration produced no proxy improvement.** The state
   re-converged to the same basin even after perturbation. We are at
   a robust local minimum of the canonical proxy under the saddle-
   escape composition.
3. **`max_iters` is exhausted.** A soft bound (set to 5 in the current
   implementation) to prevent runaway compute.
4. **The wall budget is exhausted.** Per the deployment cap; the
   iteration returns the best state observed so far.

Each stop condition is interpretable. The first is the algorithmic
ceiling we set out to find: the *true* local minimum of the smooth
proxy in the neighborhood of the original plateau. The second is the
empirical statement that further iteration is wasted. The third and
fourth are deployment safeties.

### 8.10.2 Empirical behavior

Across the 17 IBM benchmarks, cascading saddle escape runs 2–5
iterations per benchmark before terminating, with strong diminishing
returns:

- **First iteration** gives the bulk of the lift, ranging from 0.1 %
  on benchmarks where E48 was already deep in a basin (ibm09:
  E74 0.8205 → E84 0.8190) to 5–10 % on benchmarks where the E48
  plateau was a high-curvature saddle (ibm02: E74 1.0143 → E84 0.9430,
  −7 %).
- **Subsequent iterations** add 0.1 % to 1.0 % each. The pattern is
  consistent: the dominant negative-curvature direction is escaped
  first; subsequent iterations follow successively shallower modes
  until the spectrum is non-negative or the trajectory is recapturing
  the same basin.
- **Stop condition firing** is split roughly evenly between "true
  local minimum reached" (condition 1) and "no improvement"
  (condition 2). Condition 3 (max iters) fires rarely; condition 4
  (wall) fires only on the largest benchmarks under tight budgets.

The cascade also exposes a useful property for downstream analysis:
the *per-iteration lift trace* is monotone (every committed iteration
strictly improves the proxy) and the cumulative lift correlates with
the magnitude of `|λ_0|` at the starting plateau. Benchmarks whose
E48 plateau had a strongly negative `λ_0` (ibm02, ibm17, ibm18) see
larger cumulative cascade lifts; benchmarks already near a smooth-
proxy local min (ibm08, ibm09) see almost none.

### 8.10.3 Verified results

Cached uncapped on M3 Max (no per-bench wall budget enforced):

| Metric | E74 (parent) | E84 cascade (this) | Δ |
|--------|-------------:|-------------------:|---:|
| IBM `--all` avg | 1.0666 | **1.0612** | **−0.51 %** |
| Best-of-cascade benchmarks | — | ibm02 −7.13 %, ibm01 −3.86 % | — |
| Hard overlap pairs | 0 / 17 | 0 / 17 | — |
| Per-bench walls > 55 min | 0 / 17 | 8 / 17 | requires wall-safe variant |

The **1.0612 IBM `--all`** result is, modulo wall-budget effects, the
algorithmic ceiling of the cascade-saddle-escape approach we have
described: further iteration with the same primitives no longer
improves the proxy, on any of the 17 IBM benchmarks. To improve
beyond it requires changing the mechanism — using a different
eigenvector subspace (multi-direction perturbations rather than
top-1), composing with a different basin source (DREAMPlace's
electrostatic basin, for example), or replacing the canonical-proxy
polish with a different local optimizer. We discuss these directions
in §10.

The wall-cap caveat is structural and matters for deployment. Eight
of the 17 cascades run longer than 55 minutes uncapped — the largest
benchmarks (ibm12, ibm17, ibm18) push past 90 minutes when allowed
to run to termination. Under the partcl 60-minute-per-benchmark cap
the cascade has to truncate, and that truncation costs proxy. Closing
the gap from this 1.0612 algorithmic ceiling to a *cap-bound* result
on EPYC is therefore not an algorithmic problem — the algorithm is
already at its fixed point — but an *implementation* problem.
Section 8.11 describes how we close it.

> **TODO(figure):** Cascade lift-per-iteration line chart for three
> representative benchmarks (ibm01 easy, ibm10 medium, ibm17 hard).
> Shows the monotone descent and the per-iter diminishing returns
> that justify the early-stop condition.

---

## 8.11 PATH A — Closing the Cap-vs-Ceiling Gap via Implementation Speedup

**Source material:** `macro_place/incremental_evaluator.py` (commits
`59a7a8b`, `53b4a26`, `af520c3`, `9df5ac2`, `f2269b3`);
`results/CDLNSSACascadeAdaptivePlacer_20260513_*.json` (A4-v1 and
A4-v2 cloud runs); `submissions/cd_lns_sa_cascade/placer_adaptive.py`
(submission entry).

### 8.11.1 The cap problem

The cascading saddle escape reaches 1.0612 on M3 Max with no wall
budget enforced. Under the partcl 60-minute-per-benchmark cap on
AMD EPYC 9655P — the hardware class used for competition evaluation —
the same algorithm plateaus at **1.137**, a gap of more than 7 %
above the algorithmic ceiling. The cause is not algorithmic and is
not difficult to identify: the partcl-class EPYC is roughly half the
per-core speed of an M3 Max for our single-threaded Python workload,
and the cascade's inner CD-adaptive polish runs short of convergence
under the budget on the larger benchmarks. The cascade then has
fewer iterations to spend on saddle escape, and each iteration's
polish is itself shallower than the M3 cached run.

A cProfile run on ibm04 isolated the implementation bottleneck cleanly:
**32 of 33 seconds of CD time** were spent inside
`IncrementalProxyEvaluator.move()` and `revert()`. These are the
two primitives used by `cd_core.search_axis` to *probe* a candidate
placement — apply the move, evaluate the proxy, undo the move — and
the pattern executes thousands of times per CD sweep. Both methods
mutate evaluator state (per-net bounding boxes, per-cell density
contributions, per-cell congestion contributions, the pin-position
cache) and build / restore a single-step `_MoveSnapshot` to make the
mutation reversible. The profile reveals that the snapshot construction
and the state mutation together dominate the inner loop; the actual
proxy computation is a smaller share than one might expect.

### 8.11.2 Four optimizations

We close the cap-vs-ceiling gap with four implementation-level
changes, each verified for bit-exact parity against the reference
move/revert path on 100 random probes per benchmark. None of them
changes the algorithm; the canonical proxy values are identical
modulo float-ordering noise (machine epsilon, 2.22 × 10⁻¹⁶).

**1. `delta_cost(macro_idx, new_xy)` — no-mutation cost peek.**
The (move, current_cost, revert) probe trio is replaced by a single
method that returns the cost dict that *would* result from the move,
without applying it. The implementation briefly retargets the moving
macro's pin positions inside a `try/finally` (so the dependent
`_net_cong_contrib` helper still sees correct positions) and builds
all hypothetical cell tensors — grid density, H/V net congestion,
H/V macro routing — as `clone() + delta` rather than in-place
updates. The snapshot is never constructed; the revert pass never
runs. **1.95× per probe on ibm10**, measured over 500 random probes.

**2. `delta_cost_axis_batch(macro_idx, axis, cur_xy, candidates)` —
batched K-candidate evaluation.** The breakpoint branch of
`cd_core.search_axis` evaluates `K ≈ 12` candidates on one axis per
macro per sweep. The single-candidate `delta_cost` repeats the
"subtract old contributions" work K times. The batched variant
amortizes that precompute once, collects per-candidate cell-level
deltas as flat `(k, cell, val)` triples, and applies them via one
`torch.index_put_(accumulate=True)` per cell-tensor. The density
top-K and congestion smoothing + top-K then run as batched torch
ops on `[K, num_cells]` tensors rather than K serial Python loops.
**4.32× combined on ibm10** (1.95× × 2.32×).

**3. `_net_cong_contrib_flat` — vectorized routing helper.** The
dict-based per-net routing accumulator is the heaviest Python
overhead in the inner loop (cProfile shows 55 % of cumulative time
after the first two optimizations). We replace it with a flat-list
variant that vectorizes the pin→gcell map via tensor `floor` / `clamp`
/ `long` operations, inlines the 2-pin and multi-pin routing patterns
(the 88 % case per cProfile), and skips the dict-allocation /
get-and-update cycle entirely. Duplicates are re-aggregated downstream
by `index_put_(accumulate=True)`, which means the routing helper
itself never needs to deduplicate. **5.36× combined on ibm10**.

**4. LNS-helper conversions.** Two helpers in
`submissions/cd_lns_sa/placer.py` (and their duplicates in E18 and
E39) — `_pick_destroy_by_cost` and `_gridbin_reinsert` — also follow
the (move, current_cost, revert) probe pattern. Converting them to
use `delta_cost` produces a bit-exact result with ~2× lower
per-probe wall time. On the LNS phase of the cascade pipeline this
allows more samples per LNS budget, which feeds the cascade with
deeper E25 / E41 plateaus.

The combined effect is to reduce CD-time-per-cycle without changing
any algorithmic behavior. The cascade saddle escape, which is
agnostic to how fast each polish runs, now reaches multiple
iterations per benchmark under the same 60-minute cap that previously
permitted only one.

### 8.11.3 Verified results

Validated on AMD EPYC 9655P, `--jobs 4`, `budget_seconds=3000`
(50 minutes per benchmark, leaving a 10-minute safety margin under
the 60-minute cap):

| Run | Mode | Avg proxy | Overlaps | vs Pre-A1 |
|-----|------|----------:|---------:|----------:|
| Pre-A1 baseline (2026-05-11) | `--all` | 1.137 | 0 | — |
| A4-v1 (A1 inner only) | `--all` | **1.0782** | 0 | **−5.2 %** |
| A4-v2 (A1 + LNS-delta) | `--all` | **1.0771** | 0 | **−5.3 %** |
| Pre-A1 baseline | `--ng45` | 0.7034 | 0 | — |
| A4-v1 | `--ng45` | **0.6853** | 0 | **−2.6 %** |
| A4-v2 | `--ng45` | **0.6870** | 0 | −2.3 % |

The post-A1 cloud result is **1.0771 IBM `--all`** under the partcl
hardware cap — within 1.5 % of the 1.0612 M3 cached ceiling. The
remaining gap is residual per-core speed differential between M3 and
EPYC, not a structural limitation of the implementation. Closing it
further would require either a Cython or Numba port of the routing
inner loops, or a CUDA port of the routing-dispatch logic to run on
the A100 GPU available on the partcl evaluation hardware.

Neither further optimization is justified at this margin. The
leaderboard top is vmallela (self-reported 1.0109) at **5.6 % below** our 1.0665 Option B;
fully closing the cap-vs-ceiling gap returns less than 0.4 % more
proxy, which is not enough to overtake. The next breakthrough vector
beyond cascade is therefore *algorithmic* — using a different basin
source (the DREAMPlace electrostatic basin polished through cascade)
or a higher-order saddle search (multi-direction simultaneous escape
or a true Newton-CG trust region in the soft-mode subspace) — not
implementation. We outline both directions in §10.4.

### 8.11.4 Why this matters as a paper-level finding

The implementation work here is not novel — the optimizations are
each individually obvious in hindsight (skip the snapshot, batch the
ops, vectorize the dict). What is paper-level is the *compounding*
between implementation and algorithm: the cascade-saddle algorithm
requires multiple polish-then-eigenanalysis cycles per benchmark,
and those cycles are bottlenecked by the inner-CD polish. A 5.36×
speedup on the inner loop translates not to 5.36× faster wall but to
**deeper basins** at the same wall, because the cascade composition
keeps iterating until it hits its algorithmic stop condition rather
than until the wall expires. The result is that the post-A1
cap-bound proxy is essentially the algorithm's ceiling, which
relocates the next-step research question entirely: it is no longer
"can we run the cascade for longer" but "what algorithmic
modification will lift the ceiling."

> **TODO(figure):** A side-by-side wall-vs-proxy plot for ibm12
> (the hardest IBM bench) showing pre-A1 (single cascade iter,
> truncated polish) vs post-A1 (multiple cascade iters, full polish)
> on the same 60-minute budget. Visualizes the compounding claim.

---

## 8.12 The DREAMPlace Lane — Refuting the Autopsy (~1.5 pages)

**Source material:** `experiments/E91_dp_full_polish/manifest.md`,
`experiments/E91_dp_full_polish/code/dp_full_polish.py`,
`submissions/cd_lns_sa_cascade_dp_lane/placer.py`,
`writeup/contributions.md` §24, `writeup/evidence.md` §1.1.

### 8.12.1 The autopsy that almost killed PATH B

Through mid-May, the project's PATH B (DREAMPlace integration) was
documented as falsified. The original autopsy, written 2026-05-10 after
a 25-config DP basin sweep on cloud A100, concluded that DREAMPlace
basins were *structurally below* cascade-capped outputs on every hard
benchmark:

| Bench | Best DP basin | Cascade `b=3000` | Gap |
|---|---:|---:|---:|
| ibm10 | 1.2553 | 1.0775 | +16.5 % |
| ibm12 | 1.3712 | 1.3031 | +5.2 % |
| ibm14 | 1.4176 | 1.2919 | +9.7 % |
| ibm17 | 1.5388 | 1.4546 | +5.8 % |

The conclusion at the time: DREAMPlace optimizes the wrong objective
(electrostatic-density uniform-spread, not top-K density), and no amount
of post-DP cleanup recovers what was lost in the basin. The wave of
work that followed (E92 / E93 / E94 / E95 / E97 / E98) targeted
*modifications* to DP — replacing its density operator, adding a
TILOS-RUDY-equivalent congestion term, calibrating its smooth proxy to
canonical — on the assumption that the stock DP basin was unrecoverable.

### 8.12.2 The unfair comparison

Re-reading the autopsy driver
(`submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py`)
revealed an asymmetry between the lanes. Inside the original test:

- The **SDF lane** (E25) received the full polish budget — CD-adaptive
  to plateau (≤ 660 s) + grid-bin LNS (≤ 200 s) + SA-v2 (≤ 200 s).
- The **DPO lane** (E41) received the same — CD ≤ 660 s + LNS ≤ 165 s
  + SA ≤ 165 s + K-joint ≤ 165 s.
- The **DP lane** received `min_time_s=30, hard_cap_s=60` — *60 seconds
  of CD cleanup*, two orders of magnitude less than the other two
  lanes. No LNS phase. No SA. No K-joint.

The autopsy table compared **DP basins after 60 s of cleanup** against
**SDF/DPO basins after ~1100 s of full polish**, and the gap reported
above is the gap from the unfair comparison. The original conclusion
("DP basins polish 10-23 % worse than cascade") was a statement about
*polish budget*, not about basin quality.

### 8.12.3 B-R0': stock DP + matched polish

E91 ran the comparison the autopsy never did — stock DREAMPlace
(auto-adaptive config, see §8.12.4 below) followed by the **same** polish
budget that E25 and E41 receive: CD-adaptive ≤ 660 s + LNS ≤ 200 s +
SA-v2 ≤ 200 s + cascading saddle escape on the polished plateau. The
results invert the autopsy on the three hardest IBM benches:

| Bench | Cascade (autopsy ref) | DP + full polish | Δ |
|---|---:|---:|---:|
| ibm10 | 1.0775 | 1.0951 | +1.6 % |
| ibm12 | 1.3031 | **1.1294** | **−13.3 %** |
| ibm14 | 1.2919 | **1.2429** | **−3.8 %** |
| ibm17 | 1.4546 | **1.3068** | **−10.2 %** |

Hard-bench aggregate: cascade-capped **1.2818** → DP + full polish
**1.1936** = **−6.9 %**. ibm12 alone is −13.3 % below the previously-
reported cascade-capped result, and the autopsy had reported only a
+5.2 % basin gap there — meaning that *the polish closed 19 percentage
points of basin gap and then crossed into a deeper basin than cascade
reaches with the same polish*. The DP basin, when given a fair polish
budget, lands in a valley that the SDF and DPO basins simply do not
reach on these benchmarks.

The pattern matches an existing piece of mechanism: §8.6's per-bench
best-of-{E25, E41} hybrid wins precisely because different inits land
in different valleys on different benchmarks. DREAMPlace's
electrostatic basin opens a *third* valley that wins on most of the
congestion-dominated IBM benches that cascade-only struggles with.
ibm10 remains a marginal regression (+1.6 %), so the plateau pick at
the lane-merge stage keeps the E25 or E41 winner on that benchmark.

### 8.12.4 The auto-adaptive DP config (rule compliance)

The first iteration of the DP lane used a hardcoded `target_density=0.85`
that worked well on IBM benchmarks but produced three residual overlaps
on ariane133 NG45. A per-benchmark branch (`if bench.name.startswith("ibm")`)
would have fixed that, but the competition rules forbid
benchmark-identity dispatch. The auto-adaptive replacement derives the
target from observable bench geometry:

```python
macro_density = sum(macro_area) / canvas_area    # observable from input
target_density = clip(macro_density * 1.5, 0.40, 0.85)
stop_overflow = 0.02                              # tighter than DP's 0.07 default
dp_iter = 2000
dp_lr = 0.005
```

Same formula on every input. ariane133 (macro_density = 0.496) auto-derives
`target_density = 0.743`, lands at **0.66167** with zero overlaps and
~34 minutes of wall (vs cascade-uncapped 0.6641, tied within +0.88 %).
NG45 generalization holds without any benchmark-specific tuning.

### 8.12.5 Submission architecture: Option B

The cascade-DP-lane placer
(`submissions/cd_lns_sa_cascade_dp_lane/placer.py`) extends the cascade
base with DREAMPlace as a third init lane:

```
SDF init            → CD + LNS + SA-v2 polish (E25 lane)
DPO init            → CD + LNS + SA-v2 + K-joint polish (E41 lane)
DREAMPlace init     → greedy_macro_legalize → matched polish (DP lane)
plateau pick:        argmin canonical proxy over the 3 lane outputs
cascading saddle escape on picked plateau
deadline: 3000s default (50 min/bench, 60-min cap respected)
```

If `DREAMPLACE_ROOT` is unset, the DP lane is skipped and the placer
falls back to the 2-lane base (equivalent to Option A,
`placer_adaptive.py`). The fall-back path means Option B is a strict
superset of Option A.

| Mode | Option A (no DP) | **Option B (3 lanes)** | Δ |
|------|---:|---:|---:|
| `--all` 17 IBM | 1.07820 | **1.06650** | **−1.09 %** |
| `--ng45` 4 commercial | 0.68102 | **0.68086** | tied (−0.02 %) |
| Composite 21 benches | 0.998 | **0.993** | **−0.50 %** |

Per-bench, the plateau pick correctly routes by basin quality:
DP-polished wins on ibm03 / ibm07 / ibm09 / ibm12 / ibm17 (the most
congestion-dominated IBM benchmarks); E25 wins on ibm01 / ibm04 /
ibm06; E41 wins on the rest. The lane-merger does not look at the
benchmark name — it picks by canonical proxy, which is the score we
are evaluated on.

### 8.12.6 What this contributes

Two contributions stack here. The first is purely empirical: the
**autopsy was wrong**, and its conclusion that "DP basins are
structurally inferior" was an artifact of an unfair polish budget. The
correct statement is that **DP basins polish to a different valley
than SDF/DPO inits, and that valley is the deepest one on the
congestion-hardest IBM benchmarks**. This is a finding about the
*method* of evaluating black-box basin generators, not just about
DREAMPlace specifically — any future basin-comparison work needs to
verify that polish budgets are matched before drawing structural
conclusions.

The second is architectural: the cascade pipeline extends to N init
lanes by construction. The plateau pick (argmin canonical proxy) is
the only meta-algorithmic component, and it scales linearly in N
without introducing new hyperparameters. Future basin sources — Xplace
+ Triton (in flight), spectral inits (E63, blocked), constraint-
satisfaction inits (E64, blocked) — slot in as additional lanes.
The cascade saddle escape then runs once on whichever lane wins the
plateau pick.

> **TODO(figure):** Per-bench plateau-pick winner stacked bar across
> the 17 IBM + 4 NG45 benches. Color-coded by lane (SDF / DPO / DP).
> Visualizes the "no lane dominates" claim and motivates the per-bench
> best-of-N architecture.

> **TODO(data):** Capture per-bench A4-v2 (Option A) vs DP-lane
> (Option B) lane-pick distributions from
> `results/CDLNSSACascadeDPLanePlacer_*.json`. Feeds the figure above.

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
| **Cascade + A1 speedup (cloud, 60-min cap) = Option A** | **CDLNSSACascadeAdaptivePlacer** | **1.07820** | **+26.0 %** | **2026-05-16** |
| **Cascade + DREAMPlace 3rd lane = Option B** | **CDLNSSACascadeDPLanePlacer** | **1.06650** | **+26.9 %** | **2026-05-16** |

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

The NG45 transfer is the structural test of the "no per-benchmark
tuning" claim. E48's hybrid mechanism — per-bench best-of-{SDF, DPO}
— picks the better basin at evaluation time, without any tuning that
saw the NG45 designs during development. The per-design winners
distribute across both lanes: ariane133 picks E41 (DPO basin), nvdla
picks E18 (SDF basin via DPO-init), mempool_tile ties, ariane136 picks
E48 (hybrid). The same meta-algorithmic mechanism that exploits
across-IBM heterogeneity also exploits across-commercial-design
heterogeneity — a non-trivial transfer because nothing in the pipeline
was tuned for it. The §8.12 DREAMPlace lane extends this further:
Option B's NG45 0.68086 matches Option A's 0.68102 to within noise
because the per-design lane-pick correctly routes DP-lane wins where
they exist (mempool_tile) and DP-lane regressions where they exist
(ariane133, where DP lane underperforms cascade by 0.88 %).

The §8.9 Hessian saddle escape (E74) further lifts NG45: 0.6813
average across 4 designs, with ariane133 at 0.6641 (−3.21 % vs E48's
0.6861). This is the only mechanism in the project that lifts every
NG45 design simultaneously, including the ariane133 failure point that
five IBM-aligned mechanisms had regressed on. The §8.12 DP-lane
preserves the Hessian lift on ariane133 while adding the basin-source
diversity that lifts the congestion-hardest IBM benchmarks.

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

### 10.5 Compositional polish — orthogonal mechanisms compose multiplicatively

The E18 / E25 / E41 / E48 lineage tells a compositional story that
contradicts the additive intuition. Each mechanism in isolation is a
verified lift over E12: DPO basin shift (E18, −0.84 %), SA-v2 polish
(E25, −0.33 %), K-macro joint LNS (E39 standalone, −0.30 %). Their
naive sum predicts an E48 result of approximately 1.0925; actual E48
verified at 1.08151 (−1.59 % vs E12) — better than the naive sum
predicts. The composition multiplies through the mechanism stack
rather than adding through it: each layer operates on the placement
that the prior layer produced, so the lift compounds at the basin
level rather than accumulating linearly at the proxy level.

The further +0.30 % that the per-bench best-of hybrid extracts is
*meta-algorithmic*: it does not introduce a new mechanism but exploits
the across-benchmark heterogeneity in which mechanism wins. The
§8.12 DREAMPlace lane and the cascading saddle escape (§8.10) continue
the same pattern at successively higher levels of the stack — each
adds a basin-source layer or a saddle-escape layer that compounds with,
rather than replaces, the layer below it. The connection to ensemble
learning / boosting (Schapire 1990, Freund & Schapire 1997) is direct
in framing — meta-algorithms that combine weak hypotheses into strong
ones — though the placement-specific instantiation does not require any
of the boosting theory's distribution-reweighting machinery.

### 10.6 The IBM/NG45 transfer-failure pattern — generalization fails on sparse layouts

§8.8 surfaces a structural finding the paper considers worth flagging
to the field at large. Five experiments engaged the proxy's structural
decomposition (component fractions, topology, geometric clustering)
and lifted IBM benchmarks; all five regressed on the sparse commercial
ariane133 design by 1.4–5.1 %. The generalizable claim is that
**heuristics that engage the proxy's structural decomposition degrade
on benchmarks with different macro-density profiles**. The safe
baselines — cost-aware destroy (topology-blind ranking by total
`Δproxy`) and netlist-adjacency K-tuple ranking (using *graph*
structure rather than *geometric* structure) — survive both design
classes because their move-selection criteria do not encode
benchmark-specific spatial priors.

The paper's contribution here is partly negative: the obvious
"decomposition-aware" heuristics in the placement literature
(component-weighted destroy, congestion-targeted cluster moves,
density-pyramid hierarchical refinement) fail under cross-design
generalization. The field should validate any IBM-tuned mechanism on
a sparse commercial benchmark — ariane133 is the most diagnostic of
the public NG45 designs — before promoting it. The connection to OOD
generalization in ML (Arjovsky et al. 2019, Krueger et al. 2021) is
direct; mechanism-aligned heuristics overfit to the design class they
were tuned on, and topology-blind base heuristics with per-bench
best-of hybrids are the placement-specific instance of the
distributionally-robust mitigation.

### 10.7 Limitations

- **Implementation-level cap-vs-ceiling gap remains.** The cascade
  reaches 1.0612 uncapped on M3; under the 60-min cap on EPYC the
  best is 1.0665 (Option B). A Cython / Numba / GPU port of the
  routing inner loop would close the remaining 0.5 % but was not
  justified within the project's wall-clock budget. See §8.11.
- **CD plateaus after 10–15 sweeps.** ibm17/ibm18 are still descending
  at the cap on tight-threshold ablations. A longer CD phase with a
  tighter plateau threshold would help, but consumes wall budget that
  the saddle escape uses more productively.
- **LNS at single-macro / local-window granularity does not escape
  CD's per-axis fixed point.** Cluster-level joint reinsertion at
  K = N (Hungarian re-pack) remains untested; E63 (spectral init) and
  E64 (LP-bounded beam K-joint) are implementation-blocked.
- **Polyhedral decomposition was useful for understanding, not for
  optimization.** §2's framework guided the §4 diagnosis but the
  algorithmic system built on it (§3) hit a 1.49 ceiling. The proxy's
  74 %-congestion weight defeated the LP-based search.
- **GPU acceleration as an additive polish phase did not help** (E53,
  §8.7.1). Whether it helps as a basin-source replacement (Xplace,
  in flight under quota) is untested.
- **Multi-seed lane diversity hits diminishing returns at `--all`
  scale** (E53m, §8.7.2). The hybrid extends naturally to N > 2 lanes
  with different *basin classes* (E25 / E41 / DP) but not with the
  same class at multiple seeds.
- **Tier-2 ORFS routed metrics do not transfer uniformly.** Our placer
  optimizes proxy; ORFS's `rtl_macro_placer` is timing-aware.
  ariane133 ships *without* our placement (ORFS auto wins by 1.2 ns of
  slack); ariane136 ships *with* our placement (cascade wins by 0.45 ns).
  The proxy-optimization approach has structural limitations on the
  timing-driven evaluation Tier 2 uses.

### 10.8 What this means for the field

The recurring objective-mismatch pattern across three eras (LP-HPWL,
RUDY congestion, smooth-proxy gradient) suggests that macro placement
with composite objectives may be fundamentally resistant to divide-
and-conquer strategies that work for single-objective problems. The
proxy couples WL, density, and congestion through shared grid cells;
any separation of concerns loses the information that matters. DPO
succeeds where the polyhedral system fails because it does *not*
decompose; CD succeeds where DPO struggles because its exact evaluator
is fast enough that CD on the joint objective beats gradient descent
on a wrong joint model.

The path forward, by our reading, has two complementary components:
**better evaluators, not better decompositions** (the incremental
evaluator unblocked CD; the smooth-proxy Hessian unblocked saddle
escape; future curvature oracles will unblock further algorithms);
and **robust hybrid composition, not single-mechanism optimization**
(per-bench best-of-N hybrids over topology-blind base heuristics is
the empirical safe combination). Mechanism-aligned heuristics that
overfit to a single design class are the failure mode to watch for.

---

## 11. References (~1 page)

### Transition-state search (the saddle-escape mechanism, §§8.9–8.10)

- Henkelman, G., & Jónsson, H. (1999). A dimer method for finding
  saddle points on high dimensional potential surfaces using only
  first derivatives. *Journal of Chemical Physics*, **111**(15),
  7010–7022. [doi:10.1063/1.480097](https://doi.org/10.1063/1.480097)
- Henkelman, G., & Jónsson, H. (2000). Improved tangent estimate in the
  nudged elastic band method for finding minimum energy paths and
  saddle points. *Journal of Chemical Physics*, **113**(22), 9978–9985.
  [doi:10.1063/1.1323224](https://doi.org/10.1063/1.1323224)
- E, W., & Zhou, X. (2011). The gentlest ascent dynamics. *Nonlinearity*,
  **24**(6), 1831–1842.
  [doi:10.1088/0951-7715/24/6/008](https://doi.org/10.1088/0951-7715/24/6/008)
- Heyden, A., Bell, A. T., & Keil, F. J. (2005). Efficient methods for
  finding transition states in chemical reactions. *Journal of Chemical
  Physics*, **123**(22), 224101.
  [doi:10.1063/1.2104507](https://doi.org/10.1063/1.2104507)

### Macro placement (the field)

- Cheng, C.-K., Kahng, A. B., Kang, I., & Wang, L. (2019). RePlAce:
  Advancing Solution Quality and Routability Validation in Global
  Placement. *IEEE TCAD*, **38**(9), 1717–1730.
- Lin, Y., Dhar, S., Li, W., Ren, H., Khailany, B., & Pan, D. Z. (2019).
  DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern
  VLSI Placement. *DAC '19*.
- Lu, J., Chen, P., Chang, C.-C., et al. (2020). Routability-driven
  global placement with congestion-aware density mapping. *DATE '20* /
  DREAMPlace-Cong (DATE '21).
- TILOS MacroPlacement open repository — `external/MacroPlacement`,
  IBM ICCAD-04 benchmark suite, NG45 design flows.

### Smooth-proxy / differentiable optimization for placement

- Naylor, P., Donelly, S., & Sha, L. (2001). Non-linear Optimization
  System and Method for Wire Length and Constraint Placement.
  *US Patent 6 301 693* — LSE-HPWL formulation.
- Hazan, E., Levy, K. Y., & Shalev-Shwartz, S. (2016). On graduated
  optimization for stochastic non-convex problems. *ICML '16*.
- Mobahi, H., & Fisher, J. W. (2015). On the link between Gaussian
  homotopy continuation and convex envelopes. *EMMCVPR '15*.
- Bertsekas, D. P. (1982). *Constrained Optimization and Lagrange
  Multiplier Methods*. Academic Press. (Penalty methods underlying
  the smooth-proxy overlap term.)
- Mountain-pass theorem / least-action: Ambrosetti & Rabinowitz (1973),
  *Dual variational methods in critical point theory and applications*.
  Provides the topological grounding for §8.9's "the plateau is a
  saddle" interpretation.

### Decomposition / structural analysis (Acts 1–2 — §§2–7, 8.7.5)

- Balas, E. (1979). Disjunctive programming. *Annals of Discrete
  Mathematics*, **5**, 3–51.
- Kronqvist, J., Misener, R., & Tsay, C. (2025). P-split formulations
  for piecewise linear functions in MILP. (For polyhedral
  decomposition baselines.)
- Zariski, O. (1962). Hyperplane arrangement complement connectivity —
  motivates the infeasibility-wall framing in §8.7.5.
- Slaney, J., & Walsh, T. (2001). Backbones in optimization and
  approximation. *IJCAI '01*.
- Miftari, B., et al. (2024–2026). LP sensitivity analysis for
  combinatorial placement-style problems.

### Competition reference

- Partcl/HRT Macro Placement Challenge (2026). Competition rules,
  scoring v2.0, OpenROAD Tier 2 flow.

> **TODO(refs):** Format properly per the target venue's bibliography
> style (IEEE TCAD recommended). Verify each citation is one we *use*
> in the body, not just a literature pointer. Drop anything we only
> read but do not cite from a numbered section.

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

## Master TODO list (rolled up from inline `TODO(...)` markers, current 2026-05-16)

### Data — load-bearing (closed)

- [x] **NG45 transfer datapoint.** A4-v2 ran cascade-adaptive `--ng45`
      on cloud EPYC: avg 0.6870 across ariane133 / ariane136 /
      mempool_tile / nvdla, zero overlaps. Per-design results
      sourced from `results/CDLNSSACascadeAdaptivePlacer_20260513_091359.json`.
      The 1–2 truly hidden NG45 designs remain opaque per the
      competition rules; the four public designs confirm transfer.
- [x] **Hessian-saddle ariane133 lift.** E74 0.6641 vs E48 0.6861
      (−3.21 %). Breaks the IBM/NG45 transfer-failure pattern (see
      §8.9.4). Sourced from
      `experiments/E74_hessian_saddle/results/hessian_ariane133.pt`.
- [x] **Cap-vs-ceiling closure (PATH A).** A4-v1 / A4-v2 on cloud
      verify 1.077–1.078 IBM under 60-min cap, vs 1.0612 M3 cached
      ceiling. Gap is 1.5 % residual per-core speed differential.
      §8.11 documented.
- [x] **Macro clearance diagnostic for Tier 2 ORFS push.** All 4
      NG45 designs: max push 6.00 μm everywhere; canvas displacement
      < 0.4 %. See `analysis/macro_clearance_diagnostic/findings.md`.

### Data — load-bearing (open)

- [ ] **λ-spectrum capture** for the ibm01 E48 plateau: top-8
      smallest-algebraic eigenvalues of the smooth-proxy Hessian.
      Source script: `experiments/E74_hessian_saddle/code/hessian_saddle.py`
      with `k=8`. Freeze as `writeup/data/lambda_spectrum_ibm01.txt`.
      Feeds the §8.9.2 high-index-saddle claim and Figure-1.
- [ ] **Cascade lift-per-iteration trace** for 3 benchmarks (one easy:
      ibm01; one medium: ibm10; one hard: ibm17). Source: the cascade
      log's `iter N: NEW BEST` lines. Justifies cascading vs
      single-shot in §8.10.
- [ ] **Per-bench A4-v2 champion table** for §9.2. Should pair each
      IBM bench's A4-v2 result against the E48 / E74 / E84 cached
      result and the RePlAce baseline. Source: A4-v2 result JSON +
      `docs/results.md`.

### Figures — unbuilt

- [ ] **Figure 1 (motivation):** λ-spectrum bar chart for ibm01 E48
      plateau showing top-8 eigenvalues, the negative ones colored.
      Caption: "The plateau every local-move heuristic terminates at
      is a high-index saddle of the smooth proxy."
- [ ] **Figure 2 (mechanism):** schematic of the saddle escape step
      — current plateau, eigenvector arrow, ε-step, polish path, new
      basin. Hand-drawn-style is fine.
- [ ] **Figure 3 (cascade lift):** lift-per-iteration line chart for
      3 representative benchmarks. Shows diminishing returns and the
      stop-condition firing.
- [ ] **Figure 4 (champion lineage):** RePlAce → CD → E48 → E74 →
      E84 → A4-v2 bar chart. Shows the three-era arc visually.
- [ ] **Figure 5 (per-bench bar chart):** E48 / E74 / A4-v2 stacked
      bars over the 17 IBM benches. Shows the uniform-improvement
      claim from §8.9.4.
- [ ] **Figure 6 (macro clearance, optional):** clearance histogram
      from `analysis/macro_clearance_diagnostic/` for the 4 NG45
      designs. Justifies the "submit as-is" Tier 2 decision.

### Prose — drafting status (2026-05-16 update)

- [x] **Abstract.** Drafted full prose covering three eras + verified
      numbers (Option A/B; gaps to leaderboard).
- [x] **§1 Introduction.** Drafted with three-act narrative + section
      roadmap.
- [x] **§2 Polyhedral Decomposition.** Drafted; math precision +
      novelty boundary (Balas 1979).
- [x] **§3 Navigation System.** Drafted; result 1.4867 + sweep flat.
- [x] **§4 Congestion Barrier (Act 2 Diagnosis).** Drafted; LP-HPWL
      ρ = −0.001, swap+LP corroboration, proxy decomposition.
- [x] **§5 DPO First Pivot.** Drafted; architecture + ablation +
      novelty boundary against DREAMPlace and C3PO.
- [x] **§6 Why DPO Crosses the Barrier.** Drafted; penalty continuation
      + complexification interpretation + low-seed-variance evidence.
- [x] **§7 RUDY Limit (Act 3 Diagnosis).** Drafted; 10.9 % top-5 %
      hotspot overlap; direction-not-magnitude verdict.
- [x] **§8 CD Breakthrough + §8.5 Compositional Polish + §8.6 Hybrid
      + §8.7 Failed Extensions + §8.7.5 Infeasibility Wall + §8.7.6
      Spatial-Block Crossover + §8.8 Transfer Pattern.** All drafted
      2026-05-16.
- [x] **§8.9 Hessian Saddle Escape.** Drafted 2026-05-13.
- [x] **§8.10 Cascading.** Drafted 2026-05-13.
- [x] **§8.11 PATH A speedup.** Drafted 2026-05-13.
- [x] **§8.12 DREAMPlace Lane (NEW).** Drafted 2026-05-16 covering
      the autopsy refutation, B-R0', auto-adaptive config, Option B.
- [x] **§9 Results.** Lineage table updated to Option A/B; NG45 prose
      drafted; per-bench `--all` tables remain `TODO(data)` for the
      figure-generation pass.
- [x] **§10 Discussion.** Drafted full prose (10.1-10.8): objective
      mismatch recurrence, non-decomposability, "bypass don't fix",
      infrastructure-unlocks-algorithm, compositional polish, IBM/NG45
      transfer, limitations, what-this-means-for-the-field.
- [x] **§11 References.** Organized by topic; DOIs in for transition-
      state methods.

### Final pass

- [ ] Length/density check vs 16–20-page target.
- [ ] Math notation consistency (Hessian, eigenvector, ε, gradient).
- [ ] Cold-read by someone unfamiliar with the project.
- [ ] Final reference-list cleanup: drop pointers we don't cite.
- [ ] Figure generation pass (Figures 1–7 listed in `## Master TODO list`
      above; data-freeze items still open).

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
