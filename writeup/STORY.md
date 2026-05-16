# Story Arc — Source-of-Truth Narrative for the Paper

> Purpose: organize the development narrative into the standard formal-paper
> structure. Each section here = a paper section, with the beat we're hitting,
> the key insight, the citing material, and what's still TODO before the
> paper is publication-ready.
>
> Use this document to decide *what each paper section is for* before
> drafting prose in `paper.md`. The acts here correspond to paper sections
> via the `(→ §X)` annotations.

---

## The one-sentence pitch

**We diagnose macro placement's local-move plateau as a high-index saddle of
the smooth proxy, then apply transition-state methods from chemistry /
materials science to escape it — landing 1.07820 (cascade alone, Option A)
or 1.06650 (cascade + DREAMPlace 3rd lane, Option B) on 17 IBM
benchmarks vs RePlAce 1.4578 (−26.9 % at Option B), under partcl's 60-min
cap, with zero overlaps and no per-benchmark tuning.**

That sentence carries the headline: a *new mechanism* (the saddle reframing),
a *verified result* on the field's reference baseline, the *generalization
claim* (no tuning, uniform across 21 designs), and the *empirical refutation
of a documented PATH B autopsy* (DREAMPlace's basin, under matched polish,
opens a third valley the cascade cannot reach alone — §8.12).

---

## Three-act structure

The development arc has three eras. Each era opens with an *empirical wall*,
the team's *diagnostic move*, and the *mechanism* that breaks the wall.
Together they tell a coherent case study of diagnosis-driven optimization.

### Act 1 — Foundation: Diagnosis discovers CD (→ §§1–8)

**Wall:** Initial approaches (polyhedral decomposition, differentiable proxy
optimization / DPO) plateau at 1.3–1.5, well above the leaderboard target.

**Diagnostic moves:**
- E8 LP-HPWL decomposition reveals the proxy is **6 % WL / 20 % density /
  74 % congestion**. The early methods all under-weighted congestion (→ §4).
- RUDY-fidelity diagnostic (→ §7) explains why DPO can't reach the
  congestion frontier: its smooth-RUDY surrogate is mis-calibrated against
  the canonical scoring proxy.

**Mechanism:**
- *Incremental proxy evaluator* (E1, 4657× speedup) makes single-macro
  cost queries cheap → enables direct coordinate descent on the full
  canonical proxy.
- *Full-proxy CD* (E9 + plateau detection) escapes the DPO basin and
  reaches **1.0990** (CDLNSGridBin / E12 with grid-bin LNS overlay).

**Headline result:** 1.4578 → 1.0990 (−24.6 %) by a single architectural
pivot motivated by two diagnostic findings.

**Key insight (paper-level):** *Bypass, don't fix.* Rather than fixing
DPO's surrogate calibration, sidestep the surrogate and CD directly on the
canonical proxy. This recurs at every era.

**Paper sections:** §§1–8 (already drafted as scaffold; prose pending).

---

### Act 2 — Compositional polish: Each mechanism a different "move type" (→ §§8.5–8.8)

**Wall:** CD + grid-bin LNS converges to 1.0990 but cannot lift further.
Different polish heuristics applied individually each gain ≤ 0.3 %.

**Diagnostic move:**
- *Reachable-set analysis*: every prior heuristic — CD breakpoints, grid-bin
  LNS, SA Metropolis, K-macro joint LNS — operates on **one or a few macros
  at a time**. They all share the same reachable set under local moves and
  converge to the same fixed point.
- *Multi-basin analysis*: SDF and DPO inits land in **structurally distinct
  basins** post-polish (E18 vs E25). Per-bench best-of exploits the
  heterogeneity.

**Mechanism:**
- *Compositional polish stack*: layer SA-v2 (E25), DPO init (E18), K-joint
  LNS (E41). Each adds an orthogonal *move type* the prior stack didn't
  reach.
- *Per-bench best-of hybrid* (E48): run two pipelines (SDF-basin and
  DPO-basin) per benchmark, pick the lower-proxy output **by score, not by
  benchmark identity**. Rule-neutral; algorithmically valid; reaches
  **1.08151**.

**Headline result:** 1.0990 → 1.08151 (−1.7 %) by stacking three orthogonal
local-move polish mechanisms.

**Key structural finding:** *The infeasibility wall* (E65). Two
proxy-equivalent placements (E25 SDF basin and E41 DPO basin on the same
benchmark) are separated by a **thick infeasibility wall** in spatial-
configuration space — linear interpolation between them gives 0 / 9
legalizable intermediate placements. This formalizes why per-element
recombination (E61 V1) fails and motivates the saddle-escape framing in
Act 3.

**Failed extensions** (the May 1–2 wave, → §8.7): GPU DPO basin polish
(E53), congestion-targeted destroy (E54), K=4 joint LNS (E42), longer
K-joint (E43), spatial K-tuple (E44). Each plateaus near E48 on IBM and
regresses on ariane133 NG45. The pattern reveals the **IBM/NG45 transfer-
failure class** of IBM-aware mechanisms (→ §8.8) — heuristics that engage
the proxy's structural decomposition (6 / 20 / 74 %) depend on dense macro
packing and fail on sparse commercial designs.

**Paper sections:** §§8.5–8.8 (already drafted as scaffold; prose pending).

---

### Act 3 — Saddle escape: The plateau is not a local minimum (→ §§8.9–8.11)

**Wall:** E48 hybrid + all Act-2 extensions plateau at 1.08. The May 1–2
wave proves the local-move family has been exhausted.

**Diagnostic move (central to the paper):**
- *Hessian eigenanalysis* at the E48 plateau. Compute the smooth-proxy
  Hessian via `torch.autograd.functional.hvp` and find smallest-algebraic
  eigenvectors via Lanczos. **Top-4 eigenvalues on ibm01 plateau:
  (−0.14, −0.076, −0.066, −0.060) — all negative**. Same pattern on
  ibm04: (−0.69, −0.11, −0.07, −0.03).
- **The plateau is a high-index saddle of the smooth proxy, not a true
  local minimum**. The local-move family's reachable set covers the
  positive-curvature manifold around the saddle but cannot escape along
  the negative-curvature directions, because those directions correspond
  to *coordinated multi-macro displacements* no local-move set produces.

**Mechanism (the core innovation):**
- **Hessian saddle escape** (E74, → §8.9). At the plateau:
  1. Find the smallest-algebraic eigenvector `v_min` of the smooth-proxy
     Hessian via Lanczos.
  2. Step `p ← p ± ε · v_min` for `ε ∈ {0.3, 1.0, 3.0}`, both signs.
  3. Project to feasible (geometric overlap projection).
  4. CD-polish on the **canonical** proxy (the smooth proxy is the
     curvature oracle, never the cost).
  5. Keep the best polished state.

  Verified `--all` 1.0666 (−1.38 % vs E48 1.08151). Most importantly:
  **ariane133 (NG45) −3.21 %** — the first mechanism since CD itself to
  improve uniformly across all 21 benchmarks, breaking the IBM/NG45
  transfer-failure pattern from Act 2.

- **Cascading saddle escape** (E84, → §8.10). Iterate the escape: after
  each polish, re-compute the Hessian, check the smallest eigenvalue, and
  if still negative, escape again. Stops when (a) all eigenvalues are
  non-negative, (b) iteration produces no improvement, (c) max iters, or
  (d) wall budget exhausts.

  Verified uncapped on M3: **1.0612 IBM `--all`** — the *ceiling* of the
  cascade approach; further saddle escape no longer improves.

- **Implementation: PATH A speedup** (A1, → §8.11). The CD inner loop's
  (move → cost → revert) probe pattern is the wall-budget bandit (96 % of
  CD time per cProfile). Replace with:
  1. `delta_cost(macro_idx, new_xy)` — no-mutation cost peek. 1.95× per probe.
  2. `delta_cost_axis_batch` — batched K-candidate eval with amortized
     subtract-old precompute and batched torch ops. 4.32× combined.
  3. `_net_cong_contrib_flat` — vectorized pin → gcell, inlined 2-pin and
     multi-pin routing, no dict allocation. 5.36× combined.

  Under partcl's 60-min cap on EPYC: **1.0771 cloud (post-A1)** vs **1.137
  cloud (pre-A1)** — closes the gap from the cascade ceiling (1.0612) to
  within 1.5 %.

**Headline result (cascade era):** 1.08151 → 1.0612 (uncapped ceiling,
−1.88 %) → **1.07820 under 60-min cap** (Option A, the cap-bound floor).
All zero overlaps, no per-benchmark tuning, CPU-only Python.

**Key insight (the paper's strongest claim):** *The plateau every prior
local-move placement heuristic reaches is structurally a high-index saddle
of the smooth proxy, and the literature of transition-state methods —
climbing-image NEB (Henkelman & Jónsson 2000), the dimer method (Henkelman
& Jónsson 1999), gentlest-ascent dynamics (E & Zhou 2011) — provides
principled machinery to escape it.* This is the connection the paper exists
to make.

**Paper sections:** §§8.9–8.11 (drafted prose).

---

### Act 4 (epilogue) — DREAMPlace as a third init lane (→ §8.12)

**Wall:** PATH B (DREAMPlace integration) was documented as falsified
on 2026-05-10 after a 25-config sweep concluded that DP basins were
*structurally below* cascade-capped outputs on every hard benchmark
(ibm10 +16.5 %, ibm12 +5.2 %, ibm14 +9.7 %, ibm17 +5.8 %). The wave
of E92–E98 work that followed targeted modifications to DP, on the
assumption that the stock basin was unrecoverable.

**Diagnostic move (the autopsy refutation):** Re-reading the autopsy
driver revealed an asymmetry — the SDF and DPO lanes received the full
~1100 s polish budget each; the DP lane received only **60 s of CD
cleanup**, two orders of magnitude less. The autopsy table compared
DP basins after 60 s of polish against SDF/DPO basins after full
polish. The original conclusion ("DP polishes 10–23 % worse") was a
statement about *polish budget*, not basin quality.

**Mechanism (E91, → §8.12):**
- Stock DREAMPlace (auto-adaptive `target_density = clip(macro_density
  × 1.5, 0.40, 0.85)` — rule-compliant, no per-bench dispatch).
- `greedy_macro_legalize` to resolve DP's residual overlaps.
- Matched polish budget: CD-adaptive ≤ 660 s + LNS ≤ 200 s + SA-v2
  ≤ 200 s — same as E25 and E41 receive.
- Cascading saddle escape on the polished plateau.

**Result on the 3 hardest IBM benches that the autopsy ruled out:**
- ibm12: cascade 1.3031 → DP+polish **1.129** (**−13.3 %**)
- ibm17: cascade 1.4546 → DP+polish **1.307** (**−10.2 %**)
- ibm14: cascade 1.2919 → DP+polish **1.243** (**−3.8 %**)
- ibm10: cascade 1.0775 → DP+polish 1.095 (+1.6 %; plateau-pick keeps
  the cascade winner on this bench).

Hard-bench aggregate: cascade-capped **1.2818** → DP+full-polish
**1.1936** = **−6.9 %**.

The submission architecture extends the cascade base with DREAMPlace
as a third init lane and a per-bench best-of plateau pick — the same
meta-algorithmic structure E48 used for 2 lanes (§8.6), generalized to
3. Option B (`cd_lns_sa_cascade_dp_lane/placer.py`) reaches **IBM
1.06650 / NG45 0.68086** on the verified gates; Option A
(`placer_adaptive.py`, no DREAMPlace dep) is the safer fallback at
**1.07820 / 0.68102**. Falls back to A if `DREAMPLACE_ROOT` unset.

**Key insight (the paper's second strongest claim):** *Black-box basin
generators must be evaluated under matched polish budgets before
drawing structural conclusions about basin quality. The original PATH B
autopsy was not a finding about DREAMPlace; it was a finding about
the test that produced it.*

**Paper sections:** §8.12 (drafted 2026-05-16).

---

## Formal-paper section mapping

Standard CAD-paper structure → our content.

| Paper section | Our content | Status |
|---|---|---|
| **Abstract** | Four-era compressed; headline Option B 1.06650 / NG45 0.68086, Option A 1.07820, gaps to leaderboard vmallela 1.0109 +5.6 % | **drafted** 2026-05-16 |
| **§1 Introduction** | Problem statement, proxy definition, baselines, contributions list, section roadmap | **drafted** 2026-05-16 |
| **§§2-3 Polyhedra + Navigation** | Disjunctive decomposition + 7-module nav system + 1.4867 ceiling | **drafted** 2026-05-16 |
| **§4 Congestion Barrier** | LP-HPWL ρ = −0.001 diagnostic; 6/20/74 decomposition; swap+LP corroboration | **drafted** 2026-05-16 |
| **§5 DPO First Pivot** | Architecture + ablation + novelty boundary | **drafted** 2026-05-16 |
| **§6 Why DPO Crosses Barrier** | Penalty continuation + complexification interpretation | **drafted** 2026-05-16 |
| **§7 RUDY Limit** | 10.9 % top-5 % hotspot overlap; direction-not-magnitude | **drafted** 2026-05-16 |
| **§8 CD Breakthrough** | 4657× evaluator + breakpoint enumeration + plateau detection + grid-bin LNS | **drafted** 2026-05-16 |
| **§§8.5-8.7 Compositional Polish + Hybrid + Failed Extensions** | E25/E18/E41/E48 hybrid + E53-E54 wave + E65 wall + E61_v2 | **drafted** 2026-05-16 |
| **§8.8 IBM/NG45 Transfer Pattern** | 5-experiment regression class + density hypothesis + OOD framing | **drafted** 2026-05-16 |
| **§8.9 Hessian Saddle Escape (CENTRAL INNOVATION)** | E74 mechanism + ariane133 breakthrough | drafted 2026-05-13 |
| **§8.10 Cascading** | Iterate until eigenvalue ≥ 0; 1.0612 uncapped | drafted 2026-05-13 |
| **§8.11 PATH A speedup** | 5.36× delta_cost + LNS-helper conversions | drafted 2026-05-13 |
| **§8.12 DREAMPlace Lane** | E91 autopsy refutation; Option B architecture | **drafted** 2026-05-16 |
| **§9 Empirical Results** | Lineage table through Option A/B; NG45 transfer; compute envelope | **drafted** 2026-05-16 (per-bench data table TODO) |
| **§10 Discussion** | Objective mismatch recurrence; bypass-don't-fix; compositional multiplicativity; OOD pattern; limitations; field implications | **drafted** 2026-05-16 |
| **§11 References** | NEB / dimer / GAD references + macro placement literature + OOD generalization | drafted 2026-05-13 |

---

## What's missing for a publication-ready writeup

In priority order, what to draft next:

1. **§10 Related Work** — currently absent. Needs:
   - NEB / dimer / gentlest-ascent literature (Henkelman & Jónsson, E & Zhou,
     Heyden et al., others). ~5–10 papers.
   - Macro placement state of the art (RePlAce, DREAMPlace, OpenROAD,
     analytical/quadratic placers). ~5–10 papers.
   - Transition-state methods applied to other combinatorial problems
     (rare; this is part of the novelty). ~3–5 papers if they exist.

2. **Per-benchmark champion table** for §9.2 (Option A / Option B
   per-bench numbers alongside RePlAce baseline). Per-bench data
   sourced from `results/CDLNSSACascadeDPLanePlacer_*.json` and
   `results/CDLNSSACascadeAdaptivePlacer_*.json`; needs freezing
   into `writeup/data/per_bench_champion.csv`.

3. **Figures (drafted-but-not-rendered).** Six identified:
   - Figure 1: λ-spectrum bar chart for ibm01 E48 plateau (motivates
     the high-index-saddle claim).
   - Figure 2: schematic of the saddle escape step.
   - Figure 3: cascade lift-per-iteration line chart (3 representative
     benchmarks).
   - Figure 4: champion lineage bar chart (RePlAce → CD → E48 → E74 →
     E84 → Option A → Option B).
   - Figure 5: per-bench E48 / E74 / Option A / Option B stacked bar
     chart over 17 IBM benches.
   - Figure 6: per-bench plateau-pick winner stacked bar (SDF / DPO /
     DP lane color-coded; visualizes the §8.12 architecture).
   - Figure 7 (optional): macro-clearance histogram from
     `analysis/macro_clearance_diagnostic/`.

4. **Frozen-data artifacts** for figure reproducibility:
   - `writeup/data/lambda_spectrum_ibm01.txt` (top-8 eigenvalues).
   - `writeup/data/rudy_ibm01.txt` (3.1×, 10.9 %, Jaccard 0.057).
   - `writeup/data/cascade_lift_traces/{ibm01,ibm10,ibm17}.csv`.
   - `writeup/data/lp_hpwl_correlations_ibm01.csv`.
   - `writeup/data/lane_pick_distribution.csv` (lane-pick winners
     per bench for §8.12 Figure 6).

5. **Cold-read pass.** All prose drafted but unchecked against a fresh
   reader's understanding. Pass should flag run-on paragraphs,
   inconsistent terminology (Option A vs A4-v2 vs cascade-adaptive),
   stale numbers (any remaining "1.137" / "1.0771-only" / "Cezar 1.037"
   references).

6. **Math notation consistency pass.** Hessian symbol (`H` vs `∇²f`),
   eigenvector indexing (`v_0` vs `v_min` vs `v_min(0)`), ε naming
   (`ε` vs `eps` vs `epsilon`), and gradient symbol consistency
   across §§6, 8.9, 8.10.

---

## Stylistic notes for the prose pass

- **Academic tone, but case-study framing is allowed.** The CAD community
  publishes diagnostic-driven papers (especially in TCAD); we're not
  forcing a methods-paper structure onto a development log.
- **Each section opens with the "wall" being faced.** Then the diagnostic.
  Then the mechanism. Then the result. This is the through-line.
- **Quantitative claims need a citation to the experiment** (manifest or
  evidence.md entry). No claims without numbers.
- **The "transition-state methods → combinatorial placement" connection
  IS the novelty.** Highlight it explicitly in §6 and §11; don't bury it.
- **No per-benchmark tuning** is a repeated formal claim — every figure
  / table caption that shows per-bench numbers should reaffirm this.
- **Compute envelope is the ground-truth feasibility check.** "Single
  global algorithm; ≤ 60 min per bench on EPYC; zero overlaps; CPU-only
  Python." That sentence should appear verbatim in the abstract,
  introduction, and conclusion.
