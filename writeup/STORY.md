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
materials science to escape it — landing 1.0771 vs RePlAce 1.4578 (−26.1 %)
under partcl's 60-min cap, with zero overlaps and no per-benchmark tuning.**

That sentence carries the headline: a *new mechanism* (the saddle reframing),
a *verified result* on the field's reference baseline, and the
*generalization claim* (no tuning, uniform across 21 designs).

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

**Headline result:** 1.08151 → 1.0612 (uncapped ceiling, −1.88 %) →
**1.0771 under 60-min cap** (−5.2 % vs morning baseline). All zero overlaps,
no per-benchmark tuning, CPU-only Python.

**Key insight (the paper's strongest claim):** *The plateau every prior
local-move placement heuristic reaches is structurally a high-index saddle
of the smooth proxy, and the literature of transition-state methods —
climbing-image NEB (Henkelman & Jónsson 2000), the dimer method (Henkelman
& Jónsson 1999), gentlest-ascent dynamics (E & Zhou 2011) — provides
principled machinery to escape it.* This is the connection the paper exists
to make.

**Paper sections:** §§8.9–8.11 (scaffold added 2026-05-13 in commits
`7efc99e`, `438007c`; prose pending).

---

## Formal-paper section mapping

Standard CAD-paper structure → our content.

| Paper section | Our content | Status |
|---|---|---|
| **Abstract** | Three-act compressed; headline 1.0771 / NG45 0.6870 vs RePlAce 1.4578 | scaffold + TODO(prose) |
| **§1 Introduction** | Problem statement, proxy definition, baselines, contributions list, paper organization | scaffold |
| **§2 Background** | Proxy decomposition (E8), prior approaches (RePlAce, leaderboard tops), competition setup | needs new section |
| **§3 Diagnostic Phase — discovering CD** | §§4–8 of current paper merged + tightened | scaffold + needs prose |
| **§4 Compositional Polish** | §§8.5–8.6 (E25/E18/E41 layers + E48 hybrid) | scaffold |
| **§5 The Infeasibility Wall + Failed Extensions** | §§8.7–8.8 (E53–E54 wave + E65 wall + E61_v2) | scaffold |
| **§6 Hessian Saddle Escape (CENTRAL INNOVATION)** | §§8.9–8.10 (E74 + E84) | scaffold added 2026-05-13 |
| **§7 Implementation: closing the cap-vs-ceiling gap** | §8.11 (A1 speedup) | scaffold added 2026-05-13 |
| **§8 Empirical Results** | §9 (lineage table extended through A1) | table updated; per-bench TODO |
| **§9 Discussion** | §10 (objective mismatch, bypass-don't-fix, compositional, transfer pattern, saddle insight) | scaffold + needs prose |
| **§10 Related Work** | NEB / dimer / GAD references + macro placement literature | TODO — does not currently exist |
| **§11 Conclusion** | The three-act recap + the saddle insight as a transferable principle | TODO |

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

2. **§2 Background** — proxy decomposition formal statement, what
   PlacementCost computes, what compute_proxy_cost returns. Currently
   distributed across §§1, 4, 7.

3. **Prose for the central innovation sections (§§6–7)** — currently
   `TODO(prose)` blocks. These are the highest-leverage prose to write
   because they're what the paper exists to communicate.

4. **Per-benchmark champion table** for §8 (E84 / A4-v2 per-bench numbers
   alongside RePlAce baseline). Currently a TODO in §9.2.

5. **Figures.** Several already TODO'd:
   - λ-spectrum plot for ibm01 plateau (motivates the high-index-saddle claim)
   - Cascade lift-per-iteration plot (justifies cascading vs single-shot)
   - Per-bench E48 vs E74 vs E84 vs A4-v2 bar chart
   - Macro-clearance histogram from `analysis/macro_clearance_diagnostic/`

6. **Abstract prose** — currently a TODO with bullet points; needs
   ~250 words of clean prose hitting the three-act arc.

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
