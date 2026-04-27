# Pre-writeup TODO

Work that must be completed before drafting. Organized by what it
unblocks in the writeup.

---

## Blocking: cannot write the relevant section without this

### 1. Ablation experiments for DPO (section 7)

Run each on --all. Compare avg proxy to full DPO (1.42) and SDF-only
(1.50).

| Ablation | What it isolates | How to run |
|----------|-----------------|------------|
| DPO without congestion gradient | Does dC/dp help or is density enough? | Zero out congestion loss term |
| DPO without density gradient | How much does top-10% focusing add? | Zero out density loss term |
| DPO phase 1 only (exploration) | Is multi-phase continuation needed? | Run 1 phase, legalize |
| DPO with random init (not SDF) | How much does SDF contribute? | `use_sdf_init=False`, random uniform |
| Multi-seed stability (5 seeds) | How much does the result vary? | Seeds 42,43,44,45,46 |

**Effort:** ~4-5 hours (each --all run takes ~5 min, plus code changes).
**Blocks:** Section 7 (empirical results), any claim about component
contributions.

### 2. Generalize rho=-0.001 to more benchmarks (section 4)

The Miftari correlation was measured on ibm01 only. Run the same
analysis on ibm10 (large, DPO wins big), ibm06 (small, DPO loses),
and ibm14 (medium). Need at minimum:

- LP-HPWL vs refined proxy correlation
- LP-HPWL vs refined congestion correlation
- Component breakdown (WL/density/congestion fractions)

**Effort:** 2-3 hours.
**Blocks:** Section 4 (barrier analysis) — specifically, generalizing
the claim from "on ibm01" to "across benchmarks."

### 3. Literature check: congestion gradient novelty (section 6)

Search 2020-2026 analytical placement literature for any work that
computes dC/dp (gradient of congestion w.r.t. macro positions) and
includes it in the backward pass. Key papers to check:

- Lu et al. 2020 (congestion-aware DREAMPlace) — uses congestion as
  net weights, but verify the gradient computation
- Liang et al. 2022 (DREAMPlace 4.0) — mixed-size placement
- Any ISPD/DAC/ICCAD 2021-2025 papers on congestion-driven placement
- Lee et al. ICML 2025 (diffusion placement) — check if congestion
  enters the score model gradient

If the answer is "yes, someone did this," reframe contribution 3 as
"applied to competition metric with top-k aggregation" rather than
"first congestion gradient."

**Effort:** 2-3 hours of targeted search.
**Blocks:** The core novelty claim in section 6.

---

## Strengthening: writeup works without this but is better with it

### 4. Quantify polyhedra traversal in DPO

Compare the pairwise L/R/A/B assignment between SDF init and DPO
output. Count how many pairs changed. This directly tests the
"penalty-as-barrier-crossing" claim (contribution 4). If DPO changes
50+ pair assignments, the barrier crossing is real and quantified.

**Effort:** 1-2 hours.
**Strengthens:** Section 6 (DPO theory), contribution 4.

### 5. Clean up version confusion

The experiment log shows:
- Unnamed DPO --all: 1.4361
- dpo_v1: 1.4255
- dpo_v2_moresteps: 1.4107 (best)
- dpo_v3_final: 1.4246

results.md reports 1.4246 as "DPO v3." The best run is actually v2 at
1.4107 (3.2% better than RePlAce). Clarify: is v3 the intended final
configuration? If so, why does v2 beat it? Seed sensitivity? The
writeup needs one clean number backed by multi-seed runs.

**Effort:** 1 hour to investigate + ties into ablation runs.
**Strengthens:** All results claims.

### 6. Per-benchmark figures

Generate scatter plots for the writeup:
- LP-HPWL vs proxy cost (the rho=-0.001 plot)
- Congestion vs proxy cost (the rho=0.825 plot)
- DPO vs RePlAce per-benchmark comparison bar chart
- Overnight sweep results (22 experiments, all flat)

**Effort:** Half day.
**Strengthens:** Sections 4 and 7.

---

## Parallel with writing: can happen concurrently

### 7. DPO adaptive constants + multi-seed

Implement position normalization, convergence-based phase transitions,
adaptive lambda. Run 3-5 seeds. If this pushes avg proxy below 1.40,
update section 7 numbers. If not, the writeup is no weaker.

**Effort:** 1-2 days.
**Does not block any section.**

### 8. SA polish after DPO

Add 5-10s of SA refinement with real proxy evaluation after DPO +
legalization. Expected +0.2-0.5%.

**Effort:** Half day.
**Does not block any section.**
