# ADR-012a: Promote E61_v2 spatial-block GA crossover hybrid (best-of-{E48, E61_v2}) as champion

> **Status: Proposed → Superseded before acceptance.** This ADR claimed
> slot 012 first (2026-05-03) but was never moved past *Proposed*. Two
> days later, ADR-012 (E74 Hessian saddle promotion) was *Accepted*
> 2026-05-05, taking the canonical ADR-012 slot. This file is renumbered
> to **012a** and retained for the historical record. References to
> "ADR-012 *Proposed*" in earlier docs may point here.

**Status:** **Proposed → superseded before acceptance** (renumbered
2026-05-16 to resolve filename collision with ADR-012 E74). E61_v2
spatial-block GA crossover finished `--all` overnight 2026-05-03
07:32 EDT: avg **1.08083**, zero overlaps, ~6.7 hr wall-clock
(`--jobs 4` parallel; 95871 s aggregate CPU-time across workers).
Standalone delta vs E48 is **−0.07 %** (marginal, fails the
−0.30 % single-pipeline promotion threshold by ~5×). The promotion
case rests on two non-standalone facts: (1) **--ng45 lift**
0.6908 vs E48 0.6922 (−0.20 %), with **ariane133 −1.47 %** — first
NG45-positive mechanism since E18; (2) **per-bench hybrid lift**
best-of-{E48, E61_v2} = 1.08025 (−0.12 %). Champion remains E48
(ADR-011 *Accepted*); E61_v2 code stays at
`experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`.

**Date:** 2026-05-03 (proposed)
**Deciders:** project owner

## Context

After E48 hybrid (ADR-011 *Accepted*, 1.08151) was promoted, six
overnight follow-ups (2026-05-01 → 03) tested mechanisms targeted at
**lifting the SDF/DPO basin pair to a third basin**:

| Hypothesis | Verdict |
|---|---|
| E53 GPU DPO basin polish | falsified (0/350 GPU restarts accepted at full budget) |
| E53m multi-seed hybrid (E25 + E41 s42 + E41 s1) | marginal at --all (1.08128, −0.02 %) |
| E54 congestion-targeted destroy | falsified on --ng45 (ariane133 +5.14 %) |
| E62 WillSeed init lane | falsified on --fast (+1.44 % vs E48; loses every fast bench) |
| E63 spectral init | blocked on legalization (V2 Hungarian + V3 row-pack both fail boundary cases) |
| E65 NEB cross-section | structural finding: SDF↔DPO basins separated by INFEASIBILITY WALL, not proxy barrier |
| **E61 V1** macro-level GA crossover | falsified (136 unrecoverable overlaps from per-macro Bernoulli) |
| **E61 V2** spatial-block GA crossover | **MARGINAL --all + KEY --ng45 lift** (this ADR) |

E65's infeasibility-wall finding was the unifying lesson: linear
interpolation between SDF and DPO placements never legalizes (0/9
intermediate k-values on both ibm01 and ibm12 cross-sections). The
wall is a function of *spatial-configuration distance*, not proxy
distance — ibm12 endpoints differ by 0.2 % in proxy yet are separated
by a 233-residual feasibility gap. This rules out any mechanism that
bridges basins at the *solution* level via per-macro recombination.

E61_v2 threaded through the wall by operating at **block granularity**:
2×2-quadrant chunks of macros from each parent are *internally* feasible
(each block was internally consistent in its parent), so block-level
recombination preserves local feasibility while still mixing global
topology. Failed E61_v1 → succeeded E61_v2 is direct empirical
confirmation of the wall finding's shape.

## Decision (proposed)

**Two options to consider:**

**Option A: Promote E61_v2 standalone as champion.**
- Standalone --all 1.08083 vs E48 1.08151 = **−0.07 %**
- Misses standard −0.30 % promotion threshold by ~5×.
- Argues against simple promotion — the standalone delta is in the
  noise floor of run-to-run DPO seed variance.

**Option B (recommended): Promote a 3-lane hybrid as champion.**
- Extend `cd_lns_sa_hybrid/placer.py` to add `CDLNSGACrossoverPlacer`
  (E61_v2) as a third lane alongside E25 and E41.
- Per-bench best-of: **1.08025** (−0.12 % vs E48 standalone).
- E61_v2 internally already runs E25 + E41, so the 3-lane hybrid
  shares those internal placers — no redundant work; total wall
  ≈ E61_v2 wall (~6.7 hr `--jobs 4`).
- NG45: best-of-{E48, E61_v2} on commercial designs gives ariane133
  break for free (E61_v2 lane wins ariane133 by −1.47 %).
- Promotion threshold: 1.08025 < E48 1.08151 by 0.12 %, *also*
  marginal but compounds with the NG45 advantage.

**Option C: Defer.** Keep E48 as champion, document E61_v2 + E65
infeasibility-wall finding for the writeup, prioritize E63 spectral
legalization fix (~2-4 hr) before re-evaluating. Reasonable if the
human values single-pipeline simplicity over hybrid lane addition.

## Consequences (final, if Option B accepted)

- **Champion:** 1.08151 (E48) → **1.08025** (3-lane hybrid).
  Improvement:
  - vs E48 1.08151: **−0.00126, −0.12 %**
  - vs E12 1.0990: **−0.0188, −1.71 %**
  - vs leaderboard 1.1172: **−0.0370, −3.31 %**
  - vs RePlAce 1.4578: −0.378, −25.9 %
- **NG45 transfer:** 0.6922 (E48) → expected ≤0.6908 (lower of
  per-design min); ariane133 secured at 0.6760 (−1.47 % vs E48 lane).
- ADR-011 stays *Accepted* (E48 was the right step); ADR-012
  describes the next step.

### Per-bench `--all` (zero overlaps everywhere)

| Bench | E48 | E61_v2 | best-of-2 | Δ vs E48 |
|---|---:|---:|---:|---:|
| ibm01 | 0.89234 | **0.89017** | 0.89017 | −0.24 % |
| ibm02 | **1.11630** | 1.11947 | 1.11630 | tied |
| ibm03 | **0.95532** | 0.95615 | 0.95532 | tied |
| ibm04 | **0.98506** | 0.98742 | 0.98506 | tied |
| ibm06 | 1.15315 | **1.15238** | 1.15238 | −0.07 % |
| ibm07 | **1.09854** | 1.10068 | 1.09854 | tied |
| ibm08 | 1.10331 | **1.10315** | 1.10315 | −0.01 % |
| ibm09 | 0.84129 | 0.84129 | 0.84129 | tied |
| ibm10 | 1.00961 | **1.00957** | 1.00957 | tied |
| ibm11 | 0.87651 | 0.87651 | 0.87651 | tied |
| ibm12 | 1.20639 | **1.19821** | 1.19821 | **−0.68 %** |
| ibm13 | 0.94779 | **0.94776** | 0.94776 | tied |
| ibm14 | 1.19951 | **1.19387** | 1.19387 | **−0.47 %** |
| ibm15 | 1.16515 | **1.16176** | 1.16176 | **−0.29 %** |
| ibm16 | **1.14404** | 1.14518 | 1.14404 | tied |
| ibm17 | 1.33240 | **1.33180** | 1.33180 | −0.05 % |
| ibm18 | 1.35892 | **1.35849** | 1.35849 | −0.03 % |
| **AVG** | **1.08151** | **1.08083** | **1.08025** | **−0.12 %** |

E61_v2 wins 10/17 per-bench, mostly by small margins; E48 wins 5/17;
2 exact ties. Big E61_v2 wins concentrate on tied-parent benches
(ibm12 / ibm14 / ibm15 — where E25 and E41 are within 1.7 %),
confirming the mechanism's claim: spatial-block crossover finds new
basins on basin-symmetric benches, reverts to the dominant parent on
asymmetric ones.

### Per-bench `--ng45` (zero overlaps everywhere)

| Design | E48 | E61_v2 | Δ |
|---|---:|---:|---:|
| ariane133 | 0.6861 | **0.6760** | **−1.47 %** |
| ariane136 | **0.6685** | 0.6737 | +0.78 % |
| mempool_tile | 0.7375 | 0.7362 | tied |
| nvdla | 0.6767 | 0.6772 | tied |
| **AVG** | 0.6922 | **0.6908** | **−0.20 %** |

ariane133 lift is structurally significant — it breaks the consistent
NG45 failure point that killed E42 K=4 (+3.57 %), E43 longer K-joint
(+4.10 %), E44 spatial K-tuple, E54 congestion-destroy (+5.14 %),
E62 WillSeed init.

## Open questions / follow-ups

1. **Does the marginal --all delta survive a multi-seed re-verification?**
   E61_v2's --all 1.08083 might fluctuate ±0.003 across DPO seeds (per
   E52 measurement). Re-running with seed=1 would tell us if 1.08083
   is reproducible or noise.
2. **Can the spatial-block grid be tuned?** E61_v2 uses 2×2 quadrants;
   3×3 / 4×4 might find different per-bench winners. Hyperparameter
   sweep is cheap (~3 hr per grid choice on --fast).
3. **Does E61_v2 compose with E63 (when fixed-aware row-pack lands)?**
   3-basin best-of {E25 SDF, E41 DPO, E63 spectral} + E61_v2 crossover
   on each pair would extend the hybrid further.
4. **NG45 multi-design verification.** Currently E61_v2 NG45 is 4
   designs; running on more NG45 benches (e.g., the full top-7) would
   strengthen the ariane133 finding's generalization.

## Alternatives ruled out

- **E53m multiseed hybrid (1.08128, −0.02 % vs E48).** Already a
  hybrid lane candidate but contributes only −0.02 % at --all — within
  DPO seed noise. Adding it as a 4th lane gives best-of-3 = 1.07995
  (−0.14 %); marginal compounding.
- **E63 spectral standalone.** Currently blocked on legalization;
  per-bench wins TBD post-legalization fix.
- **E54 congestion-targeted destroy.** Falsified on NG45 ariane133
  (+5.14 %).
- **E62 WillSeed init lane.** Falsified on --fast (loses every IBM
  fast bench).
- **E64 LP-bounded beam K-joint.** Still scaffolded; multi-day dev.
  Lower priority than ADR-012 decision.

## Evidence

- `experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py` — placer code.
- `experiments/E61_ga_crossover/manifest.md` — full outcome.
- `experiments/E61_ga_crossover/results/best_of_analysis.md` — best-of-N
  computation from `scripts/best_of_n_analysis.py`.
- `experiments/E65_neb_cross_section/manifest.md` — infeasibility-wall
  structural finding that explains why E61 V1 failed and V2 works.
- `results/experiment_log.jsonl` rows:
  - `E61_v2_spatial_block_fast` 0.91342 (`--fast`, 2026-05-02)
  - `E61_v2_spatial_block_ng45` 0.69078 (`--ng45`, 2026-05-02)
  - `E61_v2_spatial_block_all_overnight` **1.08082**
    (`--all`, 2026-05-03 07:31)
- E25 component: `submissions/cd_lns_sa/placer.py` (CDLNSSAPlacer).
- E41 component: `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`.

## Recommendation

**Option B** — promote a 3-lane hybrid via a thin extension of
`cd_lns_sa_hybrid/placer.py`. The marginal --all standalone delta is a
weak argument for simple promotion, but the NG45 ariane133 lift is
structurally meaningful and the per-bench hybrid contribution is
real. Promoting the 3-lane hybrid captures both signals at minimal
implementation cost; deferring loses the NG45 advantage on every
commercial design with structure resembling ariane133.
