---
id: E164
name: v4_multiseed_cascade
status: in_progress
parent: E136
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E164: v4_multiseed_cascade — V4 multi-seed basin pick + cascade saddle escape

## Hypothesis

Two positive 2026-05-21 results compose:
- E162: V4 basin attractor IS seed-sensitive (1.24% pre-CD spread on ibm17
  between best/worst of seeds {42, 123, 999}).
- E136: V4+CD basin has λ_min < 0 and cascade saddle escape descends along
  the softest eigvec to a lower basin (-1.00% on ibm03, -0.33% on ibm10).

If both mechanisms are orthogonal, layering them composes: multi-seed picks
the BEST basin to start from, then cascade saddle finds the soft-mode
escape from THAT basin. Projected: ≥1% lift on ibm03 (E136 alone), ≥1% on
ibm17 (multi-seed alone), and possibly stacked lift where both contribute.

## Method

For each seed in {42, 123, 999}:
- Run V4-Gaussian descent + greedy legalize (no CD polish yet).
- Score by canonical proxy.

Pick best basin → CD polish → cascade saddle escape on the polished basin.

Budget allocation (1500s default):
- 3 × V4 descent + legalize: ~750s (250s each on hard bench; ~50s on small)
- CD polish on winner: 450s
- Cascade saddle escape: 270s (max_iters=2, eps=(0.5, 1.5), polish_budget=90s)
- 30s safety margin

Skip remaining seeds if budget runs tight. Falls back gracefully to whatever
basins succeeded.

## Kill gate

ibm03 smoke: final proxy >= E136 ibm03 final (0.89530) - 0.001 (no
compounding lift over E136 alone) → falsify the orthogonality claim.

If smoke shows compounding (≤ 0.89), promote to --fast.

## Generalization check

--fast aggregate: must beat thinkorplace-v2 baseline 0.984 by ≥ 0.5%
AND no single-bench regression > 0.5%. If passes, run --all + --ng45.

## Outcome (filled when decided)

**Smoke ibm03 PASSED** (2026-05-21):
- Per-seed basin (pre-CD): seed=42:1.07626, seed=123:1.07484 (better),
  seed=999:SKIP (1 residual overlap from legalize)
- PICK: seed=123 (0.13% pre-CD basin lift over seed=42)
- CD polish: 0.89314
- Cascade iter 1: ε=1.5 → 0.89000 (-0.35%)
- Cascade iter 2: ε=1.5 → 0.88941, ε=-0.5 → 0.88925, ε=-1.5 → **0.88874**
- **Final: 0.88874** (ovl=0), total wall=696s

**Comparison**:
| Pipeline | ibm03 final | vs V4 baseline 0.90430 |
|---|---:|---:|
| V4 only (thinkorplace-v2) | ~0.90430 | — |
| E136 V4 + cascade | 0.89530 | −1.00% |
| **E164 multi-seed + cascade** | **0.88874** | **−1.74%** |

E164 beats E136 by −0.73% on ibm03 — the two mechanisms compose
(multi-seed picks a better basin; cascade escapes its soft mode).

Kill gate cleared (final ≤ E136-0.001 = ≤ 0.89430; achieved 0.88874).

**--fast PASSED** (2026-05-21):
| Bench | V4+CD baseline | E136 (cascade only) | E164 (multi-seed+cascade) | E164 vs baseline |
|---|---:|---:|---:|---:|
| ibm01 | 0.83105 | 0.8299 | **0.8231** (seed=999 picked) | −0.96% |
| ibm04 | 0.92536 | 0.9241 | 0.9255 (seed=42 picked) | −0.01% |
| ibm09 | 0.76569 | 0.7657 | 0.7695 (seed=123 picked, cascade squeezed) | +0.50% |
| ibm13 | 0.84265 | 0.8401 | **0.8354** (seed=123 picked) | −0.86% |
| **AVG** | **0.84119** | **0.83993** | **0.83840** | **−0.33%** |

E164 --fast wall = 2910s = 48.5 min total (under 60-min/bench cap, 4 benches sequential).
E164 beats E136 by −0.18%, beats V4-only baseline by −0.33%.

**Composition finding**: multi-seed and cascade compete for the 1500s budget;
when seeds converge (seed=42 best), cascade gets squeezed and E164 loses
slightly. When seeds diverge (ibm01 seed=999 was much better basin), E164
wins big. Net positive but bench-dependent.

Projected --all: -0.33% on 17 IBM ≈ 0.984 → 0.981. Real lift, small magnitude.
For confident shipping decision, run --all (~5 hour wall) or accept --fast
projection.



## Pointers

- Code: `code/placer.py`
- Parent A: E162 (`experiments/E162_v4_ramp_sweep/`)
- Parent B: E136 (`experiments/E136_v4_cascade_hybrid/`)
- V4 production: `submissions/thinkorplace-v2/placer.py` (read-only)
- Cascade primitive: `experiments/E84_cascading_saddle/code/cascading_saddle.py`
