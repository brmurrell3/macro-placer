---
id: E88
name: diff_proxy_spike
status: falsified
parent: null
created: 2026-05-12
decided: 2026-05-12
champion_at_time: 1.0612         # canonical cascade --all (uncapped); wall-safe floor 1.137
outcome: "AdamW on smooth proxy WORSENS both inits on ibm01. Run A from cascade-cached (canon 0.84528) → post-leg 1.534 (+81%). Run B from SDF (canon 1.195) → post-leg 2.284 (+91%). Spike gate 0.898 not met by any descent trajectory."
champion_delta: null
graduated_to: null
superseded_by: null
---

# E88: differentiable canonical proxy — 1-day spike gate (PATH C1)

## Hypothesis
Surrogate-objective placers (DREAMPlace) failed because their loss
(HPWL + Gaussian density + DP-RUDY) is structurally misaligned with
the canonical proxy (PlacementCost: HPWL + top-K density + TILOS-RUDY
with TILOS weights). The DP `--ng45` sweep (PATH B, 2026-05-11) confirmed
this: 0/4 hardest IBM benches landed a DP basin within 5 % of cascade
after K=25 configs. If we instead **make the canonical proxy itself
differentiable** and optimize it directly on GPU, gradient descent
should land in a basin canonical actually likes — without the
surrogate gap.

## Method
1-day spike before committing the full 4–5 day C1 build.

- Stand up a standalone autograd-compatible proxy in
  `code/diff_proxy.py`: LSE-HPWL + Gaussian density (defer TILOS-RUDY to
  the full C1 build; spike judges whether the basic differentiable
  approach is viable at all).
- Import primitives directly from
  `writeup/archive/submissions/dpo/ablation_v2_steps.py`
  (`_lse_hpwl`, `_grid_density`, `_extract_net_data`). **Do not import
  from `experiments/E87_gpu_cd/`** — that's PATH A territory (the other
  Claude is iterating on it).
- Calibrate: confirm diff proxy tracks canonical on a fixed placement
  to within ~5 %, both at the cascade-cached ibm01 placement
  (canonical 0.84528) and at a few random perturbations.
- Descent: from a sensible init (cascade-cached placement or SDF),
  drive AdamW on placement positions with an overlap-penalty
  Lagrangian; anneal smoothing temperatures.
- Legalize: `greedy_macro_legalize` from
  `experiments/E76_dreamplace_integration/code/macro_legalizer.py`
  (the DP-path infrastructure that survived the PATH B falsification).
- Evaluate canonical proxy via `macro_place.objective.compute_proxy_cost`
  on the legalized result.

## Kill gate
**Spike fails if canonical ibm01 proxy > 0.898 after AdamW + legalize.**

0.898 = cascade's cached init 0.85527 × 1.05 (the 5 %-of-cascade target
in TODO.md §C1). If the spike clears, commit the remaining 4 days to
the full TILOS-RUDY + top-K differentiable proxy and run on all 17 IBM.
If it fails, the differentiable approach has the same structural
mismatch as DP and PATH C falls back to C2 (ILP detailed legalization,
1–2 days, ~0.3–1 % lift) followed by C3 (Newton-CG saddle, 2–3 days).

## Generalization check
Spike is ibm01-only. If it passes, full C1 build adds: TILOS-RUDY,
top-K density (matching PlacementCost weights), and a per-bench
evaluation on the 4 fast benchmarks before committing to --all and
NG45.

## Outcome (FALSIFIED 2026-05-12)

**Spike fails the gate; falsifies the spike's variant of C1.**

### Calibration (passed)
`results/calibrate_ibm01.json` — 16 placements: cascade-cached + 1 % / 5 %
perturbed + uniform-random.

- Cascade-cached: smooth 0.86318 vs canonical 0.84528 → +2.12 % (gate <5 %).
- Spearman ρ = 0.929 across 16 placements (gate >0.7).
- Note: smooth UNDER-estimates canonical by 10-18 % when overlaps are
  present. Smooth grid-density and RUDY don't punish discrete overlaps
  the way canonical top-K + TILOS does. Calibration verdict alone is
  necessary but not sufficient — rank correlation across placements
  doesn't imply that the smooth gradient at canonical's optimum points
  toward canonical's optimum.

### Descent iter 1 (raw failure)
- Run A from cascade init (canon 0.84528, 0 overlaps): post-leg **5.267**.
- Run B from SDF init: post-leg **9.607**.
- Bugs: `overlap_penalty` on un-clamped positions read 0 when positions
  escaped canvas; λ_start=0 meant no penalty at step 0; lr=2-5 too high.

### Descent iter 2 (debug-clean failure)
Fixed: clamp inside `overlap_penalty`; added `out_of_canvas_penalty`;
λ_start=100; lr=0.5/1.0; best-tracker.

- Run A from cascade init: every descent step produced canon 1.30-1.54
  with 46-70 overlaps. Smooth proxy stayed reasonable (~1.0-1.1) but
  canonical never recovered. Post-leg **1.534** = +81 % vs cascade input.
- Run B from SDF init: smooth ramped 1.0 → 1.6, canon 2.0-2.4, post-leg
  **2.284** = +91 % vs SDF baseline.
- The script's "pass" was a bug: best-tracker required ovl=0 to update,
  so it stayed at the input cascade placement. Honest descent
  contribution: **negative on both inits**.

### Why
Two factors combined:

1. **Smooth gradient at cascade's canonical optimum points away from
   canonical's optimum.** Smooth(cascade) = 0.863 with non-zero gradient
   even though canonical(cascade) = 0.845 is a verified local minimum.
   First AdamW step (any lr > 1e-3) blows the basin and the overlap
   penalty can't recover it because smooth thinks the new position is
   better.
2. **Same surrogate-mismatch PATH B already falsified.** DPO-style
   primitives (LSE-HPWL + Gaussian density + DP-RUDY) are structurally
   misaligned with canonical (top-K density + TILOS-RUDY). DREAMPlace's
   K=25 sweep (PATH B) confirmed this gap was unbridgeable by tuning.

The full C1 build (TILOS-RUDY + top-K-aware differentiable density +
matched weights) MIGHT close the gap, but the engineering cost is 4-5
days and the spike has produced no positive signal to justify it.

### Decision
C1 (as scoped in TODO.md) is falsified. PATH C falls through to C2
(ILP detailed legalization on cascade output) per TODO.md priority
order.

### What's reusable
- `code/diff_proxy.py`: clean autograd-compatible smooth proxy. Useful
  if C3 (Newton-CG saddle) wants smooth-proxy curvature outside the
  E74 `SmoothProxy` already in use.
- `code/calibrate_ibm01.py`: proxy/canonical correlation probe. Reusable
  pattern for any future smooth proxy.
- `experiments/E76_dreamplace_integration/code/macro_legalizer.py` is
  also handy as a generic snap-to-feasible — confirmed working here.

## Pointers
- Code: `code/diff_proxy.py`, `code/spike_ibm01.py`
- Results: `results/spike_ibm01.json`, `results/logs/`
- Cached input: `experiments/E84_cascading_saddle/results/cascade_ibm01.pt`
- Reused legalizer: `experiments/E76_dreamplace_integration/code/macro_legalizer.py`
- DPO primitives: `writeup/archive/submissions/dpo/ablation_v2_steps.py`
- TODO.md §PATH C, C1 (lines 95–136)
