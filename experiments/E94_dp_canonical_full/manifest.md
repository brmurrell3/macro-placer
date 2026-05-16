---
id: E94
name: dp_canonical_full
status: scoped
parent: PATH B B-R3 — full canonical losses on DP substrate
created: 2026-05-12
decided: null
champion_at_time: 1.0612 (cascade uncapped, ibm)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E94: dp_canonical_full — full canonical losses on DP substrate (B-R3)

## Hypothesis

Combine B-R1 (TILOS-RUDY) + B-R2 (top-K density) + WA-WL → smoothed
bbox-HPWL into a single patched DP. The result optimizes the canonical
proxy as a differentiable surface, using DP's Nesterov-with-noise
optimizer + FFT precondition machinery.

This **is** C1 (already falsified on plain AdamW per E88), but with
DP's optimizer in place of AdamW. The bet: DP's optimizer has
features that bridge the smoothing gap AdamW couldn't —
- Nesterov-with-gradient-noise (escapes saddles)
- multi-stage density-weight ramp
- careful learning-rate schedule
- preconditioning

## Method

Build on E92 (B-R1) + E93 (B-R2) once both land:

1. Layer all three loss replacements into PlaceObj.obj_fn.
2. Tune density_weight / congestion_weight ratio to match canonical's
   1.0 / 0.5 / 0.5 weights.
3. Add WA-WL → smoothed bbox-HPWL: replace `weighted_average` in
   global_place_stages with a custom wirelength op that LSE-smooths
   bbox max/min.
4. Calibration check: full canonical-on-DP at fixed cascade placement
   should reproduce canonical proxy within 0.5 %.
5. Run full pipeline: DP → legalize → cascade polish. Compare to E91
   (stock-DP + polish) and cascade alone.

## Kill gate

- Calibration fails (canonical-on-DP ≠ canonical proxy at fixed
  placements within 5 %): kill, smoothing approximation is too lossy.
- E88 falsification (AdamW on similar surrogate) recurs: DP's
  optimizer didn't help. Verdict: gradient methods can't beat
  cascade on canonical via smoothed surrogates, regardless of
  vehicle.

## Implementation notes

This is the heaviest route. Estimated effort: 7–10 days of focused
engineering. Do not start unless E91 + E92 evidence supports.

The "DP-as-vehicle" decision matters less than expected after E88:
since plain AdamW on a similar surrogate already failed, the value of
inheriting DP's optimizer is bounded. The more interesting question
is whether DP's *multi-stage schedule* (HPWL warmup → density ramp →
overflow targeting) can produce better basins than pure gradient
descent. That's testable independently via B-R4 (DP perturber on
cascade init) without rewriting losses.

## Pointers

- DP obj_fn: dreamplace/PlaceObj.py:322-405
- DP NonLinearPlace.py: outer optimization loop with schedule
- E92 (B-R1) RUDY loss: experiments/E92_dp_tilos_rudy/
- E93 (B-R2) top-K density: experiments/E93_dp_topk_density/
- E88 C1 falsification (AdamW): memory/e88_c1_diff_proxy_falsified.md
