---
id: E53
name: multiseed_hybrid
status: in_progress
parent: E48, E52
created: 2026-05-01
decided: null
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; E48 hybrid --fast 0.9202 — first lift above E41 saturation)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E53: multiseed_hybrid

## Hypothesis
E48 (E25 + E41 seed=42 hybrid) realized -0.17% on --fast vs theoretical
-0.57% bound. The gap came from DPO seed-noise on ibm04 (this run E41
ibm04 = 0.9988 vs verified 0.9845). E52 (E41 seed=1) measured the
seed-noise directly — different seeds win different benches:

  bench    seed=42    seed=1    winner
  ibm01    0.9120     0.9109    seed=1 (-0.12%, tied)
  ibm04    0.9845     0.9923    seed=42 (+0.79%)
  ibm09    0.8412     0.8403    seed=1 (-0.11%, tied)
  ibm13    0.9492     0.9441    seed=1 (-0.54%)

So a 3-way hybrid {E25, E41(seed=42), E41(seed=1)} captures BOTH basin
choice (E25 vs E41) AND seed-luck per bench. Projected --fast:

  best-of per bench:
    ibm01:  0.8910 (E25)
    ibm04:  0.9845 (E41 seed=42)
    ibm09:  0.8403 (E41 seed=1)
    ibm13:  0.9441 (E41 seed=1)
  avg:    0.9150 = -0.74% vs E41 fast 0.92178

That's another 0.57% beyond E48 hybrid's empirical -0.17% on --fast.

## Method
Per-benchmark pipeline:

1. Run **E25 (CDLNSSAPlacer)**: SDF init + CD + LNS + SA-v2.
2. Run **E41 seed=42 (CDLNSSADPOKJointPlacer)**: DPO + CD + LNS + SA + K-joint.
3. Run **E41 seed=1**: same E41 pipeline but with all seeds (DPO seed,
   SA seed, K-joint seed, LNS seed) set to 1.
4. Compute compute_proxy_cost on all three outputs.
5. Return the lowest-cost placement (with overlap == 0 verified).

All hyperparameters from each pipeline preserved unchanged. No
per-benchmark tuning — the per-bench winner is determined by PROXY VALUE.

Wall budget: per-bench wall = E25 + 2 × E41 walls = ~80-110 min/bench
sequential. Total --all wall ~22-30 hr serial. **Significantly exceeds
17-hr competition envelope.** Mitigation: run all 3 pipelines in
PARALLEL (--jobs 3 per bench), reducing per-bench wall to max(E25,
E41) ≈ 50-60 min. Total --all wall ~14-17 hr, just within envelope.

## Kill gate
- **Regression on --fast:** if avg --fast > E48 fast 0.9202 + 0.5 % →
  kill (something broke).
- **No realization of bound:** if avg --fast > E48 fast 0.9202 - 0.3 %
  → marginal (multi-seed doesn't help in practice).

## Generalization check
- Run --ng45 if --fast lifts ≥ 0.3 % over E48 fast 0.9202.
- Run --all if --ng45 doesn't regress.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_multiseed_hybrid.py`.
- Parents: E48 (E25 + E41 seed=42), E52 (E41 seed=1).
- Discussion: bound math in commit message + E52 manifest.
