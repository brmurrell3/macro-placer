---
id: E53
name: multiseed_hybrid
status: marginal
parent: E48, E52
created: 2026-05-01
decided: 2026-05-02
champion_at_time: 1.0990 (E12; E48 hybrid 1.08151 strongest verified candidate after --all lift, ADR-011 *Proposed*)
fast_outcome: **0.91128 (--fast); -0.97 % vs E48 fast 0.92024; -1.14 % vs E41 fast 0.92178; -2.39 % vs E25 fast 0.9336; -3.32 % vs E12 fast 0.9426.** Per-bench: ibm01 0.8923 (E25), ibm04 0.9839 (E41 seed=1 wins, beat seed=42 by 1.79%), ibm09 0.8390 (E41 seed=42), ibm13 0.92993 (E41 seed=1 wins, beat seed=42 by 1.93%). Wall 6.8 hr. Multi-seed mechanism captures real wins where DPO seed-luck differs per bench. THIS IS THE SECOND BREAKTHROUGH TODAY.
all_outcome: **1.08128 (--all); -0.021 % vs E48 1.08151 (essentially tied within noise); -0.32 % vs E41 1.0848; -1.61 % vs E12 1.0990; -3.22 % vs leaderboard 1.1172.** Wall 137418 s aggregate / ~10.3 hr wall-clock under --jobs 4. Per-bench: E25 wins same 5/17 as E48 (ibm01, 06, 07, 17, 18); E41 (whichever seed was luckier) wins same 12/17. Adding seed=1 helped on a few benches (ibm02, ibm03, ibm15 by sub-noise margins; ibm16 worse on s1) — net cancellation. Both seeds tied on most benches.
ng45_outcome: **0.6938 (--ng45); +0.23 % vs E48 NG45 0.6922.** Per-design: ariane133 0.68778 (vs E48 0.6861, +0.24 %); ariane136 0.67321 (vs E48 0.6685, +0.71 %); mempool_tile 0.73692 (vs E48 0.7375, tied); nvdla 0.67721 (vs E48 0.6767, +0.08 %). NOT a regression like E54 (+1.45 % NG45 with ariane133 +5.14 %); just run-to-run noise where this run's lanes happened to land slightly worse than E48's. Multi-seed lane ON IBM is essentially ariane133-safe; seed-noise is contained within ~0.4 %. Best-of-{E48, E53m} NG45 = 0.69206 (only -0.02 % vs E48 — trivial).

outcome: marginal — adding E41 seed=1 to the hybrid gives essentially no incremental lift over E48 on --all (0.021 %, within DPO seed-noise). The --fast big lifts (ibm04 -1.79 %, ibm13 -1.93 %) were sample-size outliers; in --all both seeds tied on those benches. **Multi-seed within DPO is a dead-end for breakthrough.** The productive direction for further lifts is NEW MOVE TYPES (E54 congestion-destroy compositional best-of: 0.91708 fast, -0.31 % vs E48) or NEW INIT CLASSES (RePlAce/learned/etc.), not seed perturbation.
champion_delta: -0.021 % vs E48 (sub-noise marginal)

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
