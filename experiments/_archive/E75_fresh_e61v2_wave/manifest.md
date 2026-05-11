---
id: E75
name: fresh_e61v2_wave
status: in_progress
parent: E61 V2 (canonical), E48 (champion), E69 (SP-search marginal), E72 (cached-pair crossover failed)
created: 2026-05-04
decided: null
champion_at_time: 1.08151 (E48 hybrid)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E75: fresh_e61v2_wave — replicate third basin via fresh same-session co-runs

## Hypothesis

E72 (cached-pair crossover) confirmed: the "third basin" E61 V2 historical
found is fragile to ~0.07% placement drift between cached E25/E41 and
fresh co-runs. To reproduce its lift (-0.55% on ibm12 historical), we
need fresh same-session E25 + E41 runs INSIDE the placer (which is what
E61 V2 itself does).

Per E48 hybrid data, **7 of 17 IBM benches have E25/E41 gap < 1%**
(tied basins): ibm08, ibm12, ibm14, ibm15, ibm16, ibm17, ibm18. E61 V2
historical lifted on ibm12/14/15. If the lift extends to all 7 tied
benches, aggregate --all could shift by ~0.2-0.4%.

Layered with iter-E69 v2 polish on top of each E61 V2 output, expect
additional ~0.05-0.1% per bench.

## Method

For each tied-basin bench (priority: ibm12, ibm14, ibm15, ibm16, ibm17,
ibm18):

  1. Run E61 V2 (CDLNSGACrossoverPlacer) fresh — runs E25 + E41 + crossover
     + polish (CD + LNS + SA-v2) internally. Wall: ~3-7 hr/bench.
  2. Save output placement.
  3. Run iter-E69 v2 (axis-preserving SP-swap, 3-5 rounds, 200 attempts/
     round, polish-per-round 180s) on top.
  4. Final placement = best of {E48 cached, E61 V2 fresh output, E61 V2
     + iter-E69}.

Run benches in parallel under `--jobs 4`. ETA ~10-20 hr aggregate
across 6 tied benches.

## Kill gate

- Per bench: if E61 V2 fresh output is NOT below E48 by ≥ 0.1% → mark
  bench-specific failure; don't compose.
- Aggregate: if 0/6 benches lift ≥ 0.3% over E48 → falsify the wave.
- Champion gate: if aggregate --all (best-of per bench) lifts ≥ 0.30%
  over E48 1.08151 → graduate to ADR-012.

## Generalization check

NG45 ariane133 — must not regress. E48 ng45=0.6922 reference; new entry
must be ≤ 0.6943.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/run_fresh_e61v2.py`, `code/run_e75_iter_e69.py`.
- E61 V2 placer at `experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`.
- iter-E69 v2 at `experiments/E72_iter_block_pair/code/run_iter_e69_only.py`.
