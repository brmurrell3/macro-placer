# ADR-005: Retain SDF init across every algorithmic generation

**Status:** Accepted
**Date:** 2026-04-15 (held through every regeneration since)
**Deciders:** project owner

## Context

The DPO ablation study made the dominant failure mode of macro placement
explicit: random init produces 4.9415 avg proxy on `--all`, +247 % vs
DPO with SDF init. The continuation cannot recover from chaos.

The SP1 stage of the 22-experiment overnight sweep tested every
plausible alternative initialization and they all lost badly to SDF
(1.5002): spectral_topology 1.78 (ignores macro sizes); hMETIS
partitioning 1.91 (shelf-packing creates dense regions);
greedy_construction 1.75 (clustering creates dense regions);
boundary_attraction killed (penalty inert/harmful at every lambda);
replace_topology killed (extract_assignment erases congestion advantage).
Only congestion_aware_extraction finished, at 1.4930 — within noise of
the SDF baseline.

Subsequent attempts to escape SDF's basin via diversification all
failed:

- E11 diverse priors (SDF + Will + greedy + random) on `--all`: 1.3839
  vs DPO 1.3834, flat (+0.04 %). ibm02 and ibm12 got worse with
  alternative priors — basin-locked benchmarks are locked by topology,
  not init.
- 0/190 cluster swaps on ibm01 improved proxy. Coarse arrangement is
  not the bottleneck; SDF's force-directed spreading already finds it.
- Multi-init via SDF jitter (`analysis/multi_init_probe/probe.py`, 8
  jittered inits on ibm09, sigma = 0.02-0.10 of canvas): unjittered
  control beat every jittered variant. SDF -> CD is contractive;
  perturbing strictly hurts.

## Decision

SDF analytical spreading is the canonical initialization for every
generation: polyhedra, DPO, CDOnly, CDAdaptive. The implementation
lives at `macro_place/sdf_init.py` (post-reorg) and is shared
across placers.

## Consequences

Positive: eliminates a class of failure modes (random-init catastrophe,
+247 %). The DPO ablation "no SDF" stays the dominant ablation in the
writeup. The contractive SDF -> CD pipeline is the reason every CD run
on every IBM benchmark converges without diverging or stalling.

Negative: limits exploration. If a fundamentally different basin exists,
SDF will not find it — and every escape mechanism we tested (jitter,
diverse priors, batched seeds with sigma = 0.04 perturbation) collapses
back to the same basin. E5 (B=64 batched seeds on GPU) collapsed all 64
seeds to one basin; best-of-N within a single basin does not help.
ibm02's basin lock (every DPO seed converges to byte-identical 1.6888)
was eventually broken by CD on the real proxy, not by re-initialization.

Alternatives ruled out: every init tested in SP1 (spectral, hMETIS,
greedy, boundary, replace_topology); E11 diverse priors; SDF jitter;
batched seeds.

## Evidence

- `writeup/evidence.md` §3.1 (random init +247 %, ablation table)
- `writeup/evidence.md` §4.1 (SP1 stage falsifications: spectral 1.78,
  hMETIS 1.91, greedy 1.75)
- `writeup/evidence.md` §6 (cluster swaps: 0/190 improved proxy on
  ibm01)
- `writeup/evidence.md` §9.5 (multi-init CD via SDF jitter,
  unjittered control beats every variant)
- `writeup/evidence.md` §9.3 (E11 diverse priors flat, ibm02/ibm12
  worse)
- `writeup/evidence.md` §9.1 (E5 batched seeds, all collapse to same
  basin)
