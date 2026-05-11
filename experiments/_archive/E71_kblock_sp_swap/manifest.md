---
id: E71
name: kblock_sp_swap
status: falsified
parent: E73 (per-pair transplant FALSIFIED — renumbered from E70), E61 V2 (quadrant block)
created: 2026-05-04
decided: 2026-05-04
champion_at_time: 1.08151 (E48 hybrid)
outcome: **FALSIFIED 2026-05-04 07:30 EDT.** V1 (cluster transplant K=10/20/50): 0/200 accepts on ibm01 (100% feasibility failure — even multi-iter project_overlaps can't legalize cluster transplants). V2 (NxN partition crossover, generalizing E61 V2 quadrant): 0/30 accepts on ibm01, 0/30 on ibm12. With extended_legalize (jitter+project iterative loop): feasibility rate jumped from 2% to 50% on ibm01, 10% on ibm12 — but ALL feasible crossovers had higher proxy than starting basin. Deep polish (CD 1200s + LNS 600s + SA 600s) on top-2 candidates on ibm12 produced 1.20530 final (vs start 1.20534, lift -0.003%). **The third basin E61 V2 historical found requires fresh co-runs of E25 and E41**; cached placements drift from historical (~0.07-0.1% per-bench), enough to make crossover output land in different (worse) territory after polish.
champion_delta: 0 (no improvement)
graduated_to: null
superseded_by: null
---

# E71: kblock_sp_swap — cluster-level transplant between basins

## Hypothesis

E70 falsified per-pair transplant: moving 2 macros to another basin's
positions creates 91.5% feasibility failures because the target
positions are occupied by OTHER macros in the current basin.
project_overlaps' 50-iter cap can't cascade-resolve.

E61 V2 spatial-block (canvas-quadrant, K~150 macros) crossover works
(marginal --all -0.07%) — the cluster IS coherent enough to move
together. **The right transplant unit is a SPATIAL CLUSTER of macros,
not isolated pairs.**

E71 generalizes E61 V2 to:
- **Finer-than-quadrant granularity**: K=10-50 macro neighborhoods
  instead of canvas quadrants of K~150.
- **Topology-targeted centers**: cluster centers chosen by
  disagreement-density (where the basins differ most) rather than
  fixed canvas geometry.
- **Multi-cluster accept-and-iterate**: try many clusters in sequence,
  accepting any that improve proxy. (E61 V2 is one-shot quadrant pick.)

If finer-granularity AND topology-targeted clustering finds basins
neither E61 V2 quadrant nor E69 per-pair reach, this is the
breakthrough.

## Method

```
encode SP_E25, SP_E41 → identify disagreement set D
for each macro m: density[m] = |{(i,j) ∈ D : i==m or j==m}|
for trial in range(N):
    if trial < 80%: center = sample macro by density (high-density first)
    else: center = random macro
    K = sample from {10, 20, 50, 100} or fixed
    cluster = K nearest macros to center.position (current state)
    candidate = current.clone()
    for m in cluster: candidate[m] = other_basin[m]
    project_overlaps
    if feasible AND proxy improved: accept (mutate current)
return current
```

After all transplant attempts, optional CD polish.

The key parameter is K (cluster size). E61 V2 uses K~150 (quadrant);
E70 used K=2. Sweet spot likely K=10-50 — large enough to be coherent
in the other basin's topology, small enough that local repair
(project_overlaps) can absorb.

## Kill gate

- ibm01 smoke (200 trials): zero accepts → kill (mechanism doesn't
  work at any tested K).
- --fast > E48 fast 0.92024 + 0.5% → kill regression.
- --fast lift < 0.3% over E48 → marginal.
- --fast lift ≥ 0.3% AND zero NG45 ariane133 regression → graduate-pending.

## Generalization check

Test on 4 fast benches. If aggregate lift ≥ 0.3%, queue --ng45.

## Outcome (decided 2026-05-04)

### V1 — cluster transplant (K=10/20/50)

Smoke on ibm01: 0/200 accepts. 100% feasibility failure: cluster
transplant moves K macros to other_basin's positions, which are
occupied by current macros, creating cascade of overlaps that 50-iter
project_overlaps cap can't resolve. Same failure mode as E70
per-pair transplant (FALSIFIED) — partial swap creates "patch vs
rest" boundary overlaps.

### V2 — NxN partition crossover (generalize E61 V2)

Initial smoke on ibm01: 0/30 accepts (1 feasible / 50 trials = 2%
feasibility). After implementing extended_legalize (jitter+project
iterative loop, 15 passes): feasibility jumped to 50% on ibm01, 10%
on ibm12. **However: all feasible crossovers had higher proxy than
the starting basin** — the polish budget (90s/trial) was too short
to escape the perturbed-state basin.

### V3 — deep polish on top-K crossovers (E72 deep_polish_crossover)

ibm12 (tied basin, E61 V2 historical -0.55% lift target): 4/20
feasible. Top 2 deep-polished with CD 1200s + LNS 600s + SA 600s.
Final: seed 53 → 1.20530, seed 49 → 1.20535. Both WORSE than
starting E41 (1.20534).

iter-E69 on top of best deep-polished: 1.20521 (-0.011% over start).
Marginal lift, far short of E61 V2 historical's 1.1982.

### Why the third basin doesn't appear

Polish phase consistently reached evaluator-proxy 1.20272 on ibm12 —
this would be a -0.21% lift IF compute_proxy_cost agreed. But canonical
proxy of the same placement is 1.20530. The evaluator-vs-canonical
float drift (gotchas.md) destroys ~80% of the polish-internal lift.

E61 V2 historical reached canonical 1.19898 on ibm12. My deep cross
with cached placements doesn't approach this — possibly because the
cached E25/E41 differ from historical by ~0.07-0.1% per-bench (run-to-
run noise in CD-LNS-SA pipeline) and the resulting "third basin
landscape" reachable from the cached pair is just inferior to the one
reachable from historical's pair.

### Verdict

FALSIFIED. The cluster/partition crossover with cached placements
doesn't reach the third basin. To reproduce E61 V2's lift, need fresh
co-runs of E25 and E41 within the same code path (E61 V2 itself), not
cached state. Multi-trial crossover with extended legalize + deep
polish DOES NOT find the third basin from cached pairs.

### Lesson

The third basin is a property of the SPECIFIC E25/E41 pair — fragile to
placement noise. Cached E25/E41 reuse breaks this fragility.

## Pointers

- Code:
  - `code/kblock_placer.py`, `run_kblock.py` (V1 cluster transplant).
  - `code/partition_crossover_placer.py`, `run_partition.py` (V2
    NxN partition).
  - `code/extended_legalize.py` (jitter+project loop legalizer).
- Cached placements at
  `experiments/E69_sequence_pair_search/results/placements/`.
- Parent E61 V2 quadrant code at
  `experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`.
- Related: E72 deep_polish_crossover combines E71 V2 with full
  E25/E41-style polish; also falsified on ibm12 from cached pair.

## Pointers

- Code: `code/kblock_placer.py`, `code/run_kblock.py`.
- Cached placements at
  `experiments/E69_sequence_pair_search/results/placements/`.
- Parent E61 V2 quadrant code at
  `experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`.
- Parent E69 SP encoder at
  `experiments/E69_sequence_pair_search/code/sp_inverter.py`.
