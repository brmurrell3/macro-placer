---
id: E45
name: multigrid
status: in_progress
parent: E41
created: 2026-04-30
decided: null
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; ADR-010 *Proposed*)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E45: multigrid

## Hypothesis
Every champion in the lineage (CDOnly → CDAdaptive → CDLNSGridBin →
CDLNSSA → CDLNSSADPOInit → CDLNSSADPOKJoint) is *flat* over 200-537
macros. Single- and 2-macro moves can't move *aggregate mass* — that's
why coupled fixed points exist and why E25 ties E12 on ibm11/13/14/15
under five different mechanisms. E41 partially escapes via DPO basin
selection + K=3 K-joint, but it still cannot REORGANIZE chunks of the
floorplan. The 3 bench losses where E41 underperforms E25 (ibm01 +2.46 %,
ibm07 +1.32 %, ibm06 +1.26 %) are exactly cases where the DPO basin
locked into a globally-suboptimal layout that local moves cannot recover.

A V-cycle multigrid scheme rearranges aggregate mass cheaply at the
coarse scale and refines at fine scales. **Standard escape from
local-operator saturation in PDE solvers, materials-science RG, and
multilevel partitioning (METIS / hMETIS) — conspicuously absent from
placement.** Today's K-joint variants (E42 K=4, E43 longer budget, E44
spatial K-tuples) all confirmed E41 K=3 at 600 s is the K-joint
mechanism's saturation floor. The next productive direction is *a
different scale of move type*, not more parameter tuning of the same
local mechanism.

E27 placement-cluster diagnostic on ibm11/13 showed DPO basin is
*spatially distinct* from SDF basin → the right basin region was found
by DPO but not SDF. On ibm14/15 DPO is in the same spatial cluster as
SDF (just at a deeper proxy point). Multigrid attacks BOTH classes:
- Coarse-scale super-macro placement explores spatial-cluster choices
  cheaply (super-macros mean fewer entities → fast CD/LNS).
- Fine-scale refinement (constrained to parent super-macro region) does
  the within-cluster proxy-deepening that E41 already does.
- The combination should find better super-macro layouts (basin choices)
  AND better fine-scale placements (basin depth) than any single-scale
  pipeline.

## Method
Hierarchical (V-cycle) placement on top of E41's primitives:

1. **Coarse-scale formulation (per benchmark, build-time):**
   - Build macro-macro graph: edges between macros sharing ≥ 1 net,
     weight = sum of `1 / max(1, net_size - 1)` over shared nets
     (consistent with the K-joint adjacency score).
   - Partition into K = round(sqrt(num_hard_macros)) super-macros via
     pymetis (METIS graph partitioner; clique-expansion of the
     hypergraph).
   - For each super-macro: bounding box = sum of constituent areas
     packed at the average aspect ratio of the canvas.
   - Quotient netlist: each net's pin set is reduced from macros to
     super-macros (drop self-edges within a super-macro).

2. **Coarse-scale placement (~minutes):**
   - Build a synthetic Benchmark wrapper for the coarse problem.
   - Place super-macros via E12 CD plateau + grid-bin LNS.
   - Output: super-macro centers and bounding-box dimensions.

3. **Uncoarsening (one level):**
   - For each super-macro, the placement region is `(super_macro_center
     ± half_super_bbox)`.
   - Build sub-Benchmark for each super-macro: hard macros are its
     constituents; canvas is the super-macro's region; soft macros and
     fixed macros from the original benchmark that fall inside the
     region (pinned at original positions if fixed).
   - Place each sub-benchmark via E12 CD + grid-bin LNS.
   - Aggregate per-super-macro placements back to global coordinates.

4. **Post-smoother (full E41 pipeline):**
   - The aggregated placement is the multigrid output.
   - Run as INIT replacement in the E41 pipeline (DPO replaced by
     multigrid output): project_overlaps → CD plateau → grid-bin LNS
     → SA-v2 polish → K-joint LNS K=3 → validate.

All hyperparameters global. No per-benchmark tuning. K (number of
super-macros) is the round-sqrt of macro count — a global rule.

## Kill gate
- **Regression on --fast:** if avg --fast > E41 fast 0.92178 + 0.5 %
  (i.e., > 0.9264) → kill, status=falsified.
- **No incremental lift over E41:** if avg --fast within ±0.05 % of
  E41 fast 0.92178, mark marginal — multigrid doesn't beat single-scale
  on the small benches. Larger NG45 designs may still benefit; queue
  --ng45 in that case as a tiebreaker.
- **NG45-only lift:** if --fast ties E41 but --ng45 lifts ≥ 0.5 %, mark
  graduate — the IBM benches are too small to expose hierarchical
  structure but commercial designs benefit.

## Generalization check
- If --fast lifts ≥ 0.3 % vs E41 fast 0.92178, run --ng45.
- NG45 is the *load-bearing* test for multigrid: commercial designs
  (ariane133/136, mempool_tile, nvdla) are MORE hierarchical than IBM
  (datapath/control/memory regions). If multigrid is real, NG45 lift
  ≥ IBM lift.
- If NG45 lift < IBM lift: the multigrid hypothesis is wrong, OR our
  level count/cluster count is wrong (try 3 levels instead of 2; try
  K=20 vs K=40).

Wall budget: ~5 days build (Phase 0-3) + 1 day eval (Phase 4 --fast
→ --ng45). Wall per `--all` benchmark: TBD; expect ~50 min/bench
(coarse phase ≤ 5 min + uncoarsening ~10 min + E41 post-smoother
~50 min, but with much shorter CD plateau on the post-smoother since
the multigrid init is already near-optimal).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint_multigrid.py` (defines
  `CDLNSSADPOKJointMultigridPlacer`).
- Helpers: `code/multigrid_clustering.py`, `code/multigrid_vcycle.py`.
- Parents: E41 (post-smoother pipeline). Reuses E12 CD+LNS at coarse
  and fine scales.
- Dependencies: `pymetis==2023.1.1` (graph partitioner; METIS
  clique-expansion of the macro-net hypergraph).
- Related: E27 basin-cluster diagnostic (motivates the multi-basin
  signature). E29 MIQP joint reinsertion (alternative escape mechanism;
  K=10-20 instead of multi-scale).
- Discussion: `docs/experiment_index.md`; `writeup/evidence.md` §X if
  graduated.
