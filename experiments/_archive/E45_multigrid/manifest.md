---
id: E45
name: multigrid
status: falsified
parent: E41
created: 2026-04-30
decided: 2026-04-30
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; ADR-010 *Proposed*)
outcome: falsified — block-LNS produces 0 commits on every benchmark tested. ibm01 single-bench (1391s wall): 0 commits, final 0.92003 ≈ E41 ibm01 within DPO seed-noise. ibm10 single-bench (3756s wall): 0 commits in 0.3s wall (converged immediately), final 1.0157 vs E41 ibm10 --all 1.0096 = +0.61% (pure DPO noise; block-LNS was a no-op). Block-LNS hypothesis FALSIFIED: rigid block translations don't escape the post-K-joint local minimum. The bounds/overlap checks reject most candidates and the few that pass don't improve proxy. The SDF/DPO + CD + LNS + SA + K-joint pipeline is locally optimal for block translations.
champion_delta: 0.0 (block-LNS no-op)
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
**Pivot 2026-04-30 22:30:** Phase 2 evidence on ibm10 showed SDF init is
already near-optimal at the coarse scale. Coarse CD on super-macros
moves only 1/21 super-macros under non-overlap constraint, and removing
the constraint just collapses super-macros onto each other. Pure init
replacement doesn't yield a breakthrough — SDF basin already encompasses
the cluster-level optimum on IBM benchmarks.

The new design uses super-macro structure as a **multi-macro MOVE TYPE**
added to the E41 pipeline AFTER the K=3 K-joint phase. Block moves of
30-100 macros simultaneously (vs E41's K=3 macro joint moves) explore a
fundamentally new reachable set:

1. **Phase 1 (unchanged):** Cluster hard macros into K = round(sqrt(N))
   super-macros via pymetis. Output: cluster_ids, super_members.

2. **NEW Phase 2: Super-macro block LNS, runs after E41's K-joint phase.**
   For each super-macro k:
   - Save current placement of all constituents.
   - Compute current centroid of constituents.
   - Sample target centroids on a grid (M=8 per axis = 64 candidates).
   - For each target, propose a rigid translation of all constituents by
     (target − current_centroid). Check:
     - All constituents stay in canvas.
     - No constituent overlaps a non-cluster macro at the proposed position
       (use eps=1e-9 strict separation, like K-joint fix).
   - If valid: apply moves, evaluate proxy via incremental evaluator.
   - Revert all constituents to saved positions.
   - Track the best-improving target across all candidates.
   - Commit the best block move if it improves baseline by > 1e-7.

3. **Multiple passes** until no super-macro produces an improving move
   in a full pass, OR budget exhausted.

K (super-macros), top_M (candidate targets per super-macro =
default 8), block_lns_budget_s (default 600 s), block_lns_seed (=42).
All hyperparameters global; no per-benchmark tuning.

The full pipeline becomes:
   DPO best_of_v2 init -> project_overlaps -> CD -> LNS -> SA-v2 ->
   K=3 K-joint -> **NEW: super-macro block LNS** -> validate

The super-macro block-LNS is the post-smoother that exploits the
multigrid structure — single- and 2- and 3-macro mechanisms can't move
aggregate mass; block-LNS does.

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
