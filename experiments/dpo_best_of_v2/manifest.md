---
id: dpo_best_of_v2
name: dpo_best_of_v2
status: superseded
parent: polyhedra_navigation
created: 2026-04-23
decided: 2026-04-27
champion_at_time: 1.4867
outcome: 1.3834
champion_delta: -0.1033
graduated_to: null
superseded_by: E9
---

# dpo_best_of_v2 (DPO era umbrella)

## Hypothesis
The polyhedra Phase-2 ceiling at 1.4867 is structural: local
constraint-flips can't reach better topologies. A *differentiable
penalty optimization* (DPO) — relaxing the proxy's hard constraints into
a smooth loss with progressive penalty schedules — should let gradient
methods cross polyhedra boundaries that polyhedra navigation cannot.

## Method
DPO with multi-phase penalty continuation, congestion + density gradient
terms, and SDF init. Best variant: best-of-v2 = best-of(SDF, DPO with
v2-step counts) per benchmark, captured in `submissions/dpo/best_of_v2_placer.py`.

Versions covered by this umbrella:
- v1 (`submissions/dpo/placer.py`, seed 42) — first to beat RePlAce, 1.4255.
- v2 / v2_steps_restore — more gradient steps, 1.4107.
- v3 final — refined config, 1.4246.
- best_of_sdf_dpo — best-of(SDF, DPO, seed) per bench, 1.4145.
- best_of_v2 — best-of(SDF, DPO with v2-steps), **1.3834 (champion)**.
- ablations: no-cong, no-dens, phase1-only, random-init, cong-weight
  sweep cw\_{050…150}, multi-seed best_of_v2_seed{43–46}.

## Kill gate
Cross polyhedra ceiling (1.4867). Continued ablations were tracked with a
≥ 1 % rule per variant; basin-locked benchmarks (ibm02, ibm12) were the
cross-cutting veto.

## Generalization check
Multi-seed stability: 5-seed range 1 % (1.3790–1.3927); ibm02/ibm12 zero
variance (SDF always wins on those — SDF basin is the basin).

## Outcome (filled when decided)
**Superseded by E9 CDAdaptive.** Best avg `--all`: **1.3834** (+5.1 % vs
RePlAce). The DPO line proved that:

1. Gradient methods CAN cross polyhedra boundaries — DPO changes
   3 500–15 500 pairwise L/R/A/B relations from SDF to final output (the
   "barrier crossing" finding, evidence.md §8).
2. But within-DPO refinements cap at 1–2 % (E5 batched seeds, E10 cong-only
   refinement, E11 diverse priors all falsified).
3. ibm02 and ibm12 are byte-identical across 4 seeds — the basin is
   structural to DPO's gradient, not random.
4. 4 / 17 benchmarks WORSEN under DPO vs SDF (ibm01, ibm02, ibm06, ibm12).
   The differentiable proxy is an imperfect model that actively misleads
   on these benchmarks.

The basin lock on ibm02 / ibm12 was the smoking gun for the CD pivot:
whatever was wrong with DPO required a fundamentally different optimizer,
not another within-DPO knob. CDAdaptive (E9) dropped ibm02 from 1.6888 to
1.1534 — a basin-changing 32 % improvement.

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16`). Originally at
  `submissions/dpo/placer.py`, `best_of_v2_placer.py`,
  `congestion_refine_placer.py`, `diverse_priors_placer.py`,
  ablation files, and seed variants.
- Per-bench champion table: `writeup/evidence.md` §2.3.
- Ablation study: `writeup/evidence.md` §3.
- DPO version chronology: `writeup/evidence.md` §10.
- DPO worsening 4 / 17 vs SDF: `writeup/evidence.md` §3.4.
- Discussion: `docs/experiment_index.md` "DPO line (all superseded by CD)"
  rows.
- Predecessor: `experiments/polyhedra_navigation/manifest.md`.
- Successor: `experiments/E9_plateau_detection/manifest.md`.
