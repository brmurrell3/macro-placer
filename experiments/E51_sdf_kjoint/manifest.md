---
id: E51
name: sdf_kjoint
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

# E51: sdf_kjoint

## Hypothesis
**Basin attribution probe.** E41 = DPO + CD + LNS + SA-v2 + K-joint
beat E25 by -0.97 % and beat E18 by -0.46 % on `--all`. The composition
of "DPO basin" + "K-joint mechanism" was the win, but we don't know
which component contributes more.

E39 already partially answered this: E39 = SDF + CD + LNS + SA-v2 +
K-joint = 0.93070 fast (-0.31 % vs E25), but PARTIAL on `--all` (only
6/17 valid before the K-joint bug crashed it). The post-fix `--all`
rerun was never queued (superseded by E41).

E51 = SDF + CD + LNS + SA-v2 + K-joint with the K-joint bug fix —
exactly the experiment E39 should have been if not for the crash.

If E51 ≈ E25 (no lift over the SDF-basin pipeline): K-joint is
load-bearing only when paired with DPO basin (which opens the K-tuple
structure K-joint exploits). E41's lift = DPO basin contribution +
K-joint × DPO-basin-multiplier.

If E51 < E25 by ≥ 0.3 %: K-joint helps SDF basin too. E41's lift =
DPO basin + K-joint, additively. Then per-bench: max(E25, E51, E41)
might hybridize even better than E48.

If E51 > E25: SDF-init pipeline doesn't tolerate K-joint (the K-joint
phase commits net-worse moves on SDF basin). Possible if SDF places
macros in a configuration where the K-joint mechanism's joint moves
cluster macros differently.

This experiment fills in the missing data point in the
{E25, E18, E41} comparison: what does {SDF init, K-joint} give us?

## Method
Same pipeline as E41 but step 1 (DPO init) replaced by SDF init.
All other phases unchanged: project_overlaps → CD plateau → grid-bin
LNS → SA-v2 polish → K=3 K-joint → validate.

Implementation: monkey-patch `_best_of_v2_init` in E41's module to
return SDF-only init, instantiate CDLNSSADPOKJointPlacer, run.

K=3, top_N=5, kjoint_budget_s=600 — same as E41. All hyperparameters
global; no per-benchmark tuning.

## Kill gate
- Regression on --fast: avg --fast > E25 fast 0.9336 + 0.5 % (i.e.,
  > 0.9383) → kill, status=falsified. K-joint hurts SDF basin.

## Generalization check
- If --fast lifts ≥ 0.3 % vs E25 fast 0.9336, run --ng45.
- If --ng45 lifts ≥ E25 ng45 (need E25 ng45 number — wasn't in the
  experiment log; estimate from E12 ng45 0.7037 + E25 vs E12 lift bound
  ≈ 0.7000), queue --all.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_sdf_kjoint.py`.
- Parents: E41 (DPO + K-joint pipeline), E25 (SDF-init pipeline).
- Related: E39 (the original SDF + K-joint experiment, partial --all).
- Discussion: completes the {init class, K-joint} comparison matrix.
