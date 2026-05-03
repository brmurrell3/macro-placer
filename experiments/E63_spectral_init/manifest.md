---
id: E63
name: spectral_init
status: blocked_on_implementation
parent: E41 (pipeline) + spectral / quadratic placement (init)
created: 2026-05-02
decided: null
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **implementation V2 in progress 2026-05-02 23:55** — Hungarian-assignment legalizer (added by Claude after V1 push-apart approach stalled) computes spectral assignment in 0.02 s. **Issue**: ibm01 has max_macro=7.01×3.84 on canvas ~30×30, fitting only 28 max-size slots versus 246 hard-movable macros. Falling back to a finer slot grid (1.15× n_movable slots) violates the no-overlap-by-construction guarantee → 9-36 residual overlaps after multi-pass project_overlaps + jitter retry. Hungarian *would* work on lower-density benches (canvas/max_macro_size² ≥ n_macros) but ibm01 fails this. **Remaining work**: row-packing legalizer (sort by spectral y/x, greedily pack rows L→R respecting actual sizes — standard EDA technique) is what's needed for high-density benches. Estimated 30-60 min focused dev. Deferred to next session — E65 NEB cross-section is the more immediate diagnostic that consumes the existing E25/E41 basins.
champion_delta: not yet measurable
graduated_to: null
superseded_by: null
---

# E63: spectral_init — netlist Laplacian eigenvectors as a third basin source

## Hypothesis
The netlist hypergraph's graph Laplacian L = D − A has eigenvectors
encoding the netlist's **global connectivity modes**. The smallest
non-trivial eigenvalues correspond to "low-frequency" global structure
(broad clusters, hub macros), and using them as (x, y) coordinates gives
a placement that respects netlist topology directly.

This is *Gordian-style quadratic placement*, the 1980s spectral
predecessor to gradient-based methods. The point is **not** that
spectral placement beats SDF / DPO standalone (it doesn't — early
experiments hit 1.78 vs SDF's 1.50). The point is the **basin landed by
CD on a spectral init is structurally different** from SDF or DPO basins.

If the spectral basin polishes (via E41 pipeline: CD + LNS + SA + K-joint)
to within ~1.10 average and **wins on at least one IBM-fast bench** AND
verifies safe on NG45 ariane133, it qualifies as a third hybrid lane in
future E65 = best-of-{E25 SDF, E41 DPO, E63 spectral}.

The E62 WillSeed falsification (2026-05-02) showed that NOT every "different
init" produces a useful basin. WillSeed's GPU-legalized state landed CD in
a structurally inferior basin to both SDF and DPO. **Spectral inits are
distinguished from WillSeed by being benchmark-structure-aware**: the
eigenvectors are computed from the netlist itself, so the basin landed
should reflect topology, not arbitrary GPU optimization artifacts.

## Method
Pipeline per benchmark (identical to E41 except step 1 init):

  1. **Spectral init.** Build the weighted graph Laplacian L = D − A
     where A_{ij} = sum over nets containing both macros i and j of
     weight 1/(|net| − 1) (clique expansion with normalization).
     Compute the 3 smallest eigenvalues / eigenvectors via
     `scipy.sparse.linalg.eigsh` (sparse symmetric solver).
     Discard eigenvalue 0 (the constant eigenvector); use eigenvectors
     2 and 3 as (x, y) coordinates. Linearly rescale to canvas bounds
     with margin matching SDF init.
  2. project_overlaps to clear residuals (spectral init has no
     non-overlap constraint built in; expect non-trivial repair work).
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive (≤ 2400 s).
  5. Grid-bin LNS (≤ 600 s, cost-aware destroy).
  6. SA-v2 (≤ 600 s, T₀=5e-4).
  7. K-joint LNS (≤ 600 s, K=3, top_N=5).
  8. Validate, preserve fixed macros.

All hyperparameters from E41 preserved unchanged. Only init differs.

## Kill gate
- **--fast > 0.929** (E41 fast 0.92178 + 0.8 % = same threshold as E62):
  spectral basin polishes to a state worse than the IBM-overfit margin
  → falsified.
- **--fast within 1 % of E48 0.92024 (≤ 0.929):** *plus* at least one
  per-bench win on ibm01/04/09/13 vs E48 → *qualifies for --ng45*. If
  no per-bench wins on fast, falsify (won't help hybrid).
- **--ng45 ariane133 > E48 0.6861 + 1 % (= 0.694):** spectral basin
  fails NG45 transfer → falsified.

## Generalization check
NG45 is the gate. The spectral init is benchmark-structure-aware
(eigenvectors come from the netlist), so generalization should be
better than IBM-overfit mechanisms (E42/E43/E44/E54/E62). But the
NG45 verification is required regardless.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_spectral_kjoint.py`
  (`CDLNSSASpectralKJointPlacer` and `_spectral_init`).
- Pipeline backbone: E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- Discussion: motivated 2026-05-02 by roadmap §4.6.B "global topology
  navigation" reframing, after E62 WillSeed falsification confirmed
  that not every "different init" produces a useful basin. Spectral
  is distinguished by being NETLIST-STRUCTURE-AWARE.
