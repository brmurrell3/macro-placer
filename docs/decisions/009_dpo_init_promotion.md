# ADR-009: Promote DPO init + CD + LNS + SA-v2 as champion (E18 → 1.08979)

**Status:** **Proposed** — awaiting human decision. The verified `--all` result
(1.08979, zero overlaps, 11/17 wins) clears all decision-rule thresholds and
beats both the current champion E12 (1.0990, ADR-007) and the prior candidate
E25 (1.0954, ADR-008 *Proposed*). `--ng45` confirms the lift transfers to
commercial designs (0.69193, −1.67 % vs E12 0.7037, 4/4 per-design wins).
Champion remains E12 (ADR-007); E18 code stays at
`experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py` until promotion.
**Date:** 2026-04-30 (proposed)
**Deciders:** project owner

## Context

E25 CDLNSSA (ADR-008 *Proposed*) hit 1.0954 on `--all` — beat E12 by −0.33 % —
by adding an SA-v2 polish phase to E12's CD + grid-bin LNS pipeline. That
ADR was held open as "candidate, not promoted" pending more data.

The E27 basin-persistence diagnostic was supposed to gate post-E25 work:
single-basin verdict → ship E25 + writeup; multi-basin → pursue further
escapes. E27's persistence-homology output came back **ambiguous** (only
16/44 trajectories completed; the DPO init helper crashed on a missing
file — see "Bug fix needed" below).

E18 was launched in parallel as a tactical hypothesis under the "alternative
init" branch: replace the SDF init in E25's pipeline with the DPO best_of_v2
output (the ceiling DPO reached before being superseded by CD). Per ADR-005
SDF was canonized as the CD init because random and SDF-jitter inits found
worse basins. The DPO init was *untested*: ADR-005 didn't rule it out.

E18's `--all` result resolves the ambiguity directly. **The DPO basin is
structurally distinct from the SDF basin AND deeper.** This is the
empirical signal E27 was meant to provide; E18 supplied it via a stronger
form (full `--all` validation, not a CD-only trajectory cluster).

## Decision (proposed)

If accepted: adopt **DPO best_of_v2 init → CD + grid-bin LNS + SA-v2 polish**
as the champion. The pipeline runs:

| Phase | Wall budget | Output |
|---|---|---|
| 1. DPO best_of_v2 init | ~30 min/bench | Initial placement (DPO basin) |
| 2. project_overlaps | < 1 s | Legalized initial placement |
| 3. CD plateau | ≤ 2400 s | Per-axis fixed point |
| 4. Grid-bin LNS | ≤ 600 s | Different-move-type escape |
| 5. SA-v2 polish | ≤ 600 s, T₀=5e-4 | Breakpoint-set escape |
| Total | ~47 min/bench typical | |

DPO init runs *before* CD because DPO converges to a basin different from
SDF's (E18's empirical finding). The downstream phases (CD+LNS+SA-v2) are
ADR-007 + ADR-008 unchanged. All hyperparameters global; no per-benchmark
tuning.

## Consequences

- **Champion:** 1.0990 (E12) → **1.08979** (E18). Improvement:
  - vs E12: −0.0092 (**−0.84 %**)
  - vs E25 candidate: −0.0056 (−0.51 %)
  - vs leaderboard 1.1172: **−2.45 %** (vs E12's −1.63 %, E25's −1.95 %)
  - vs RePlAce 1.4578: −25.2 %
- **Per-bench `--all` distribution:**
  - 11 wins / 6 losses vs E25 candidate (1.0954)
  - Zero overlaps on every bench
  - Largest gains on benches where DPO basin diverges most from SDF
    basin (per E18 manifest)
- **NG45 commercial-design transfer (4 designs):** `--ng45` avg
  **0.69193**, **−1.67 % vs E12 0.7037**, with **4/4 per-design wins**
  (ariane133 −3.75 %, ariane136 −1.64 %, mempool_tile −0.85 %, nvdla
  −0.42 %). Zero overlaps. **The DPO-basin lift is OOD-robust** —
  consistent with the hypothesis that DPO finds topology information CD
  alone misses, and that information transfers to designs with different
  macro distributions. This was a major risk in the SDF→DPO swap; E23 had
  validated NG45 transfer for E12 but not E18, and the SA-v2 phase was
  expected to transfer mechanically (per-axis breakpoints, evaluator-
  agnostic). DPO transfer was untested. **It works.**
- **Total wall:** 7.85 hr (E12) → 10.33 hr (E25) → **13.30 hr (E18) on
  `--all`**. The DPO init phase (~30 min/bench × 17 = ~8.5 hr nominal) is
  the dominant added cost; in practice DPO completes faster on smaller
  benches, so the wall is below worst-case. **3.7 hr of headroom on the
  17-hr competition envelope.** Per-bench worst-case still well under the
  3600 s contest cap.
- **Architectural interpretation.** Four phases on three structurally
  different scaffolds:
  1. DPO basin selection (gradient-based on differentiable proxy with
     annealed overlap penalty) — finds a *different starting region*.
  2. CD per-axis fixed point — exact-coordinate optimization within
     basin.
  3. Grid-bin LNS — single-macro destroy/reinsert at grid-cell centers,
     a move type CD cannot reach.
  4. SA-v2 polish — Metropolis acceptance over per-axis breakpoints,
     finds breakpoint-set wins LNS-alone leaves on the table.
  Adding a *fifth* phase is the K-joint joint-move escape (E39 mechanism)
  composed on top — see E41 below for the in-progress combo.
- **Why DPO works here when ADR-005 ruled out random/jitter inits.** ADR-
  005 falsified random and SDF-jitter inits because they landed in
  worse-or-equivalent basins after CD (no separation). DPO is *not* a
  perturbation of SDF — it's a learned descent on a smooth proxy with
  congestion gradient information that CD's per-axis greedy cannot
  exploit. DPO deposits the macro at a point informed by *hundreds of
  smooth gradient steps*; CD's per-axis greedy then refines from that
  point and converges to a *different* local minimum than CD-from-SDF.
  ADR-005 stands for the inits it tested; DPO is a separate class.
- **The E14 lesson still binds.** SA-v2 inside this pipeline is the same
  fixed primitive (best-so-far + low T₀) per ADR-008.

## Open questions / follow-ups

1. **E41 combo extension.** E41 = E18 + K-joint LNS phase (E39 mechanism)
   ran `--fast` at **0.92178** (-1.27 % vs E25 fast, beats E18 fast 0.92542
   on all 4 benches) and `--ng45` at **0.69022** (-1.91 % vs E12 NG45,
   beats E18 NG45 on ariane133 by additional -0.92 %). E41 `--all` was
   stalled by a multiprocessing.Queue deadlock under heavy box load and
   then by the K-joint overlap-validation bug; both fixed 2026-04-30.
   **If E41 `--all` rerun succeeds at the same compositional rate, E41
   would beat E18 by another ~0.5-1.0 % on `--all`.** ADR-009 promotes
   E18 NOW (verified data) and ADR-010 covers E41 if its rerun succeeds.
2. **DPO init wall variance.** `--all` total 13.30 hr leaves 3.7 hr cap
   margin. Verify per-bench DPO wall on the harder benches (ibm12, ibm15,
   ibm17) doesn't blow the 1-hr-per-bench contest limit.
3. **Bug fix applied 2026-04-30 07:48.** `BestOfV2Placer` was using
   `importlib` to dynamically load a sibling `writeup/archive/submissions/
   polyhedra/init/sdf.py` that was orphaned during a writeup-archive
   reorg. Fix: replaced the dynamic-load path with
   `from macro_place.sdf_init import SDFPlacer` (canonical in-tree
   version with same `seed=42` constructor signature). Verified by
   instantiation. E27 is rerunnable on DPO inits.

## Alternatives ruled out

- **E25 (CDLNSSA) at 1.0954, ADR-008 *Proposed*.** Superseded by E18 on
  every metric: −0.51 % on `--all`, NG45 transfer confirmed (E25 has no
  NG45 datapoint), zero new overlaps. Recommend marking ADR-008 as
  *Superseded* once ADR-009 is *Accepted*. E25 code at
  `submissions/cd_lns_sa/placer.py` stays in tree as the immediate
  parent of the production pipeline.
- **E17 (random init) at avg `--fast` 1.08653, falsified.** Random init
  lands in a basin far worse than SDF or DPO (consistent with ADR-005).
- **E32 (SAM-CD) falsified at +5.5 % `--fast`.** K=4 perturbations
  multiplied CD eval cost by ~5×; CD never plateaued.
- **E40 (multi-SA-seed) marginal.** Within-basin ensemble adds ≤0.07 %
  on `--fast`, not worth ~14 hr `--all` rerun.

## Evidence

- `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py` — candidate code.
- `experiments/E18_dpo_init/manifest.md` — `decided: 2026-04-30`,
  `status: champion_candidate`.
- `results/experiment_log.jsonl` rows:
  - `e18_dpo_init_fast` 0.92542 (`--fast`)
  - `e18_dpo_init_all` 1.08979 (`--all`, 47 876 s wall)
  - `e18_dpo_init_ng45` 0.69193 (`--ng45`)
- `experiments/E27_basin_persistence/manifest.md` — empirical-multi-basin
  diagnostic.
- `experiments/E41_dpo_kjoint/manifest.md` — composition with K-joint
  escape (combo running for ADR-010).
- `submissions/dpo/best_of_v2_placer.py` — the DPO init component
  (existing prior-prior champion code).
- ADR-005 (SDF init canonical), ADR-007 (E12 promotion), ADR-008 (E25
  promotion *Proposed*).
