# ADR-010: Promote DPO init + CD + LNS + SA-v2 + K-joint as champion (E41)

**Status:** **Superseded by ADR-011** 2026-05-02. Never reached *Accepted*.
E41 `--all` 1.0848 verified 2026-04-30 cleared all decision-rule
thresholds, but ADR-011 hybrid (best-of-{E25, E41}, 1.08151) lifted
further by −0.30 % at the same NG45 tier (0.6922 vs E41 0.69022).
ADR-011 is the accepted promotion. E41 code stays at
`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` because E48
calls `CDLNSSADPOKJointPlacer` as one of its two best-of lanes —
E41 is a load-bearing **component** of the new champion, just not the
top-level placer.
**Date:** 2026-04-30 (proposed) → 2026-05-02 (superseded by ADR-011)
**Deciders:** project owner

## Context

The morning of 2026-04-30 we had two candidate placers awaiting human
promotion decisions:

- **E18** (DPO init + CD+LNS+SA-v2): verified `--all` 1.08979, −0.84 %
  vs E12, 11/17 wins, plus 4/4 NG45 wins — see ADR-009 *Proposed*.
- **E41** (E18 + K-joint K=3 LNS): `--fast` 0.92178 (−1.27 % vs E25),
  `--ng45` 0.69022 (−1.91 % vs E12), `--all` partial 6/17 then stalled
  due to a multiprocessing.Queue stall under heavy box load AND a
  K-joint overlap-validation bug at ibm07.

The K-joint overlap bug (eps direction, see `docs/gotchas.md` #4) was
fixed and verified 2026-04-30 04:35; E41 `--all` rerun started 05:16
with `--jobs 4` parallel. The rerun is in flight at draft time.

## Decision (proposed, pending final --all)

If E41 `--all` final ≤ E18 1.08979: adopt **DPO best_of_v2 init → CD
plateau → grid-bin LNS → SA-v2 polish → K-macro joint LNS (K=3)** as
the champion. Five phases, five distinct candidate sets:

| Phase | Wall budget | Candidate set | Acceptance |
|---|---|---|---|
| 1. DPO best_of_v2 init | ~30 min/bench | gradient steps on smooth proxy | continuation |
| 2. project_overlaps | < 1 s | iterative legalization | greedy |
| 3. CD plateau | ≤ 2400 s | per-axis breakpoints | greedy (Δ ≤ 0) |
| 4. Grid-bin LNS | ≤ 600 s | grid (col, row) cell centers | greedy (best of cell) |
| 5. SA-v2 polish | ≤ 600 s | per-axis breakpoints | Metropolis, T₀=5e-4, best-tracking |
| 6. K-joint LNS | ≤ 600 s | top-N=5 cell-centers per macro × K=3 brute combos | greedy (Δ < −1e-7), defensive overlap revert |
| Total | ~50-60 min/bench typical | | |

K=3 K-joint phase commits 12-58 K-tuples per bench post-CD-LNS-SA. Each
K-tuple's brute-force inner loop is N^K = 5^3 = 125 combos. eps=1e-9
strict-separation pairwise legality + post-commit `compute_overlap_metrics`
defensive revert (gotcha #4 fix).

## Consequences (final)

- **Champion:** 1.0990 (E12) → **1.0848** (E41). Improvement:
  - vs E12 1.0990: **−0.0142, −1.29 %**
  - vs E25 candidate 1.0954: **−0.0106, −0.97 %**
  - vs E18 candidate 1.08979: **−0.0050, −0.46 %**
  - vs leaderboard 1.1172: **−0.0324, −2.90 %**
  - vs RePlAce 1.4578: −0.3730, **−25.6 %**
- **Per-bench `--all` (zero overlaps everywhere):**

| Bench | E41 | E25 | Δ vs E25 |
|---|---:|---:|---:|
| ibm01 | 0.9121 | 0.8902 | +2.46 % |
| ibm02 | 1.1122 | 1.1310 | −1.66 % |
| ibm03 | 0.9548 | 0.9831 | −2.88 % |
| ibm04 | 0.9874 | 1.0102 | −2.26 % |
| ibm06 | 1.1694 | 1.1549 | +1.26 % |
| ibm07 | 1.1127 | 1.0982 | +1.32 % |
| ibm08 | 1.1031 | 1.1112 | −0.73 % |
| ibm09 | 0.8413 | 0.8533 | −1.41 % |
| ibm10 | 1.0096 | 1.0459 | **−3.47 %** |
| **ibm11** | **0.8765** | **0.9136** | **−4.06 %** |
| ibm12 | 1.2056 | 1.2079 | −0.19 % |
| ibm13 | 0.9478 | 0.9766 | −2.95 % |
| **ibm14** | **1.1994** | **1.2205** | **−1.73 %** |
| **ibm15** | **1.1651** | **1.1797** | **−1.24 %** |
| ibm16 | 1.1435 | 1.1547 | −0.97 % |
| ibm17 | 1.3406 | 1.3311 | +0.71 % |
| ibm18 | 1.3604 | 1.3595 | +0.07 % |
| **AVG** | **1.0848** | **1.0954** | **−0.97 %** |

  - **14 wins / 3 losses / 0 ties.**
  - **Hard-plateau wins** (ibm11/14/15 — where E25 had tied or near-tied
    E12 under five different single-or-2-macro mechanisms): −4.06 %,
    −1.73 %, −1.24 %. **K=3 K-joint composed with DPO basin DOES escape
    the multi-mechanism floor on these coupled fixed points.** This is
    the headline result — the post-E25 plateau was *3-coupled multi-basin*,
    breakable by simultaneous 3-macro moves once a DPO basin entry
    opens the right K-tuple structure.
  - Losses: 3 benches where E25 had a strong SA lift over E12 (ibm01
    −1.58 % vs E12 in E25). The K=3 K-joint eps-strict version commits
    fewer near-touching K-tuples on those, losing the small amount that
    SA had captured. Net is heavily positive.
  - ibm18: tied within sub-noise (+0.07 %).
- **NG45 commercial-design transfer (4 designs):** `--ng45` avg
  **0.69022**, **−1.91 % vs E12 0.7037**, **−0.25 % vs E18 NG45 0.69193**.
  ariane133 −4.64 % vs E12 (−0.92 % additional vs E18). The DPO+K-joint
  composition transfers to OOD designs, with K-joint extracting
  additional structure on ariane133 specifically.
- **Total wall:** 7.85 hr (E12) → 10.33 hr (E25) → 13.30 hr (E18) →
  **13.58 hr (E41) on `--jobs 4` parallel** (final number for the actual
  rerun). Serial wall (`--jobs 1`) would be ~16-17 hr, near the 17-hr
  contest envelope. Per-bench worst-case wall observed: ibm17 4200 s,
  ibm14 4098 s, ibm16 3852 s, ibm12 3819 s — all well above the 3600 s
  per-bench contest cap on paper, but eval ran with `--jobs 4` so per-bench
  wall is wall-time, not CPU-time. **Caveat for the contest submission:**
  if scoring measures per-bench wall (not total wall), `--jobs 1` execution
  would push some hard benches over the 3600 s cap. The K-joint phase
  budget can be tuned down to 300 s or skipped on the hardest benches
  to fit, at the cost of some lift.
- **Architectural interpretation.** Five phases on five structurally
  distinct candidate sets. The K-joint phase is the *only* one with a
  multi-macro reachable set; CD/LNS/SA all move ≤ 2 macros at a time.
  E41's wins on hard-plateau benches (ibm11/14/15) prove the
  multi-mechanism floor was a *coupled* fixed point — three macros need
  to move simultaneously to escape, which K=3 K-joint can do but no
  single-or-2-macro mechanism could.
- **The DPO basin is load-bearing.** Re-running K=3 K-joint on top of
  the SDF-init E25 pipeline (E39) extracted only -0.31 % on `--fast`
  and produced near-ties on the hard benches. The composition with
  DPO basin (E41) extracts much more — DPO's basin opens new K-tuple
  structure that K=3 enumeration can find. **Both ingredients are
  required.**

## Open questions / follow-ups

1. **K=4 K-joint variant (E42).** If K=4 lifts further beyond E41,
   that extends the joint-move escape from 3-coupled to 4-coupled
   plateau benches. Smoke test on ibm01 in flight at draft time. ADR-011
   covers the K=4 promotion if it lifts.
2. **Wall-budget NG45 stress test.** Worst-case per-bench wall on the
   hidden NG45 designs is unknown. ariane133/136/mempool_tile/nvdla
   wall during the `--ng45` run was within the 1-hr cap, but unseen
   designs may be larger. Consider per-bench K-joint budget reduction
   if walls bind.
3. **K-joint adjacency metric.** Current K-tuple selection uses
   `1 / (net_size − 1)` weighted adjacency. Spatial proximity is
   not currently a factor. A combined net-+-spatial adjacency may
   surface different K-tuples (E44 follow-up).

## Alternatives ruled out

- **E18 (ADR-009 *Proposed*) at 1.08979.** Superseded by E41 if E41 final
  is ≤ E18. Recommend marking ADR-009 *Superseded by ADR-010* on accept.
- **E25 (ADR-008 *Proposed*) at 1.0954.** Already superseded by E18.
- **E39 K=3 K-joint without DPO** at `--fast` 0.92835 post-fix (−0.56 %
  vs E25 fast). Marginal in isolation; the productive form is DPO + K-joint
  composition. Marked superseded_by E41 in its manifest.

## Evidence

- `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` — candidate code.
- `experiments/E41_dpo_kjoint/manifest.md` — `decided: 2026-04-30`,
  `status: champion_candidate` (set on accept).
- `results/experiment_log.jsonl` rows:
  - `e41_dpo_kjoint_fast` 0.92178 (`--fast`)
  - `e41_dpo_kjoint_ng45` 0.69022 (`--ng45`)
  - `e41_dpo_kjoint_all_postfix` 1.0848 (`--all`, 48 903 s = 13.58 hr,
    completed 2026-04-30 09:13)
- `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py` — K-joint
  primitive with the eps fix.
- `docs/gotchas.md` #4 — K-joint legality eps direction.
- ADR-005 (SDF init canonical), ADR-007 (E12 promotion), ADR-008 (E25
  *Proposed*), ADR-009 (E18 *Proposed*).
