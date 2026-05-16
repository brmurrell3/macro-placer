# TODO — submission day countdown

**Deadline:** May 21, 2026, 11:59 PT (5 days from 2026-05-16)

## Status at-a-glance

| Tier | Verified candidate | Score | Risk |
|---|---|---:|---|
| **Tier-1 proxy (top pick)** | Option B (`cd_lns_sa_cascade_dp_lane/placer.py`) | IBM **1.06650** / NG45 **0.68086** | Needs DREAMPlace setup in eval env; falls back to A if `DREAMPLACE_ROOT` unset |
| **Tier-1 fallback** | Option A (`cd_lns_sa_cascade/placer_adaptive.py`) | IBM **1.07820** / NG45 **0.68102** | None (no external deps) |
| **Tier-2 ORFS** | Per-design strategy: ariane133 no-tcl, ariane136 cascade-tcl | (mempool/nvdla untested) | Mempool + nvdla coverage |
| **Moonshot** | Xplace integration (Carrotato-class) | Target ≤ 1.00 | GPU quota blocked |

Leaderboard reference: vmallela #1 self-reported 1.0109. Gap to Option B:
**+5.6 %**. Best verified leaderboard entry below us: MTK 1.2818 (we
beat by **−16.8 %**).

## P0 — must ship by May 21

- [ ] **Pick Tier-1 entry (A or B).** B is verified strictly better on both
      gates; A is safer (no external dep). Recommendation: ship B if
      partcl judging hardware has DREAMPlace; ship A otherwise.
      *Decision artifact*: comment line in `submissions/README.md` and/or
      the submission-form upload.
- [ ] **Drain `writeup/paper.md` prose TODOs.** ~25 `TODO(prose)` markers.
      All datapoints already live in `writeup/evidence.md` §1 / §1.1;
      task is writing connective text + citations.
- [ ] **Final cross-validation on partcl-class hardware.** Re-run
      `--all` + `--ng45` on chosen placer on EPYC (AWS c6a.4xlarge spot
      is a stand-in; ~$2/run). Confirm zero overlaps and walls under
      60-min cap.
- [ ] **Submit via form**: <https://forms.gle/YDRtYV5Vq68SZgKW9>. Public
      submission; repo open-source under Apache 2.0 or GPL.

## P1 — nice-to-have if time permits

- [ ] **Xplace integration end-to-end smoke** (`experiments/Xplace_integration/`).
      If AWS GPU quota approves by May 19, run ibm01 single-bench
      `--fast` with Xplace as a 4th init lane. If proxy ≤ 1.00 in
      ≤ 30 min wall, paper.md gains a Tier-1 Option C section.
- [ ] **Tier-2 mempool_tile + nvdla coverage.** Per-design test with
      and without `MACRO_PLACEMENT_TCL`. ariane133 wins without,
      ariane136 wins with; we don't know the other two.
- [ ] **ADR-013** drafting for cascade-DP-lane promotion (the missing
      ADR for the current submission floor).

## P2 — paper polish

- [ ] **Backport post-E25 contributions** into `writeup/contributions.md`
      (mirrors `writeup/evidence.md` §1.1).
- [ ] **Compress `MEMORY.md`** — drop stale session-state entries.

## Algorithm summary (Option B = Option A + DREAMPlace lane)

Both options share the cascade base:

```
init lane 1: SDF basin           -> CD + LNS + SA-v2 polish (E25)
init lane 2: DPO basin           -> CD + LNS + SA-v2 + K-joint polish (E41)
[B only] lane 3: DREAMPlace      -> greedy_macro_legalize -> matched polish (DP-lane)
plateau pick: argmin over the 2 or 3 lane outputs
cascading saddle escape:
    while remaining > 1.2 * avg_iter_wall:
        smooth-proxy Hessian -> Lanczos smallest eigvec
        ±ε perturb along soft mode -> CD-adaptive polish
        if no improvement: break
deadline: 3000s default (50 min); 60-min cap respected
validate: zero overlaps, fixed macros immovable
```

PATH A acceleration (`macro_place/cd_core.py` delta-cost API) reduced
CD wall by **5.36×** on ibm10 via `delta_cost` / `delta_cost_axis_batch`
/ flat routing variant (commits `59a7a8b`, `53b4a26`, `af520c3`). The
2026-05-16 cascade budget-management fix (rolling
`avg_iter_wall × 1.2` vs fixed 60s safety) keeps cascade from stopping
early when one more iter could fit.

See [`docs/approach.md`](docs/approach.md) for the full mechanism
description.

## Hardware reference

- **Partcl judging hardware**: AMD EPYC 9655P, 16 cores, 100 GB,
  RTX 6000 Ada 48 GB. 60-min/bench hard timeout.
- **AWS c6a.4xlarge spot ($0.29/hr)**: EPYC-class CPU stand-in for
  cross-validation. Cross-validation chain running 2026-05-16 morning.
- **AWS g5.xlarge spot (~$0.50/hr)**: A10G for Xplace integration once
  quota approves. Quota poll daemon at `/tmp/aws_quota_poll.log`.
- **Local M3 Max**: prototype only; per-core ~2× faster than EPYC,
  results NOT predictive of submission performance.

Required cloud env on EPYC: `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
MKL_NUM_THREADS=8` (otherwise numpy defaults to 1 thread; ~3-9× slowdown).

## Codepaths (load-bearing)

- `submissions/cd_lns_sa_cascade/placer_adaptive.py` — Option A entry
- `submissions/cd_lns_sa_cascade_dp_lane/placer.py` — Option B entry
- `submissions/cd_lns_sa_cascade/placer.py` — wall-safe E84 cascade base
- `submissions/cd_lns_sa/placer.py` — E25 SDF lane (component)
- `macro_place/cd_core.py` — PATH A delta_cost optimization (load-bearing)
- `macro_place/incremental_evaluator.py` — E1 4657× speedup (load-bearing)
- `experiments/E74_hessian_saddle/code/hessian_saddle.py` — saddle primitives
- `experiments/E84_cascading_saddle/code/cascading_saddle.py` — cascade logic + 2026-05-16 budget fix
- `experiments/E91_dp_full_polish/code/dp_full_polish.py` — DP-lane (Option B)
- `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` — E41 DPO lane (component)

## See also

- [`docs/results.md`](docs/results.md) — verified per-benchmark numbers
- [`docs/roadmap.md`](docs/roadmap.md) — day-by-day plan
- [`docs/approach.md`](docs/approach.md) — full mechanism description
- [`docs/experiment_index.md`](docs/experiment_index.md) — every experiment, post-E84 wave
- [`docs/handoffs/`](docs/handoffs/) — dated session notes
- [`writeup/evidence.md`](writeup/evidence.md) §1, §1.1 — frozen-number archive
- [`writeup/paper.md`](writeup/paper.md) — innovation prize draft
- [`submissions/README.md`](submissions/README.md) — entry catalog
