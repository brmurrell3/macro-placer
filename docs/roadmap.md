# Roadmap — submission day countdown

Last updated: 2026-05-16
Competition deadline: **May 21, 2026, 11:59 PT** (5 days)

## Status

| Tier | Best verified | Score | Notes |
|---|---|---:|---|
| Tier-1 proxy IBM `--all` | Option B (`cd_lns_sa_cascade_dp_lane`) | **1.06650** | +5.6 % gap to vmallela #1 self-report 1.0109 |
| Tier-1 proxy NG45 `--ng45` | Option B / A (tied within noise) | **0.68086** (B), 0.68102 (A) | — |
| Tier-2 ORFS | Per-design strategy verified | (untested aggregate) | ariane133 no-tcl, ariane136 cascade-tcl |
| Moonshot | Xplace integration | Target ≤ 1.00 | Blocked on GPU quota |

Verified leaderboard standings (2026-05-16):

| Rank | Team | Verified | Method | Δ vs Option B |
|---|---|---:|---|---:|
| 1 | vmallela | 1.0109 (self-report) | Hessian-class saddle escape | +5.6 % gap |
| 2 | Carrotato | 0.967 (self-report) | Xplace + Triton | +10.3 % gap |
| (us) Option B | CDLNSSACascadeDPLane | **1.0665** verified | cascade + DP lane | — |
| (us) Option A | CDLNSSACascadeAdaptive | **1.07820** verified | wall-safe cascade | +1.1 % |
| 3 | KLA MACH | 1.2121 verified | ProxCD + LNS | -13.6 % |
| 9 | MTK | 1.2818 verified | DREAMPlace++ | -16.8 % |

Both Options beat every **verified** leaderboard entry by ≥13 %.

---

## Day-by-day (May 16 → May 21)

| Day | Work | Outcome |
|---|---|---|
| **Mon May 16 (today)** | Pre-writeup doc cleanup + commits | Repo in good state for paper drafting |
| Tue May 17 | Cross-validate Option A on AWS EPYC stand-in; settle A-vs-B decision | Final placer choice locked |
| Wed May 18 | Paper draft pass 1: drain `TODO(prose)` markers in `writeup/paper.md` | Paper draftable |
| Thu May 19 | Paper draft pass 2 + figure generation; Tier-2 mempool/nvdla coverage if time | Paper near-final |
| Fri May 20 | Final verification on partcl-class hardware; freeze placer | Submission-ready |
| **Sat May 21** | Submit via form, share repo with judges | Done |

If Xplace GPU quota approves any day before May 19, insert a 1-day
Xplace smoke between paper passes (potential Option C closer to
leaderboard #1).

---

## Priority order

### P0 — must ship

- [ ] Pick Tier-1 entry (A or B). B is verified better on both gates;
      A is safer (no external dep). Recommendation: ship B if partcl
      judging hardware has DREAMPlace; else A.
- [ ] Drain `writeup/paper.md` prose TODOs (~25 markers). All
      datapoints already live in `writeup/evidence.md` §1 / §1.1.
- [ ] Final `--all` + `--ng45` verification on partcl-class hardware
      with 60-min/bench cap.
- [ ] Submit: <https://forms.gle/YDRtYV5Vq68SZgKW9>.

### P1 — nice-to-have

- [ ] Xplace integration (if GPU quota approves) — moonshot toward
      Carrotato-class 0.967.
- [ ] Tier-2 mempool_tile + nvdla coverage (currently only ariane133
      and ariane136 verified).
- [ ] Draft ADR-013 for the cascade-DP-lane promotion.

### P2 — postmortem only

- [ ] Backport post-E25 contributions into `writeup/contributions.md`.
- [ ] Compress `MEMORY.md` — drop stale session-state entries.
- [ ] Sweep variants archive cleanup — already moved 15 to `_archive/`
      this session; no follow-up needed.

---

## Reference baselines

| Method | Score | Source |
|---|---:|---|
| Greedy row (demo) | 2.2109 | `submissions/examples/` |
| SA baseline | 2.1251 | TILOS MacroPlacement |
| Will's pre-fork seed | 1.5338 | `submissions/_archive/will_seed/` |
| RePlAce baseline | 1.4578 | TILOS MacroPlacement |
| Public leaderboard ref | 1.1172 | partcl README |
| **vmallela #1 (self-report)** | **1.0109** | unverified |
| **Carrotato #2 (self-report)** | **0.967** | Xplace + Triton, unverified |
| **Our Option A (verified)** | **1.07820** | `placer_adaptive.py` |
| **Our Option B (verified)** | **1.06650** | `cd_lns_sa_cascade_dp_lane/` |
| Theoretical ceiling (per-bench best across 121 cached `.pt`) | 1.05156 | NOT directly usable — rule-compliance violation |

---

## Submission requirements (from partcl rules)

- ≤ 60 min wall per benchmark (hard timeout).
- Zero hard-macro overlaps (no tolerance).
- General algorithm, no per-benchmark dispatch.
- Hard-macro orientation flips limited to Klein-4 (N, FN, FS, S).
- Open-source under Apache 2.0 or GPL for winners.
- Eval hardware: AMD EPYC 9655P (16-core), 100 GB, RTX 6000 Ada 48 GB.

---

## Risks

| Risk | Mitigation |
|---|---|
| Option B's DREAMPlace dep fails on partcl eval env | Falls back to A automatically if `DREAMPLACE_ROOT` unset. |
| Cascade saddle wall overruns on hard benches | Wall-safe variants enforce `budget_seconds=3000` (50 min, 10-min margin). 2026-05-16 budget-management fix uses rolling `avg_iter_wall × 1.2`. |
| Cross-validation on AWS EPYC differs from partcl EPYC | OpenBLAS thread env required: `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Without these, numpy defaults to 1 thread → 3-9× slowdown. |
| Per-bench-tuning accusation | Both Options use a single algorithm with the same hyperparameters on every bench. Verified by inspection. |
| Tier-2 ORFS failure on hidden NG45 designs | Auto-place fallback documented per-design; cascade only shipped on ariane136 (verified win). |

---

## See also

- [`../TODO.md`](../TODO.md) — task-level breakdown
- [`results.md`](results.md) — verified per-benchmark tables
- [`approach.md`](approach.md) — algorithm description
- [`experiment_index.md`](experiment_index.md) — every experiment run
- [`handoffs/`](handoffs/) — dated session notes
- [`../submissions/README.md`](../submissions/README.md) — entry catalog
- [`../writeup/paper.md`](../writeup/paper.md) — innovation prize draft
