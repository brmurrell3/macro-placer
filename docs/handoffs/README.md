# Session handoffs

Dated session-end notes captured by Claude agents during the run-up to the
2026-05-21 submission. Filed here so the repo root stays clean and the
context survives across sessions. Newest first.

Handoffs are *operational state at a point in time* — they go stale
quickly. Treat them as primary sources for "what was running / what was
decided that day," not as current truth. Canonical state lives in
`CLAUDE.md`, `docs/results.md`, `docs/experiment_index.md`, and
`writeup/evidence.md`.

| Date | File | One-line summary |
|------|------|------------------|
| 2026-05-16 | [`2026-05-16_morning_handoff.md`](2026-05-16_morning_handoff.md) | AWS CPU 17-IBM cross-validation of `placer_adaptive` (1.078 floor) in flight; GPU box blocked on quota; Xplace integration ready. Cached-best ceiling 1.05156. |
| 2026-05-16 | [`2026-05-16_tier2_orfs_findings.md`](2026-05-16_tier2_orfs_findings.md) | Tier-2 ORFS results overnight: ariane133 ships **without** macros.tcl (auto-place wins by 1.2 ns); ariane136 ships **with** cascade. Per-design strategy, not universal. |
| 2026-05-15 | [`2026-05-15_priorities.md`](2026-05-15_priorities.md) | Top-priority work items (post-research). |
| 2026-05-15 | [`2026-05-15_notes_innovation.md`](2026-05-15_notes_innovation.md) | Innovation notes: pure Lévy + multi-objective Hessian eigvec. |
| 2026-05-15 | [`2026-05-15_speculation_paths_beyond_106.md`](2026-05-15_speculation_paths_beyond_106.md) | Speculation report: paths to lift beyond 1.06 (response to "speculate and research how we can improve"). |
| 2026-05-15 | [`2026-05-15_multi_claude_coordination.md`](2026-05-15_multi_claude_coordination.md) | Multi-Claude coordination notes, including intel from partcl repo git history pre-commit `3da0a04`. |
| 2026-05-14 | [`2026-05-14_champion_found_dp_lane.md`](2026-05-14_champion_found_dp_lane.md) | `cd_lns_sa_cascade_dp_lane` verified IBM 1.06650 / NG45 0.68086 on 17 IBM + 4 NG45 (zero overlaps). Tournament v1 worse than DP-lane; tournament v2 killed when user pivoted to Tier-2. |
| 2026-05-12 | [`2026-05-12_morning_report_2.md`](2026-05-12_morning_report_2.md) | Overnight 8+ cloud experiments, 80+ lane-hours; identified `finegrain` cascade as new winner. |
| 2026-05-11 | [`2026-05-11_overnight_queue.md`](2026-05-11_overnight_queue.md) | Overnight autonomous queue snapshot: 5 cascade variants × 4 benches ETA ~21:00. |

## Still at repo root (not filed here — parallel-agent territory)

- `TIER2_FINDINGS.md` — Tier-2 ORFS findings (canonical copy in this dir; root version is the Tier-2 ORFS agent's working copy).
- `MORNING_HANDOFF_BRENDAN.md` — E107 morning handoff (E107 periphery work agent's territory).
- `LEADERBOARD.md` — operational doc tracking every placer run (kept at root alongside `SCORING.md`, `SETUP.md`).
