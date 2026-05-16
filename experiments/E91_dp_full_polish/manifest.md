---
id: E91
name: dp_full_polish
status: in_progress
parent: PATH B autopsy verification — autopsy made on basin-only data
created: 2026-05-12
decided: null
champion_at_time: 1.0612 (cascade uncapped, ibm)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E91: dp_full_polish — stock DP + full E25-equivalent polish + cascade saddle

## Hypothesis

PATH B was falsified on a basin-only test. The sweep code
(`experiments/E76_dreamplace_integration/code/dp_sweep_b1.py:10-12`) is
explicit: "no CD polish at sweep time; cheap evaluation". The polished
test (`p1b_polish_test.py`) used CD-adaptive at hard_cap_s=900 (15 min)
— *not* the full E25 pipeline (CD + LNS + SA-v2, ~660s per stage) nor
the saddle escape that produces cascade's 1.0612 result. The falsified
submission `cd_lns_sa_hessian_dp/placer.py` gave DP only **60 s** brief
CD polish (line 192: `min_time_s=30, hard_cap_s=60`) while E25 and E41
got ~660s each in the plateau pick.

**Stock DP basin + the full cascade-equivalent polish was never tested.**

The literature on init-vs-polish gaps in this codebase is consistent:
- SDF basin proxy ≈ 1.50 → E25 polish → ~1.09 (-27 % lift)
- DPO basin proxy ≈ 1.38 → E41 polish → ~1.08 (-22 % lift)
- DP basin (autopsy ibm10) = 1.2553 → ?

If DP basin polishes by 22-27 % like SDF/DPO, final proxy ≈ 0.93–0.98
on ibm10 — would beat cascade's 1.0775. The autopsy's "structural
objective mismatch can't be bridged" claim is contradicted by the
data already in this codebase if DP basins polish similarly.

## Method

Per benchmark (start with ibm10, the worst-gap bench from autopsy):

1. Stock DREAMPlace (same config as cd_lns_sa_hessian_dp).
2. greedy_macro_legalize + project_overlaps cleanup.
3. Full E25 polish pipeline with DP placement as init: run_cd_adaptive
   (~660s) + run_lns_gridbin + run_sa_polish_v2.
4. Cascading saddle escape (~1100s budget, same as cascade for E25).
5. Compare proxy + overlap_count vs cascade reference on same bench.

Total budget per bench: ~3300s (55 min). Bench order: ibm10 (worst gap),
ibm14, ibm12, ibm17.

## Kill gate

- DP+full-polish > cascade by ≥3 % on ibm10 → autopsy holds.
- Within 3 % → autopsy basin-only artifact; proceed to other benches.
- Beats cascade on any bench → autopsy wrong, B-R1 lifts on top.

## Outcome

[To be filled with measurements.]

## Pointers

- Sweep code: experiments/E76_dreamplace_integration/code/dp_sweep_b1.py
- 15-min polish probe: experiments/E76_dreamplace_integration/code/p1b_polish_test.py
- Falsified submission: submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py (line 192)
- E25 polish pipeline: submissions/cd_lns_sa/placer.py
- Cascade saddle escape: experiments/E84_cascading_saddle/code/cascading_saddle.py

## Intermediate results (2026-05-12 session)

### ibm01 (smoke, 600 s budget) — DONE 2026-05-12 23:15 UTC

| Stage | Proxy | Wall | Δ from prev |
|------|------:|-----:|------------:|
| DP basin (stock cfg, CPU) | 1.01109 | 24 s | — |
| + greedy_macro_legalize | 1.04691 | 3 s | +3.5 % |
| + run_cd_adaptive (cap 114 s) | 0.87591 | 114 s | **−16.3 %** |
| + run_lns_gridbin (34 s) | 0.87541 | 34 s | −0.06 % |
| + run_sa_polish_v2 (34 s) | 0.87541 | 35 s | tied |
| + cascading_saddle (342 s, 2 iters) | **0.86172** | 449 s | −1.6 % |
| **Total wall** | | 668 s | |

Comparison: ibm01 cached cascade uncapped (~50 min budget) ≈ 0.85.
**DP + full polish lands at 0.862 in 668 s — only +1.4 % above cascade
uncapped, despite shorter wall and CPU-only DP.**

**The autopsy's "structural objective mismatch can't be bridged"
claim is empirically wrong on ibm01.** The polish curve from
DP-legalized (1.05) → final (0.86) is a −18 % lift, slightly less
than SDF's −27 % polish curve, but the basin starts ~30 % closer to
the optimum (1.05 vs SDF 1.50), so the absolute final is comparable.

### ibm10 (full bench, 3300 s budget) — RUNNING

DP basin 1.215 (12 s, 310 overlaps) → legalize 1.444 (49 s, 0 overlaps,
+19 % over basin — concerning). In CD-adaptive (cap 639 s). Awaiting
full result.

Cascade-capped ref (autopsy): 1.0775. Cascade-uncapped local cached: 0.989.

### ibm14 (full bench, 3300 s budget) — RUNNING

Launched after smoke completed. Phase 1.

### ibm12, ibm17 — queued depending on ibm10/14 results.

### B-R4 inverted-use — falsified

PERTURB_ITERS={100, 10}: both produced proxy ~6.18 and exactly 308 505
overlaps on ibm10 (i.e. all 786 hard-macro pairs overlap). The number
786·785/2 = 308 505 matches a "everyone collapsed to one point"
configuration. Either DP's gradient pulled all cells to the same
location, or the bookshelf write-back of cascade init didn't take.
Either way the result is unusable as a perturbation.

Diagnosis: DP's optimization is destabilizing on a canonical-optimal
init at any iteration count tested. The "cascade plateau →
DP-as-perturber → re-polish" hypothesis is empirically falsified for
stock DP with this driver. May warrant retry with much smaller
learning rate (1e-5) or a non-Nesterov optimizer, but not in this
session.
