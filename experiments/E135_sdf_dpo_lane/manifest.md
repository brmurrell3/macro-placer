---
id: E135
name: sdf_dpo_lane
status: in_progress
parent: E127
created: 2026-05-21
champion_at_time: 1.0575
decided: null
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E135: sdf_dpo_lane

## Hypothesis
DPO-init finds structurally different basins than SDF (E18 verified —
3/4 fast wins, 4/4 NG45 wins as init for E25). V4+Gauss has never been
run from DPO init as a separate lane. Running V4+Gauss twice — once
from SDF-init, once from DPO-init — and picking the better canonical
basin should beat single-lane V4+Gauss on at least 1/4 of benchmarks
where DPO basin dominates.

## Method
Two-lane init:
  Lane A: V4+Gauss with init="sdf"  (500 steps, legalize)
  Lane B: V4+Gauss with init="dpo"  (500 steps, legalize) — V4's
          init="dpo" calls `_best_of_v2_init` from E18 which itself
          picks better of {raw SDF, DPO-v2 optimized}. The descent
          then starts from a different basin attractor than Lane A.
Score both by canonical proxy, pick better, CD polish on winner.

Budget 900s/bench. Each lane ~150-250s descent + legalize, CD polish
~400s. Tight on largest benches; defaults aim for 900s safety margin
under the 1-hr cap.

## Kill gate
ibm04 smoke proxy >= 0.92 (worse than v2 baseline 0.929 by >1%) → kill.
Also kill if wall > 900s on ibm04 (budget bust).

## Generalization check
--all 17 IBM avg < 1.0575 (current champion 2026-05-17 stacked_periphery).
If 0.5% better AND no single-bench regression > 2%, queue --ng45 run.

## Outcome (filled when decided)
**Smoke ibm04 MARGINAL**: proxy=**0.92580**, ovl=0, wall=264s.
Kill gate <0.92 missed by 0.6%; tied with v2 baseline 0.929 (within M3
run-variance 0.3-0.5%); worse than E134 baseline 0.917 by +0.9%.

Per-lane:
  Lane A (SDF): smooth=0.96185 ovl_pre=28 → legalize left 1 overlap → SKIPPED
  Lane B (DPO): smooth=0.95939 canonical=1.07895 ovl_pre=15 ovl_post=0 wall=30s
  [PICK] DPO basin (only viable lane)

CD polish drove 1.079 → 0.926 (−14% lift from DPO basin).

CAVEAT: Lane A SKIPped due to legalize residual (project_overlaps bounded at
50 iters left 1 overlap). The two-lane diversity hypothesis was NOT
tested on this smoke — only the DPO lane ran. Result is effectively
"single-lane V4+Gauss from DPO init" = 0.9258 ≈ v2 single-lane baseline.

Recommendation: DO NOT run --all without a fallback for legalize failures
in Lane A (e.g. retry with different seed, or fall back to descent-final
positions even with residual overlaps, then trust CD polish to resolve).
Even if both lanes worked, picking the better of {0.9258, ~equiv} is unlikely
to beat E134's 0.917 ibm04 baseline by enough margin to justify the 2x lane cost.

## Pointers
- Code: `code/placer.py`
- Parent inits: E127 V4+Gaussian descend; E18 `_best_of_v2_init`
- Smoke log: `/tmp/e135_smoke.log`
