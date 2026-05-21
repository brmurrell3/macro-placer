---
id: E134
name: pruned_restart
status: in_progress
parent: E129
created: 2026-05-21
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E134: pruned_restart

## Hypothesis
Best-of-5 dominates best-of-3 statistically (more samples from the FP32-stochastic
basin distribution). But running 5 full descents = 5×100s = 500s of GPU, leaving
little budget for CD polish under 720s/bench. SOLUTION: run all 5 seeds in serial
for 150 steps (cheap exploration), kill 3 worst by smooth proxy, finish top-2 for
350 more steps each. Pick best by canonical proxy. CD polish.

Saves ~3×80s ≈ 240s of wasted descent on losers → more CD time, or more candidates.

## Method
Subclass `SmoothGlobalPlacerV4Gaussian` to support checkpoint-resume:
  - `descend_partial(bench, plc, num_steps_a)` → (pos_a, smooth_score_a, state_dict)
  - `descend_resume(state_dict, num_steps_b)` → pos_final

5 seeds [42, 142, 242, 342, 442]:
  Phase A: each seed runs 150 steps → snapshot
  Sort by smooth proxy score → keep top-2
  Phase B: top-2 resume to 500 steps total
  Legalize both, score canonical, pick best
  CD polish ~400s

## Kill gate
ibm04 smoke proxy >= 0.92 (v2 baseline 0.929 — must beat E129/v2 by >1%). If smoke
fails by >0.5%, falsify before --all. Also kill if wall > 800s on ibm04 (budget bust).

## Generalization check
--all 17 IBM avg < 1.0575 (current champion). If 0.5% better on average AND no
single-bench regression > 2%, promote. Otherwise marginal.

## Outcome (filled when decided)
**Smoke ibm04 PASS**: proxy=**0.91679**, ovl=0, wall=278s.
Target was <0.92 (v2 baseline 0.929); E134 lift = −1.3% on ibm04.

Per-seed Phase A smooth scores (150 steps each):
  seed 42:  0.97912 (killed)
  seed 142: 0.97914 (killed)
  seed 242: 0.97876 (kept)
  seed 342: 0.97920 (killed)
  seed 442: 0.97835 (kept, BEST)

Phase B canonical proxy after legalize (350 more steps + legalize):
  seed 442: 1.06724  ← picked
  seed 242: 1.08163

CD polish ran for ~200s (plateau exit), drove 1.067 → 0.917 (−14%).
Phase A wall 42s (5 seeds), Phase B wall 24s (2 seeds), CD ~210s.
Total 278s, well under 720s budget — leaves headroom for --all on
larger benches (ibm17/18 will be 3-5× slower).

Smoke gate satisfied. Recommend --all run.

## Pointers
- Code: `code/placer.py`, `code/batched_descender.py`
- Smoke: `/tmp/e134_smoke.log`
