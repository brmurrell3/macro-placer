---
id: E130
name: cascade_multirestart
status: in_progress
parent: E128
created: 2026-05-20
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E130: cascade_multirestart

## Hypothesis
E128 (V4+Gaussian descent + cascade saddle escape + CD polish) and E129
(multi-restart best-of-N descent) are orthogonal wins. Stacking them
should improve over E128 alone because multi-restart picks a better
seed basin before the saddle/CD machinery refines it. Expected lift on
top of E128's --fast 0.8317.

## Method
Pipeline per bench:
1. For seed in [42, 142, 242]: run V4+Gaussian descent → legalize →
   canonical-proxy score the basin.
2. Pick best basin (min proxy).
3. Run cascading_saddle_escape on best basin
   (saddle_budget_s=240, max_iters=3).
4. Run CD polish on saddle output (cd_polish_s=480, plateau_threshold=0.001).

Budget seconds = 1800 (per-bench cap). N=3 descents at ~120s each +
saddle 240 + CD 480 ≈ 1080s expected wall, with safety margin.

## Kill gate
ibm04 smoke proxy ≥ 0.92 (i.e. fails to improve over E128 baseline
0.9165) or any overlap on any bench.

## Generalization check
If smoke passes, run --all and --ng45. Lift over E128 must hold on at
least 2/3 NG45 designs to be considered for promotion.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/placer.py`.
- Parent E128: `experiments/E128_cascade_on_v4gauss/code/placer.py`.
- Parent E129: `experiments/E129_multirestart/code/multirestart_placer.py`.
