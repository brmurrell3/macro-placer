---
id: E148
name: saddle_after_cd
status: falsified
parent: E138
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.984   # v2-extCD EPYC --all combined
outcome: ibm17 1.17860 smoke; saddle saturated 1 iter; same noise band as v2-extCD / E138
champion_delta: null
graduated_to: null
superseded_by: null
---

# E148: saddle_after_cd

## Hypothesis

Cascade saddle escape has more lift available on the CD local optimum
than on the Adam descent basin. E128/E138 ran saddle on Adam descent
output (high proxy ~0.96-1.28) and tied at --all. CD polish takes the
Adam basin down to the canonical local optimum (~0.82-1.20); the
Hessian structure at the CD local optimum is different from the Hessian
at the Adam descent point, and the softest mode there may point in a
structurally different escape direction.

## Method

Reorder the E138 pipeline so CD polish runs BEFORE saddle escape, not
after:

1. V4+Gaussian descent (same as v2)
2. Legalize + project_overlaps
3. CD polish 1 — 700s (full polish to plateau)
4. Bounded cascade saddle escape on the CD-polished position
   (`eigsh_maxiter=50`, `tol=1e-2`, `saddle_budget=120s`, `max_iters=2`)
5. CD polish 2 — 200s (polish saddle output)
6. Ship CD2 if it beats CD1 baseline; else ship CD1 (safety floor)

Total budget 1500s/bench. Reuses `bounded_saddle.py` from E138 unchanged.
No mutation of E84 / E128 / E138 source.

## Kill gate

Smoke EPYC ibm17 — if final proxy >= 1.18269 (v2-extCD baseline) within
noise, kill. Bench-specific lift in saddle log without final-proxy lift
also kills (means saddle accepted but CD2 didn't preserve the
improvement).

## Generalization check

If smoke passes (< 1.18269), recommend --all on EPYC against v2-extCD
combined 0.984.

## Outcome (filled when decided)

Smoke EPYC ibm17 (2026-05-21):
- descent 224s -> 1.27863
- CD1 791s -> 1.17860 (delta -0.10003)
- saddle 500s, 1 iter: lambda_min = -0.6715 found, but eps perturbations did not improve on CD1 ("no improvement (< 1e-05); saturated"). Saddle ACCEPT logged at equality.
- CD2 128s on saddle output -> 1.18032 (regressed); safety floor ships CD1 1.17860.
- Total wall 1737s; harness `timeout 1800` consumed JSON summary (placer itself finished cleanly, zero overlaps).

vs ibm17 EPYC references (run-to-run noise ~0.3-0.5%):
- v2-extCD                 1.18269 (smoke target)
- v2-extcd_2way            1.17715
- v2_xcd1200               1.17648
- E138 smoke (saddle on Adam) 1.17709
- E138 all_ibm17           1.17996
- **E148 smoke             1.17860**

E148 lands in the same ~1.176-1.180 noise band as v2-extCD and E138 - no lift from reordering. Hypothesis FALSIFIED: at the CD local optimum the softest mode exists (lam_min = -0.67) but every +/-eps perturbation along it failed to lift after a polish pass. CD2 even regressed, so the safety floor was the only thing preventing a slight loss.

Decision: do not run --all. With ~60 min wall budget remaining and 17 IBM benches at 25 min/bench, --all is not feasible before deadline anyway, and the smoke result is solidly in the same band as the shipped placer.

## Pointers
- Code: `code/placer.py`
- Reuses: `experiments/E138_bounded_saddle/code/bounded_saddle.py`
- Results: `results/`
