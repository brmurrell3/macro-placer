---
id: E109
name: stacked_cascade_portfolio
status: graduated
parent: E100
created: 2026-05-16
decided: 2026-05-17
champion_at_time: 1.06650
outcome: "IBM --all 1.0575 / NG45 0.6893 / combined 0.987. Beats Option B 1.0665 IBM by -0.85%; beats Option B 0.993 combined by -0.6%. Promoted as Option C 2026-05-17."
champion_delta: -0.85%
graduated_to: submissions/cd_lns_sa_cascade_stacked_periphery/placer.py
superseded_by: null
---

# E109: stacked_cascade_portfolio

## Hypothesis

E100 portfolio_saddle (4-weight Hessian eigenvector search) gave -1.089%
lift on cascade ibm10 plateau in spike. But running portfolio_saddle
FROM E25/E41 plateau (without prior cascade step) wastes iters on the
canonical direction (which cascade_saddle would handle faster). Stacking
cascade_saddle FIRST (canonical only) and portfolio_saddle SECOND (3
non-canonical weights: cong-focus, density-focus, non-WL) lets each
component contribute orthogonally:

  - cascade saturates canonical eigvec quickly (1-2 iters at polish=180s)
  - portfolio's 3 non-canonical weights explore directions cascade misses

Falsified prior attempt: pure portfolio from E25/E41 plateau (committed
at 88f6cc9 by parallel agent) gave 1.07404 — WORSE than Lévy alone
1.07268.

## Method

Sequential composition in `submissions/cd_lns_sa_cascade_stacked/placer.py`:
  1. E25 lane (SDF + CD + LNS + SA-v2 polish-v2) — 0.32 × B
  2. E41 lane (DPO + CD + LNS + SA-v2 + K-joint K=3) — 0.32 × B
  3. Plateau pick: best of {E25, E41} by canonical proxy
  4. Phase 3a: cascade_saddle (canonical eigvec (1, 0.5, 0.5), max_iters=5,
     polish_budget=180s) — 0.20 × B
  5. Phase 3b: portfolio_saddle ([(1,0,1), (1,1,0), (0,1,1)],
     max_iters=2, K_eps=2, polish_budget=60s) — 0.13 × B
  6. Final pick: lowest-proxy candidate with 0 overlaps

Plus periphery wrapper (`cd_lns_sa_cascade_stacked_periphery/`) that
attempts α=0.01 edge push + CD polish at the end, strict-conservative
accept (revert if overlaps or proxy regress). Observed: 100% reject
rate in --all run, but no regression risk and may help on benches we
haven't tested.

Robustness fixes shipped together:
  - `submissions/cd_lns_sa/placer.py`: `project_overlaps` fallback
    before final validity check (catches SA-v2 best-restore residuals)
  - `cd_lns_sa_cascade_stacked/placer.py`: `try/except` around E25 and
    E41 lane calls — if a lane raises (e.g., NG45 nvdla unfixable
    2-overlap), the other lane carries the placer

## Kill gate

- --all aggregate > 1.067 (loses to Option B). KILLED.
- ariane133 NG45 > 0.700 (catastrophic regression, repeating E54/E62
  failure pattern).

## Generalization check

- 17 IBM `--all` aggregate < 1.067
- 4 NG45 `--ng45` aggregate < 0.69 (within 1% of Option B 0.681)
- Zero overlaps on all benches

## Outcome (2026-05-17)

**Status: GRADUATED** to `submissions/cd_lns_sa_cascade_stacked_periphery/placer.py`.

Per-bench (M3, --jobs 4, ~3.8 hr wall):
- IBM 17 avg: 1.05750
- NG45 4 avg: 0.6893
- Combined: 0.987

Per-bench vs cached cascade (1.0612 uncapped):
8 wins (>0.5%) / 5 ties / 4 small losses (none > +1.7%):

| bench | new | cached | Δ% |
|---|---:|---:|---:|
| ibm01 | 0.8596 | 0.8453 | +1.69 |
| ibm02 | 1.0066 | 1.0210 | **−1.41** |
| ibm03 | 0.9274 | 0.9463 | **−2.00** |
| ibm04 | 0.9729 | 0.9907 | **−1.80** |
| ibm06 | 1.1336 | 1.1179 | +1.40 |
| ibm07 | 1.0529 | 1.0645 | **−1.09** |
| ibm08 | 1.0816 | 1.1007 | **−1.74** |
| ibm09 | 0.8249 | 0.8227 | +0.27 |
| ibm10 | 0.9958 | 0.9894 | +0.65 |
| ibm11 | 0.8604 | 0.8681 | **−0.89** |
| ibm12 | 1.1920 | 1.1977 | **−0.48** |
| ibm13 | 0.9363 | 0.9346 | +0.18 |
| ibm14 | 1.1908 | 1.2063 | **−1.28** |
| ibm15 | 1.1582 | 1.1551 | +0.27 |
| ibm16 | 1.1224 | 1.1085 | +1.25 |
| ibm17 | 1.3226 | 1.3316 | **−0.68** |
| ibm18 | 1.3393 | 1.3399 | −0.04 |

Portfolio chose as winner on 10/17 benches; cascade on the remaining 7.
Portfolio's non-canonical weights consistently added 0.1-0.5% lift on
top of cascade plateau where applicable.

NG45 nvdla: E25 produced 2 residual overlaps; try/except engaged, lane
fell back to E41-only. Final 0.67646 (matches cached E107 0.6765).

## Pointers

- Code: `submissions/cd_lns_sa_cascade_stacked/placer.py`
- Wrapper: `submissions/cd_lns_sa_cascade_stacked_periphery/placer.py`
- Parent saddle: `experiments/E100_weight_portfolio_saddle/code/portfolio_saddle.py`
- Parent cascade: `experiments/E84_cascading_saddle/code/cascading_saddle.py`
- Handoff: `docs/handoffs/2026-05-17_morning_champion.md`
- Memory: `champion_2026_05_17_stacked_periphery.md`
- Commit: 08f2af9
