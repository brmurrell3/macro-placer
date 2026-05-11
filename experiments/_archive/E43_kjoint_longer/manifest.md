---
id: E43
name: kjoint_longer
status: falsified
parent: E41
created: 2026-04-30
decided: 2026-04-30
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; ADR-010 *Proposed*)
fast_outcome: 0.9187 (--fast); -0.33 % vs E41 fast 0.92178; tied with E42 K=4 fast 0.9186. Per-bench: ibm01 0.9077, ibm04 0.9959 (+1.16 % LOSS vs E41), ibm09 0.8284, ibm13 0.9427 (-0.68 % vs E41 — only bench where longer budget genuinely helped). Per-bench variance dominated by DPO seed-noise (~1 % per bench). ibm13's 72 commits in pass 1 (vs ibm01/04/09's 2-50 commits, all converging in pass 2) confirms K-joint is *budget-bound on large benches, saturation-bound on small ones* — but the differential lift only shows on the largest --fast bench (3450 macros).
ng45_outcome: 0.7009 (--ng45); **+1.55 % REGRESSION vs E41 ng45 0.69022**. Per-design: ariane133 0.7009 (+4.10 % LOSS vs E41 0.6733), ariane136 0.6895 (+2.48 % LOSS vs E41 0.6728), mempool_tile 0.7366 (-0.12 % sub-noise tie), nvdla 0.6764 (-0.13 % sub-noise tie). Same overfit pattern as E42 K=4 NG45 but hits BOTH ariane designs. Longer budget on the K-joint phase exposes the placer to additional commits that don't generalize — the NG45 ariane designs apparently have a different K-tuple structure where K=3 K-joint at 600 s is the sweet spot.
outcome: falsified — IBM-fast lift was right at noise threshold (0.33 %) but NG45 generalization fails by +1.55 % avg, with catastrophic regressions on the two ariane designs. Same overfit class as E42 K=4. The K-joint mechanism saturation floor at K=3, 600 s budget, netlist-adjacency selection is the right operating point for NG45 transfer.
champion_delta: --fast -0.33 % (marginal); --ng45 +1.55 % (falsified)
graduated_to: null
superseded_by: null
---

# E43: kjoint_longer

## Hypothesis
E41 K=3 K-joint (600 s budget per bench) commits 12-58 K-tuples
post-CD-LNS-SA. On --fast, the K-joint phase converges before budget
on most benches (pass 1 commits then pass 2 zero), so 600 s is enough.
But on harder benches (especially the multi-coupled plateaus where
K=3 finds wins), the budget may bind: more iterations could find
more K-tuples.

If K-joint is *budget-bound* (extending budget extracts more lift),
E43 (1200 s K-joint budget) lifts further. If K-joint is
*saturation-bound* (the K=3 reachable set's wins are exhausted in
~600 s), E43 ties E41 — analogous to E26 vs E25 where extending the
SA budget produced only float-drift fluctuation.

The expected outcome is *saturation-bound on easy benches,
budget-bound on hard benches*. ibm11/14/15/17/18 are the test cases.

## Method
Same pipeline as E41 (DPO best_of_v2 init -> CD plateau -> grid-bin
LNS -> SA-v2 polish -> K-joint LNS -> validate) with **kjoint_budget_s
= 1200** (was 600). All other hyperparameters identical (K=3, top_N=5,
seed=42).

CD/LNS/SA budgets unchanged. Total per-bench wall: 2400 + 600 + 600 +
1200 = 4800 s ≈ 80 min/bench (over the 1-hr-per-bench contest cap on
paper, but in practice CD plateaus well below 2400 s on most benches
and the cap is per-bench-evaluation, so realistic wall ≈ 60-70 min).
**Research probe — over contest cap on paper.**

## Kill gate
- **Tied with E41:** if avg --fast within ±0.05 % of E41 fast 0.92178,
  K-joint is saturation-bound at 600 s on the --fast set; mark
  marginal.
- **Regression:** if avg --fast > E41 fast 0.92178 + 0.5 %, kill.
  (Unlikely; longer budget is strictly more search.)

## Generalization check
If --fast lifts ≥ 0.3 % vs E41 fast 0.92178, run --ng45.
If --ng45 lifts ≥ E41 ng45 0.69022, queue --all (will exceed 17-hr
total wall on `--jobs 4`; partial result acceptable for a research
probe).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint_longer.py`.
- Parents: E41 (DPO + K=3 K-joint at 600 s budget).
- Related: E26 (longer SA budget — falsified, SA was saturation-bound).
- Discussion: `docs/experiment_index.md`.
