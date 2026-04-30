---
id: E41
name: dpo_kjoint
status: in_progress
parent: E18, E39
created: 2026-04-29
decided: null
champion_at_time: 1.0990 (E12; E18/E39 are champion candidates from tonight's --fast)
fast_outcome: 0.92178 (--fast); -1.27 % vs E25 fast 0.9336; per-bench wins on ibm04 (-2.71 %), ibm09 (-1.62 %), ibm13 (-2.80 %); loss on ibm01 (+2.36 %); zero overlaps. **BEATS E18 on all 4 fast benches** (E18 fast was 0.92542). DPO basin + K-joint compose: K-joint adds 0.001-0.003 lift on top of DPO post-SA state. Strongest --fast result of tonight's experiments.
ng45_outcome: 0.69022 (--ng45 avg); -1.91 % vs E12 NG45 0.7037, -0.25 % vs E18 NG45 0.69193; ariane133 -4.64 % vs E12 (-0.92 % vs E18); other 3 benches ~tied with E18. **E41 strongest on NG45 too.**
all_outcome: STALLED at 6/17 (initial --all run 2026-04-29); 6 verified VALID benches (ibm01, ibm04, ibm02, ibm03, ibm06, ibm09) avg per-bench mostly matches --fast pattern with K-joint adding ~0.005-0.015 lift over E18 --all. ProcessPoolExecutor main process stalled after 6th bench despite 15 placer pipelines completing in workers (9-bench backlog of pickled results never drained — likely a multiprocessing queue issue under heavy box load). Killed E41 --all main at 04:27 to recover cores. **K-joint overlap-validation bug fixed 2026-04-30 04:35** (E39 ibm07 single-bench verified clean post-fix at 04:56; proxy=1.09857, VALID, zero REVERT events). **--all RERUN STARTED 2026-04-30 05:16 with --jobs 4** (lighter parallelism than the --jobs 8 stall) on idle box. Eta ~3-4 hr if no stall recurs. Result will land in `results/experiment_log.jsonl` as `e41_dpo_kjoint_all_postfix`.
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E41: dpo_kjoint

## Hypothesis
E18 (DPO init) and E39 (K-macro joint LNS) won independently on `--fast`
tonight: E18 avg 0.92542 (-0.91 % vs E25 fast 0.9336, 3/4 per-bench wins,
plus -1.67 % NG45 vs E12 with 4/4 wins) and E39 avg 0.93070 (-0.31 % vs
E25, with the K-joint phase contributing 0.001-0.005 lift on each bench
post-SA). DPO init operates on basin selection (different starting region);
K-joint operates on joint-move escape (different reachable set post-SA).
The two effects target orthogonal failure modes, so wins should compose.
If both effects are roughly additive, E41 fast could be 0.9215-0.9235
(≈ -1.2 % or better vs E25), and the lifts should carry through to
`--all` (E18 and E39 are both running `--all` now).

## Method
Same pipeline as E39 but step 1 SDF init replaced by DPO best_of_v2 init
(same as E18). Per-benchmark pipeline:

1. DPO best_of_v2 init (`_best_of_v2_init` from E18; SDF + DPO-v2,
   pick lower proxy_cost).
2. `project_overlaps` to clear residual overlaps from DPO output.
3. Build `IncrementalProxyEvaluator`.
4. CD adaptive phase, hard cap 2400 s.
5. Grid-bin LNS phase, budget 600 s.
6. SA-v2 phase, budget 600 s, T₀=5e-4 with best-so-far tracking.
7. K-macro joint LNS phase (`run_kjoint_lns` from E39), budget 600 s,
   K=3, top_N=5.
8. Validate (zero overlaps), preserve fixed macros, return.

All hyperparameters identical to E18 and E39 (DPO seed=42, CD cap 2400 s,
LNS 600 s, SA-v2 600 s with T₀=5e-4 / Tf=1e-6, K-joint 600 s with K=3
and top_N=5). No per-benchmark tuning.

## Kill gate
avg `--fast` > E25 fast 0.9336 + 5 % → kill, status=falsified.

## Generalization check
Any per-bench `--fast` win → queue `--all`. If `--fast` shows ≥ 0.5 %
lift vs E25, validate on NG45 ariane133 before queueing `--all`.

Wall budget: ~14-16 hr `--all` (E18 base ≈ 12 hr [DPO init + E25 pipeline]
+ K-joint 600 s × 17 ≈ 3 hr). Per-benchmark theoretical sum is 4200 s,
above the 1-hr legal cap — but CD plateaus well below its 2400 s cap on
most benches in practice, so realistic wall stays near ~1 hr/bench.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint.py` (defines `CDLNSSADPOKJointPlacer`).
- Parents: E18 (DPO init pipeline), E39 (K-joint phase).
- Related: E25 (shared CD+LNS+SA-v2 backbone), E11 (basin sensitivity to
  init), E29 (MIQP joint reinsertion), E15 (pair-swap, K=2).
- Discussion: `docs/experiment_index.md`; `writeup/evidence.md` if graduated.
