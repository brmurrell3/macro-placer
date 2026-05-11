---
id: E39
name: kmacro_joint_lns
status: marginal
parent: E25
created: 2026-04-29
decided: 2026-04-30
champion_at_time: 1.0990
fast_outcome: 0.93070 (--fast initial run, 2026-04-29); -0.31 % vs E25 fast 0.9336; per-bench wins on ibm09 (-0.79 %) and ibm13 (-0.63 %); ties on ibm01/ibm04; zero overlaps; K-joint phase committed 12-58 K-tuples per bench (Δ -0.0013 to -0.0047) — proves the K=3 joint move type extracts wins post-CD-LNS-SA. **Post-fix --fast rerun (2026-04-30 07:05): 0.92835, -0.56 % vs E25, all 4 benches improved vs initial run** (ibm01 0.8883, ibm04 1.0081, ibm09 0.8484, ibm13 0.9686; per-bench wins -0.30 % to -0.82 % vs E25). Strict-eps fix produces *cleaner* K-joint enumeration with no near-overlap candidates → marginal lift over the loose-eps version.
ng45_outcome: 0.70126 (--ng45 avg of 4 commercial designs); -0.35 % vs E12 NG45 0.7037; ariane133 -0.76 %, ariane136 -0.54 %, mempool_tile/nvdla ~tie. K-joint adds marginal lift on NG45 but E18 (-1.67 %) dominates on OOD.
all_outcome: CRASHED (--all initial run, 2026-04-29); ibm07 produced 1 overlap (area 0.0000) → validation raised RuntimeError on bench 7 of 17 (after 6 valid benches). Bug in `_ktuples_pairwise_legal` check: K-joint commits a near-overlap that passes the eps=1e-4 pairwise check but fails `compute_overlap_metrics` boundary. Float-precision regression — the K-joint mechanism extracts wins on --fast but produces invalid placements at scale on --all. Partial results: ibm01 0.8926, ibm04 1.0131, ibm03 0.9828, ibm02 1.1181, ibm06 1.1475, ibm09 0.8521 (sum 6.0062 vs E25 sum 6.0227 = -0.27 %). **BUG FIXED 2026-04-30 04:35:** changed `_is_legal_2d_excluded` and `_ktuples_pairwise_legal` from `< min - eps` (eps=1e-4, tolerated overlap) to `< min + eps` (eps=1e-9, requires strict separation). Added defensive `compute_overlap_metrics` post-commit revert. **VERIFIED 2026-04-30 04:56:** single-bench ibm07 rerun gives proxy=1.09857, VALID, K-joint Δ=-0.00276 (61 commits), zero REVERT events, total wall 2392 s = 40 min. Pattern documented in `docs/gotchas.md` #4.
outcome: marginal (--fast -0.31 %, below 0.5 % gen-check threshold). The K-joint mechanism is real (12-58 commits/bench post-CD-LNS-SA, Δ -0.0013 to -0.0047) but small magnitude. K-joint composes more strongly with DPO basin (E41 --fast -1.27 %, --ng45 -1.91 %); E41 is the stronger candidate. E39 --all not rerun standalone because E41 --all (running 2026-04-30 05:16, --jobs 4) is the more useful test of K-joint at scale.
champion_delta: -0.31 % (--fast); n/a (--all)
graduated_to: null
superseded_by: E41 (DPO basin + K-joint composition)
---

# E39: kmacro_joint_lns

## Hypothesis
E25 (CDLNSSA, avg `--all` 1.0954) ties E12 champion exactly on the
hardest benches (ibm11/13/14/15) — the same coupled fixed point under
five distinct mechanisms (CD per-axis breakpoints, grid-bin single-macro
LNS, SA-v2 on per-axis breakpoints, pair-swap, spatial cluster). All
five share the same reachable set in spirit: each move adjusts ≤ 2 macros
at a time. If those plateaus are floor-of-multi-basin (multiple coupled
macros need to move *simultaneously* to escape, no single-macro nor
2-macro adjustment improves), then K-macro joint reinsertion is a
genuinely new move type with a different reachable set, and we should
see lift on those benches.

If E39 still ties on ibm11/13/14/15, the floor is *deeper* than 3-coupled
multi-basin and the right next move type is K=4–5 or LP-relax / MIQP.

## Method
After the E25 pipeline (CD ≤ 2400 s + LNS ≤ 600 s + SA-v2 ≤ 600 s)
converges, run a 600 s K-macro joint LNS phase:

1. Coupling metric: per-macro adjacency = sum of `1 / (net_size - 1)`
   over nets the macro participates in (HPWL net weighting, reuses
   `evaluator.macro_to_nets` and `evaluator.net_pins`). Sort hard
   movables by adjacency descending.
2. Iteration loop. First pass: walk the sorted list in K-tuples (top-K,
   next-K, …). Subsequent passes: random sample of K-tuples from the
   top-3K macros (seeded RNG).
3. Per K-tuple:
   - For each macro k_i in the tuple INDEPENDENTLY: enumerate every legal
     grid-bin cell `(col, row)` where placing k_i alone is overlap-free
     w.r.t. the other (non-K-tuple) macros. Score by single-macro
     proxy delta (move + revert; never commit during enumeration).
     Keep the top-N=5 candidates per macro.
   - Brute-force the cartesian product of N^K = 125 combos. For each:
     pairwise non-overlap check among the K macros at their proposed
     centers; if pass, apply the K moves, record proxy, revert all K to
     baseline.
   - If the best combo improves baseline by > 1e-7, commit it. Else
     revert all K, advance to the next K-tuple.
4. Stop when budget exhausted OR a full pass over the candidate
   K-tuples produces no improvement.

K=3, top_N=5, kjoint_budget=600 s, kjoint_seed=42. Plus all E25 kwargs.
All hyperparameters global; no per-benchmark tuning.

## Kill gate
- **Regression:** if avg `--fast` > E25 fast 0.9336 + 0.5% (i.e., > 0.9383)
  → kill, status=falsified.
- **No lift on hard plateau:** if `--all` reproduces E25 ties on
  ibm11/13/14/15 with avg ≥ E25 1.0954 (i.e., zero net lift), mark as
  marginal — K=3 doesn't escape multi-basin floor.

## Generalization check
If `--fast` passes the regression gate AND shows ≥ 0.1% lift over
E25 fast 0.9336, validate on NG45 ariane133 before queueing `--all`.

Wall budget: ~10–12 hr `--all` (E25 base ≈ 10.3 hr + ~2 hr K-macro phase).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_kjoint.py` (defines `CDLNSSAKJointPlacer`).
- Parents: E25 (CDLNSSA candidate). Inlines `run_lns_gridbin`,
  `run_sa_polish_v2`, `_is_legal_2d` from E25.
- Related: E29 (MIQP joint, requires Gurobi); E15 (pair-swap, K=2);
  E32 (spatial cluster, K-region but greedy).
- Discussion: `docs/experiment_index.md`; `writeup/evidence.md` if
  graduated.
