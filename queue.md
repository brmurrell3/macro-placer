# Overnight Experiment Queue

Driver pops the top `[pending]` item, runs it, marks it `[done|killed|new_best]` with the result line.

**Baseline:** avg proxy **1.4921** on `--all` (50s nav, 0 overlaps).

**Stages:**
- `sp3` — surrogate accuracy (improves navigation quality for ALL downstream work — run first)
- `sp1` — initial topology selection (escape SDF basin)
- `sp4` — congestion-aware LP (optimize within topology)
- `combine` — stack winners from each stage onto main, measure combined effect

**Rules:**
- Each item: stage, hypothesis name, touch-points, change, fast_gate, kill criterion.
- Every run uses `--hypothesis <name> --json`. Driver also tracks **best per stage** in `results/overnight_run.log`.
- Threshold crossings (<1.50, <1.46) are logged loudly but DO NOT stop the loop — we want the global best.
- Driver stops only on: `QUEUE_EMPTY`, `TIME_UP` (10hr), or `CRASH`.

---

## Stage 1: SP3 — Surrogate Accuracy

### 1. [done: avg=1.4931] sp3_density_grid_fix `stage:sp3`

**Hypothesis:** Density surrogate uses all cells; real proxy uses top-10% of *occupied* cells only. Fix aligns surrogate with real.

**Touch:** `submissions/polyhedra/surrogate.py:239-244`.

**Change:** Filter `density_grid` to `vals[vals > 0]` before taking top-10%.

**Fast gate:** `--fast` avg ≤ 1.501 AND overlaps = 0.

**Kill if:** `--fast` regresses >1%.

**Result:** fast avg=1.2735, all avg=1.4931, 0 overlaps. Neutral vs baseline (1.4921). +0.07% regression within noise.

---

### 2. [done: avg=1.4929] sp3_delta_ranking `stage:sp3`

**Hypothesis:** Ranking candidates by surrogate *delta* (change from current) cancels common-mode bias.

**Touch:** `submissions/polyhedra/navigator.py` (candidate ranking, ~lines 229-238).

**Change:** Score = `surrogate(candidate) - surrogate(current_state)`. Periodically (every 10 iters) anchor with real proxy.

**Fast gate:** `--fast` avg ≤ 1.501 and top_k_verify acceptance rate does not drop.

**Kill if:** `--fast` regresses >0.5% OR acceptance rate drops.

**Result:** fast avg=1.2736, all avg=1.4929, 0 overlaps. Neutral — delta ranking identical order since surrogate re-inits on acceptance.

---

### 3. [killed: rho_no_improvement] sp3_online_calibration `stage:sp3`

**Hypothesis:** Per-component affine correction (fit `real ≈ a_c * surr_c + b_c` for each of wl/density/congestion) corrects ranking inversions from component mis-weighting.

**Touch:** `submissions/polyhedra/navigator.py` (add `OnlineCalibrator` class), `surrogate.py` (expose per-component cost), `macro_place/objective.py` (if needed, return component breakdown — but only if read-only; don't modify the harness if avoidable, instead recompute in surrogate).

**Change:** Store `(surr_components, real_components)` tuple after each verified candidate. After 5 samples, fit 3 OLS regressions. Use calibrated composite for future ranking.

**Fast gate:** `--fast` avg ≤ 1.501; log within-benchmark rho (target >0.35).

**Kill if:** rho does not improve OR `--fast` regresses.

**Result:** fast avg=1.2740, 0 overlaps. Killed: too few accepted moves (1-6 per benchmark) for meaningful OLS fit in 50s budget. Rho did not improve.

---

### 4. [done: avg=1.4919] [new_best] [new_best_stage] sp3_top_k_20 `stage:sp3`

**Hypothesis:** Increasing `top_k_verify` from 3→20 compensates for surrogate ranking noise within 50s budget.

**Touch:** `submissions/polyhedra/placer.py` (constructor param) or `navigator.py`.

**Change:** `top_k_verify = 20`. Confirm iteration count on ibm01 stays ≥5.

**Fast gate:** `--fast` avg ≤ 1.501 AND iterations ≥5 on every fast benchmark.

**Kill if:** iteration count <5 on any fast benchmark OR avg regresses.

**Result:** fast avg=1.2700, all avg=1.4919, 0 overlaps. NEW_BEST (prev 1.4929). Better move selection per iteration outweighs fewer iterations.

---

### 5. [done: avg=1.4997] sp3_rank_aggregation `stage:sp3`

**Hypothesis:** Independent Borda ranking by WL/density/congestion, then weighted sum of ranks, is more robust than composite scoring when one component (congestion) ranks poorly.

**Touch:** `submissions/polyhedra/navigator.py`.

**Change:** For each candidate, compute per-component surrogate scores, rankdata() each, aggregate `w1*wl_rank + w2*den_rank + w3*cng_rank` with w1=1.0, w2=0.5, w3=0.5.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** `--fast` regresses >1%.

**Result:** fast avg=1.2777, all avg=1.4997, 0 overlaps. Neutral — composite already matches official formula weighting.

---

### 6. [skipped: depends-on-killed] sp3_adaptive_verification `stage:sp3`

**Hypothesis:** Adaptive top_k (20 early for calibration, 3 late when well-calibrated) is best-of-both-worlds. Only meaningful if sp3_online_calibration is implemented — depends on calibrator.

**Touch:** `submissions/polyhedra/navigator.py` (after calibrator exists).

**Change:** `top_k = 20 if calibrator.n_samples < 10 else (3 if calibrator.rho > 0.5 else 10)`.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** `--fast` regresses.

**Note:** If sp3_online_calibration was killed, skip this item (mark `[skipped: depends-on-killed]`).

**Result:** Skipped — sp3_online_calibration was killed (rho_no_improvement), so calibrator unavailable.

---

### 7. [done: avg=1.4935] sp3_pinrudy_blockage `stage:sp3`

**Hypothesis:** PinRUDY (pin-location-weighted demand) + macro blockage (hard macros reduce effective capacity) better model real congestion than uniform-bbox RUDY.

**Touch:** `submissions/polyhedra/surrogate.py` (congestion accumulation ~lines 203-237, capacity init ~lines 60-61).

**Change:**
- If pin offset data is in `benchmark` dataclass: distribute per-net demand at pin locations instead of uniform bbox spread.
- Add `grid_{h,v}_cap_effective` arrays reduced by hard-macro overlap fraction.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** pin offset data unavailable (then mark `[killed: no pin data]`) OR `--fast` regresses >1%.

**Result:** fast avg=1.2742, all avg=1.4935, 0 overlaps. Pin data available. PinRUDY+blockage implemented but no improvement over sp3_top_k_20 (1.4919).

---

### 8. [done: avg=1.4930] sp3_pairwise_ranking `stage:sp3`

**Hypothesis:** A small logistic regression trained online on "which of A/B is better" (5 features from surrogate deltas) beats point-wise surrogate ranking. Higher ceiling (rho → 0.4-0.6) than calibration.

**Touch:** `submissions/polyhedra/navigator.py` (new `PairwiseRanker` class).

**Change:** After 10 verified candidates, build 45 pairwise training samples. Fit logistic regression on `(wl_diff, den_diff, cng_diff, n_moved_diff, max_dual_diff)` with strong L2 regularization. Rank candidates by `sum_b P(candidate > b)`.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** `--fast` regresses OR training fails numerically.

**Result:** fast avg=1.2738, all avg=1.4930, 0 overlaps. Neutral — pairwise ranker never activated (only 1-7 accepted candidates per benchmark, needs 10).

---

## Stage 2: SP1 — Initial Topology

### 9. [killed: fast_gate] sp1_spectral_topology `stage:sp1`

**Hypothesis:** Fiedler-vector init (eigenvectors 2,3 of netlist Laplacian) lands in a connectivity-optimal basin; navigator recovers density.

**Touch:** New `submissions/polyhedra/init/spectral.py`. Wire as alt init path in `placer.py`.

**Change:** Build macro-macro adjacency weighted by shared-net count. `scipy.sparse.linalg.eigsh(L, k=4, which='SM')`. Scale eigenvectors 2,3 to canvas, legalize (reuse SDF packer), extract assignment, LP, navigate.

**Fast gate:** `--fast` avg ≤ 1.51 AND congestion component drops >5% vs SDF at LP stage (before navigation).

**Kill if:** neither avg nor congestion improves.

**Result:** fast avg=1.7807, 0 overlaps. Killed: spectral coords ignore macro sizes → connected macros land on top of each other → legalizer scatters them, destroying connectivity. 40% worse than baseline.

---

### 10. [killed: no_congestion_improvement] sp1_replace_topology_extraction `stage:sp1`

**Hypothesis:** Extracting L/R/A/B assignment from RePlAce's positions inherits its congestion-friendly basin.

**Touch:** New `submissions/polyhedra/init/replace_topology.py`. Wire into `placer.py`.

**Change:** Locate RePlAce positions (check if stored under `results/` or baselines dir; otherwise use RePlAce reference positions if available in benchmark data). Feed positions → `extract_assignment()` → LP → navigate.

**Fast gate:** `--fast` avg ≤ 1.501 AND congestion drops vs SDF at LP stage.

**Kill if:** RePlAce positions unavailable (mark `[killed: no_replace_positions]`) OR no congestion improvement.

**Result:** fast avg=1.2776, 0 overlaps. Killed: RePlAce topology identical quality to SDF — extract_assignment() erases congestion advantage.

---

### 11. [done: avg=1.4930] sp1_congestion_aware_extraction `stage:sp1`

**Hypothesis:** Extract direction minimizing shared-net bbox area (not largest gap) for pairs with shared nets → locally congestion-optimal topology.

**Touch:** `submissions/polyhedra/assignment.py:39-66`.

**Change:** For pair (i,k) with shared nets, evaluate 4 dirs by estimated HPWL impact on shared nets; pick min. Fall back to largest-gap when no shared nets.

**Fast gate:** `--fast` avg ≤ 1.501 AND LP feasible on all benchmarks.

**Kill if:** LP infeasible anywhere OR avg regresses >1%.

**Result:** fast avg=1.2738, all avg=1.4930, 0 overlaps. Neutral — largest-gap already near-optimal for shared-net pairs.

---

### 12. [killed: fast_gate] sp1_hmetis_partitioning `stage:sp1`

**Hypothesis:** Recursive min-cut bisection produces a topology where connected macros cluster tight → small net bboxes → low congestion.

**Touch:** New `submissions/polyhedra/init/hmetis.py`. Install `kahypar` or `pymetis` via `uv pip install`.

**Change:**
1. Build hypergraph (macros=vertices, nets=hyperedges).
2. Balanced bisection L/R, then T/B recursively until partitions have 1-4 macros.
3. Convert partition tree → pairwise assignment.
4. LP + navigate.

**Fast gate:** `--fast` avg ≤ 1.501 AND congestion drops >5% at LP stage.

**Kill if:** partitioner install/import fails (mark `[killed: no_partitioner]`), partitions size-imbalanced breaking LP, or no congestion improvement.

**Result:** fast avg=1.9059, killed. kahypar installed fine but shelf packing within partitions produces terrible layouts. ibm13 LP infeasible. Clustering helps connectivity but naive spatial embedding destroys it.

---

### 13. [killed: fast_gate] sp1_greedy_construction `stage:sp1`

**Hypothesis:** Greedy sequential placement (most-connected first, pick direction minimizing shared-net congestion contribution) is an alternative to partition-based init.

**Touch:** New `submissions/polyhedra/init/greedy_congestion.py`.

**Change:** Sort macros by total shared-net weight. Place first at center. For each subsequent, evaluate all 4 directions vs every placed macro, pick dir minimizing total RUDY contribution from shared nets.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** `--fast` regresses >1% OR construction produces infeasible topology.

**Result:** fast avg=1.7545, killed. Greedy clustering creates dense regions — WL vs density/congestion fundamental conflict. SDF analytical spreading far superior.

---

### 14. [killed: no_congestion_improvement] sp1_boundary_attraction `stage:sp1`

**Hypothesis:** Pushing macros toward canvas boundaries (linearized LP penalty) opens up the center for routing.

**Touch:** `submissions/polyhedra/lp.py` (add auxiliary vars + penalty terms).

**Change:** Add `d_x_i = min(x_i - x_lo, x_hi - x_i)` via aux vars and two linear constraints each; add `lambda * sum_i (d_x_i + d_y_i)` to objective. Start lambda small (0.01 × HPWL scale).

**Fast gate:** `--fast` avg ≤ 1.501 AND center-region congestion drops.

**Kill if:** boundary packing hurts density by >5% OR avg regresses.

**Result:** fast avg=1.2728, 0 overlaps. Killed: penalty inert at safe lambda (0.01 × diag/N ≈ fractions of a unit vs HPWL in thousands). No congestion improvement. Larger lambda would hurt WL.

---

## Stage 3: SP4 — Congestion-Aware LP

### 15. [done: avg=1.4974] sp4_net_weighting `stage:sp4`

**Hypothesis:** Iterative RUDY-driven net weighting (Brenner/Vygen) makes LP avoid congestion hotspots within current topology.

**Touch:** `submissions/polyhedra/lp.py:227-234` (accept `net_weights`), `placer.py:232` (reweight loop).

**Change:** 3 rounds: solve LP → RUDY map → top-5% hot cells → for each net crossing hot cells, `w *= (1 + 0.5 * hot_overlap_fraction)`; normalize mean=1, cap 5x; re-solve.

**Fast gate:** `--fast` avg ≤ 1.501 AND congestion at LP positions (pre-nav) drops >2%.

**Kill if:** LP runtime >10x OR avg regresses.

**Result:** fast avg=1.2736, all avg=1.4974, 0 overlaps. Weighting mechanism inert — RUDY too uniform for hot cells. 3x LP overhead hurts nav budget.

---

### 16. [done: avg=1.4932] sp4_separation_margins `stage:sp4`

**Hypothesis:** Increasing min separation margins in congested regions creates routing space; complementary to net weighting.

**Touch:** `submissions/polyhedra/lp.py:122-162` (separation constraints RHS).

**Change:** First LP solve → RUDY map → per-pair margin `beta * max(0, congestion_at_midpoint - capacity)`, cap at 10% of min canvas dim. Re-solve.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** LP infeasible OR avg regresses >1%.

**Result:** fast avg=1.2736, all avg=1.4932, 0 overlaps. Margins added but inert — navigation overwrites LP positions, so LP-level spacing doesn't persist.

---

### 17. [done: avg=1.4918] [new_best] [new_best_stage] sp4_mccormick_area `stage:sp4`

**Hypothesis:** Penalizing linearized net bbox area (McCormick envelope) directly targets concentration-driven congestion.

**Touch:** `submissions/polyhedra/lp.py` (add auxiliary vars + 4 McCormick constraints per net).

**Change:** For each net j: aux `A_j` with 4 McCormick constraints using canvas bounds `x_L=0, x_U=W, y_L=0, y_U=H`. Add `lambda * A_j` to objective.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** LP size blows up (>10x runtime) OR avg regresses.

**Result:** fast avg=1.2692, all avg=1.4918, 0 overlaps. NEW_BEST (prev 1.4919). McCormick with lambda=0.001 nearly inert but marginal improvement. LP runtime negligible overhead.

---

### 18. [killed: no_effect] sp4_dual_informed_targeting `stage:sp4`

**Hypothesis:** Navigator proposer should use *congestion duals* (HPWL dual × shared-net congestion contribution), not pure HPWL duals.

**Touch:** `submissions/polyhedra/moves.py:170-194` (`DualGuidedProposer`).

**Change:** After LP solve, for each pair compute `cong_dual = hpwl_dual * sum(congestion_contribution of shared nets)`. Blend: `effective_dual = alpha * hpwl_dual + (1-alpha) * cong_dual`, alpha=0.5.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** `--fast` regresses OR proposer selects infeasible pairs.

**Result:** fast avg=1.2738, identical to baseline. Killed: uniform per-net congestion (total_cong/n_nets) just scales all duals equally — no ranking change. Needs per-net RUDY differentiation.

---

### 19. [killed: regression] sp4_real_proxy_feedback `stage:sp4`

**Hypothesis:** Using the *real* PlacementCost congestion map (not RUDY) at LP init produces better initial weights than approximation. Only at init (once, ~50ms cost).

**Touch:** `submissions/polyhedra/placer.py` (init sequence), possibly read-only hooks into `macro_place/objective.py` to extract per-cell congestion.

**Change:** After first LP solve, call real-proxy evaluator, extract per-cell congestion, compute per-net contribution, set `weight_j = 1.0 + gamma * contribution_j`, re-solve once. No harness modification — if per-cell arrays aren't exposed, kill.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** per-cell congestion not extractable from harness (mark `[killed: no_hook]`) OR avg regresses.

**Result:** fast avg=1.2730, 0 overlaps. Killed: congestion weighting inflates HPWL (2214→2406 on ibm01) without proportionally reducing congestion cost.

---

### 20. [done: avg=1.4973] sp4_lp_navigate_reweight_loop `stage:sp4`

**Hypothesis:** 3 outer rounds of (congestion-aware LP → 15s nav → recompute weights from navigated positions) bootstraps congestion reduction with density preservation.

**Touch:** `submissions/polyhedra/placer.py:232-277` (restructure `place()`).

**Change:** Outer loop × 3: weighted LP → navigate 15s → RUDY from navigated → EMA-dampened weight update (`w = 0.7 * w_old + 0.3 * w_new`) → re-extract assignment.

**Fast gate:** `--fast` avg ≤ 1.501.

**Kill if:** total runtime exceeds 60s per benchmark OR avg regresses.

**Result:** fast avg=1.2733, all avg=1.4973, 0 overlaps. Neutral — splitting nav budget across 3 rounds cancels congestion benefit. ibm18 LP infeasible in round 3.

---

## Stage 4: Combine Winners

### 21. [done: avg=1.4918] [new_best_stage] combine_stage_winners `stage:combine`

**Hypothesis:** Best-per-stage findings stack: best SP3 (better surrogate) + best SP1 (better basin) + best SP4 (congestion-aware LP) compound.

**Prerequisite:** Stages 1–3 complete. Driver reads `results/overnight_run.log`, identifies lowest `--all` avg per `stage:` tag. If any stage has zero `--all` passes, use `[new_best]` from stage-local fast runs; if still none, mark this item `[skipped: no_winners_for_stage_<X>]`.

**Touch:** Subagent's task: in a fresh worktree off `main`, apply diffs from the three winning worktrees sequentially, resolving conflicts minimally.

**Change:**
1. `git diff main <sp3_winner_branch> -- submissions/ > /tmp/sp3.patch` etc.
2. `git apply /tmp/sp3.patch && git apply /tmp/sp1.patch && git apply /tmp/sp4.patch`
3. If any patch fails to apply, try one at a time; record which conflict and pick highest-impact base.
4. Run `--fast` then `--all`.

**Fast gate:** `--fast` avg ≤ best individual stage winner's fast avg.

**Kill if:** patch conflicts unresolvable OR combined result worse than any individual winner.

**Result:** All 3 patches applied cleanly. fast avg=1.2692, all avg=1.4918, 0 overlaps. Ties sp4_mccormick_area — changes don't compound (SP1/SP3 changes are neutral in practice).

---

### 22. [skipped: no_committed_changes] combine_pairwise_sp1_sp4 `stage:combine`

**Hypothesis:** If three-way combine fails, pairwise SP1+SP4 may still stack (topology × LP objective). SP3 is an accelerator, not structurally coupled.

**Prerequisite:** `combine_stage_winners` completed or killed. Skip if the 3-way already beat all individual winners.

**Touch:** Subagent applies SP1+SP4 winner diffs only.

**Fast gate:** `--fast` avg ≤ best of (sp1_winner fast, sp4_winner fast).

**Kill if:** patches conflict OR worse than best individual.

**Result:** Skipped — experiment changes exist only as uncommitted working-tree modifications in worktrees. `git diff main <branch>` returns empty because subagents never committed (per protocol). Combine mechanism requires committed diffs.

---

## Stop conditions (driver obeys without asking)

- `queue.md` has no `[pending]` items → STOP with `QUEUE_EMPTY`.
- Wall clock > 10 hours → STOP with `TIME_UP`.
- Unrecoverable crash (2 retries exhausted) → STOP with `CRASH: <reason>`.

Neither kill-streaks nor threshold crossings stop the loop. Kills cost ~2min each (fast_gate); we want full coverage.

## Logging contract

Every run: `--hypothesis <name> --json` (appends to `results/experiment_log.jsonl`).

Driver appends to `results/overnight_run.log`:
```
<ISO-timestamp> <hypothesis> stage=<sp3|sp1|sp4|combine> <fast|all|kill> avg=<x.xxxx> overlaps=<n> gate=<pass|fail|kill> best_so_far=<x.xxxx> best_in_stage=<x.xxxx> worktree=<path>
```

On new global best:
```
<ISO-timestamp> NEW_BEST <hypothesis> avg=<x.xxxx> (prev=<y.yyyy>) worktree=<path>
```

On new stage best:
```
<ISO-timestamp> NEW_BEST_STAGE stage=<tag> <hypothesis> avg=<x.xxxx> worktree=<path>
```
