# E67 engineering notes — K=50 Hungarian re-pack scaffold

## Reused components (file:line)

* **Cluster selection — `_adjacency_scores`**:
  `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py:447-465`.
  Per-macro adjacency = sum_{n ∈ macro_to_nets[m]} 1 / max(1, |net_n| - 1).
  Reused verbatim into `kjoint_hungarian.py:_adjacency_scores`. Aligned with
  the `hard_movable` order; benchmark-blind (netlist-only weights, same as
  HPWL and SDF).

* **Legality check — `_is_legal_2d_excluded`**:
  `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py:103-144`.
  Reused verbatim. eps=1e-9 STRICT separation per the E39 ibm07 fix
  (`docs/gotchas.md` / memory `kjoint_overlap_eps_gotcha.md`). Excludes
  cluster siblings as blockers since they're being reassigned in this step.

* **Grid-bin slot generation pattern — `_gridbin_reinsert`**:
  - Original: `submissions/cd_lns_gridbin/placer.py:134-180`
    (E12 champion's grid_col × grid_row enumeration with canvas clamp).
  - E39 adaptation: `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py:147-200`
    (same pattern with the excluded-blocker variant).

  Adapted (not verbatim) into `kjoint_hungarian.py:generate_slots`. Differences:
  (1) restricted to the cluster bbox + 10 % padding instead of the full
  canvas; (2) seeded with current cluster centers as fallbacks; (3)
  full-canvas fallback if the bbox produces < n_slots viable bins.

## Decisions

### Adjacency vs LP-dual

Used adjacency (`mode='adjacency'`). LP-dual significance is documented in
the manifest as a fallback if --all reveals an IBM-vs-NG45 imbalance, but
implementing it requires the polyhedra LP infrastructure deleted in commit
44efd16 (see `e63_e64_blocked.md`); roughly 2-3 days dev. Adjacency is the
default because it's benchmark-blind (netlist-only weights, identical
math to the HPWL net weighting in SDF and the proxy) and ships today.

### Fixed-macro handling in slot generation

Slot generation is fixed-macro-blind on purpose: `select_cluster` returns
hard *movable* indices only, and `build_cost_matrix` uses
`_is_legal_2d_excluded` against the *non-cluster* background — which
includes every fixed macro at its current position. So a slot landing on
a fixed macro auto-falls to cost = +inf for any cluster macro that would
overlap it. No special case needed.

### Incremental evaluator vs full proxy recompute

Used incremental evaluator deltas (`evaluator.move(m, slot)` →
`current_cost()` → `revert()`) for every (macro_i, slot_j) cell. The
alternative — full proxy recompute per cell — was ruled out by simple
arithmetic: K=50 × n_slots=100 = 5000 cells × ~10-50 ms full proxy = 50-
250 s per Hungarian step, prohibitive. The incremental path measured at
~1.4 ms per cell on ibm01 (5000 cells in 7.24 s; see smoke output below).

Single-step revert is fine here because every cell is `move; cost; revert`
in immediate sequence — there's no second move between the revert and the
snapshot getting overwritten. (`docs/gotchas.md` single-step revert
gotcha applies only when consumers chain moves.)

### Cost matrix sentinel for all-+inf rows

`solve_hungarian` zeros rows that are entirely +inf so scipy doesn't
crash, then replaces remaining +inf cells with `1e6 * max_finite` so the
linear sum stays well-conditioned. The post-commit overlap check is the
real gate; the Hungarian is a fast first pass.

### Defensive revert via `compute_overlap_metrics`

`commit_if_better` is the safety net. The Hungarian's linear cost ignores
pairwise interactions among the K cluster macros — independently-optimal
slots can collide. The smoke confirmed this (16 post-commit overlaps from
8 moves of 50; all reverted cleanly). Future work: pre-filter the cost
matrix with pairwise constraints, or do sequential Hungarian-with-
revalidation, or accept a few clusters worth of failed steps and budget
accordingly.

## Smoke output (ibm01, K=50, n_slots=100)

```
[load] ibm01: num_macros=1140, n_hard=246, canvas=22.9x23.0, wall=0.40s
[sdf_init] wall=3.55s, project_iters=0, residual_overlaps=0
[eval_init] wall=0.28s, proxy=1.19532 [wl=0.0731 d=0.9426 c=1.3019]
[hungarian] running one K=50, n_slots=100 step ...
[result] per-substep wall:
  t_eval_init_s            = 0.0000 s   (passed pre-built evaluator)
  t_cluster_select_s       = 0.0010 s
  t_slots_gen_s            = 0.0003 s
  t_cost_matrix_s          = 7.2423 s   ← bottleneck (5000 cells × ~1.4 ms)
  t_hungarian_s            = 0.0001 s   ← polynomial scaling confirmed
  t_commit_s               = 0.0595 s
  t_total_s                = 7.3033 s
[result] sizes + content:
  k_actual                = 50
  n_slots_actual          = 100
  cost_finite_cells/total = 2638 / 5000 (52.8% finite)
  cost_min, cost_max      = -8.6654e-03, 1.7736e-02
  proxy_before / after    = 1.19532 / 1.20167
  commit overlap_count    = 16  (4.166 area)
  accepted                = False
  reason                  = overlap
  n_moved                 = 8
[verify] shape=(1140, 2) OK; final overlap_count=0, area=0.000000e+00
[smoke] total wall = 11.59 s — OK
```

## Substep wall analysis vs predictions

| Substep            | Predicted     | Measured | Notes                     |
|--------------------|---------------|----------|---------------------------|
| eval_init          | ~0.5 s        | 0.28 s   | OK; one-time init         |
| cluster_select     | < 1 ms        | 1.0 ms   | 246 hard movables; trivial |
| slots_gen          | ~10 ms        | 0.3 ms   | bbox restricts walk; fast |
| cost_matrix        | ~1-2 s        | 7.24 s   | **3-7× over** target      |
| hungarian          | ~100 ms       | 0.1 ms   | **1000× under** target    |
| commit + ov check  | ~50 ms        | 60 ms    | OK                        |

The Hungarian solve is far faster than the research-doc estimate (the
"~100 ms" budget assumed K=50 brute-force-style; scipy's
`linear_sum_assignment` on a 50×100 matrix is genuinely μs-scale). The
cost-matrix build is the real bottleneck — 5000 incremental-evaluator
probes at ~1.4 ms each. Three observations:

1. **It's still in budget.** 600 s post-E48 polish ÷ 7 s/step = ~85 steps.
2. **It scales sub-linearly with skip-rate.** 47 % of cells are infeasible
   (skipped before the move/revert); a tighter slot pre-filter (e.g.
   "must be inside the cluster macro's legal axis range") could cut the
   probe count further.
3. **Hungarian's polynomial guarantee is intact** even if cost-matrix
   build dominates — bumping K to 100 or n_slots to 200 raises the
   matrix build linearly, not the solve itself.

## V2: sequential commit (pairwise-blindness fix) — 2026-05-03

Added `commit_if_better_sequential` (option (b) from the bottlenecks
list, the cheapest of the three). Algorithm:

1. Sort assignment by predicted improvement (most-negative cost first),
   filtered to non-no-op moves.
2. For each move in priority order, check `_is_legal_2d_excluded(...,
   excluded=[])` against the *committed* state. The committed state
   includes already-moved cluster siblings at their new positions, so
   collisions with them are caught here without any per-cell pairwise
   computation.
3. Skip illegal moves; apply legal ones via `evaluator.move()`.
4. After all moves: full `compute_overlap_metrics` + cumulative-proxy
   check; revert the entire batch if either fails. (With per-move
   legality, overlap should be 0 by construction; the revert path mostly
   guards against cumulative-proxy regression from cost-matrix drift.)

`kjoint_hungarian_step` now takes `commit_mode='sequential' | 'joint'`,
default `'sequential'`. V1's `commit_if_better` is preserved for
diagnostics.

### V2 smoke result (`smoke_v2.py`, ibm01 SDF init)

| metric           | joint (V1)  | sequential (V2) |
|------------------|-------------|------------------|
| accepted         | False       | **True**         |
| reason           | overlap     | ok               |
| n_pending        | —           | 8                |
| n_moved          | 8           | 2                |
| n_skipped_illegal| —           | 6                |
| n_no_op          | —           | 42               |
| overlap_count    | 16          | 0                |
| proxy_before     | 1.195324    | 1.195324         |
| proxy_after      | 1.201669    | **1.182850**     |
| t_total          | 7.33 s      | 7.25 s           |

**V2 accepts 2/8 movable assignments and improves proxy by −0.0125
(−1.04 %)** on a single step. The 6 skips are exactly the pairwise
collisions V1 produced as overlaps. Wall time is unchanged (cost-matrix
build dominates, identical work for both modes).

CAVEAT: this measures lift on a *non-converged* SDF init (proxy 1.195),
not on a polished E48 plateau (ibm01 final proxy ~0.892). Lift on the
plateau is the integration test, deferred to next session — needs a
saved post-E48 placement, which requires a re-run since
`experiment_log.jsonl` doesn't store placement tensors.

## V2 plateau-lift result — 2026-05-03 23:5X (`smoke_plateau.py`)

Used the cached E25/E41 lane outputs at
`experiments/E69_sequence_pair_search/results/placements/` (E69 logged
them at 21:23 → 22:50). One V2 sequential-commit step per cached
placement, K=50, n_slots=100.

| bench | lane | proxy_in | proxy_out | Δ_full | Δ% | moved/pending | accepted |
|-------|------|---------:|----------:|-------:|----:|---------------|---------|
| ibm01 | E25  | 0.89464  | 0.89463   | -0.0000 | -0.00% | 1/1  | True  |
| ibm04 | E25  | 1.03361  | 1.03045   | -0.0032 | -0.32% | 5/6  | True  |
| ibm09 | E25  | 0.86655  | 0.86392   | -0.0026 | -0.26% | 6/14 | True  |
| ibm12 | E25  | 1.21304  | 1.21123   | -0.0018 | -0.18% | 3/3  | True  |
| ibm01 | E41  | 0.92067  | 0.92067   | +0.0000 | +0.00% | 0/0  | False (zero_moved) |
| ibm04 | E41  | 1.01526  | 1.01526   | +0.0000 | +0.00% | 1/3  | False (no_proxy_improvement) |
| ibm09 | E41  | 0.83298  | 0.83298   | +0.0000 | +0.00% | 0/0  | False (zero_moved) |
| ibm12 | E41  | 1.20866  | 1.20831   | -0.0004 | -0.04% | 2/2  | True  |

**Aggregate:** 5/8 accepts, avg Δ on accepts = -0.16 %; total wall 215 s
(~25 s/bench-lane).

### Critical observation: lift is on the LOSING lane

For each bench, E48 picks `min(E25, E41)`. Per-bench winners on the
fast subset:
- ibm01: E25 wins (0.895 < 0.921). K=50 lift on E25 = trivial −0.00%.
- ibm04: E41 wins (1.015 < 1.034). K=50 lift on E25 = −0.32% (irrelevant
         to E48); E41 lift = 0.
- ibm09: E41 wins (0.833 < 0.867). K=50 lift on E25 = −0.26% (irrelevant);
         E41 lift = 0.
- ibm12: E41 wins (1.209 < 1.213). K=50 lift on E25 = −0.18% (irrelevant);
         E41 lift = −0.04%.

**Net effective E48 lift from one K=50 step:** roughly zero on the fast
subset. Mechanism explanation: the E41 lane already runs K-joint K=3
which exhausts local 3-tuple moves. By the time E41 hands off, n_no_op
in the K=50 Hungarian step is 47–50 / 50 — the cluster picked by
top-adjacency is already locally-optimal at the K=3 frontier. E25
(no K-joint) leaves more 3-tuple slack and so K=50 finds room, but
those benches are picked from E41 anyway.

This is **not** the "no lift, kill" verdict. The pairwise-blindness fix
clearly works (V2 accepts what V1 rejected). What the one-shot test
shows is that the *cluster the heuristic picks* is already polished by
E41's existing K-joint, and the ONE step we tried doesn't move past it.
The integration question is whether **multi-step + cluster
diversification** can find lift on the winning lane.

## V2 multi-step plateau-lift result — 2026-05-04 00:1X (`smoke_multistep.py`)

Added `cluster_seed: Optional[int]` to `select_cluster` (random sample
of K from top-(K * pool_multiplier) by adjacency, RNG seeded by
cluster_seed). Threaded through `kjoint_hungarian_step`. Ran N=50 steps
with seeds 0..49 on the *winning lane* per bench (E48 hybrid picks
min(E25, E41); only the winner can affect E48 score):

| bench | winning lane | proxy_in | proxy_out | Δ | Δ% | accepts | moves | wall |
|-------|---|----:|----:|----:|----:|----:|----:|----:|
| ibm01 | E25 | 0.89464 | 0.89463 | -0.000009 | -0.00 % | 1   | 1  | 118 s |
| ibm04 | E41 | 1.01526 | 1.01313 | -0.002131 | **-0.21 %** | 2 | 3  | 106 s |
| ibm09 | E41 | 0.83298 | 0.83157 | -0.001411 | **-0.14 %** | 7 | 12 | 242 s |
| ibm12 | E41 | 1.20866 | 1.20779 | -0.000868 | -0.09 % | 5   | 7  | 501 s |

**Aggregate:** avg Δ = -0.110 %; verdict by rule = MARGINAL.

### Three findings worth integrating

1. **Cluster diversification works.** All 4 benches ran 24-40 steps
   before hitting the 15-consecutive-rejection early stop; ibm09 had
   accepts at steps 1, 3, 4, 5, 13, 19, 25 — staircase pattern that
   the deterministic top-50 cannot produce. The seed-driven random
   sample from the top-3 K clearly finds new clusters.

2. **Lift is concentrated on the E41 lane (where E48 picks).** ibm04,
   ibm09, ibm12 are the three E41-winners in the fast subset, and all
   three show meaningful lift (-0.09 % to -0.21 %). ibm01 (E25 winner,
   the lane that E41's K-joint has not polished) shows trivial lift —
   the cluster heuristic finds little slack because the top-adjacency
   macros are already optimized by SA after CD.

3. **Saturation, not budget exhaustion, ends the loop.** All four
   benches hit the rejection-streak early-stop, not the 50-step
   budget. Per-step wall is 3-10 s; at 600 s budget we'd run 60-200
   steps but we never need them. Saturation is the active constraint
   — the cluster heuristic exhausts its useful candidates after
   ~30-40 random clusters from the top-3 K pool.

### What the MARGINAL aggregate masks

The -0.11 % aggregate is misleading because ibm01 dilutes it to zero
while three benches show real lift. A more useful framing: among
**E41-winning benches** (where K=50 Hungarian polish is integrated),
the average lift is **-0.15 %**. Extrapolated to the 12 E41-winners
out of 17 IBM benches, that's ~-0.10 % on the E48 average — taking
1.08151 to ~1.0805. That's a real but small win; well within the
"useful for May 21" range.

### Integration sketch (next session)

The natural integration is a 5th phase in the **E41 lane only**
(E25 lane sees no lift on its own winners):

```
E41 lane: DPO init → CD → LNS → SA-v2 → K-joint K=3 → K-joint Hungarian K=50-multi → validate
```

Budget: share the existing K-joint 600 s with the new phase, or
allocate a fresh 600 s. Multi-step driver pseudocode:

```
for seed in range(N_max):
    new_pl, info = kjoint_hungarian_step(..., cluster_seed=seed)
    if info["accepted"]: placement = new_pl
    rejection_streak = 0 if accepted else rejection_streak + 1
    if rejection_streak >= 15 or wall > budget: break
```

ibm12 is the slowest (~9 s/step → 27 steps in 240 s). For larger
benches in the --all set (ibm17, ibm18 — both ~3 K hard movables),
each step would be ~15-25 s; saturation should still come before 600 s.

### Open questions

1. **K=20 falsified — K=50 is the right size.** `smoke_multistep_k20.py`
   completed; results vs K=50:

   | bench | K=50 Δ% | K=20 Δ% | K=50 wall | K=20 wall |
   |-------|---:|---:|---:|---:|
   | ibm01 | -0.00 | -0.00 | 118 s | 20 s |
   | ibm04 | -0.21 | -0.15 | 106 s | 50 s |
   | ibm09 | -0.14 | -0.03 | 242 s | 94 s |
   | ibm12 | -0.09 | -0.04 | 501 s | 261 s |

   K=20 is faster per step (smaller cost matrix) but finds substantially
   less lift per cluster — same accept count, smaller deltas per
   accept. Aggregate K=20 = -0.053 % vs K=50 = -0.110 %; K=50 wins
   ~2× on aggregate. Mechanism: a 50-macro coupled cluster spans more
   pairwise net interactions, so the joint reassignment has more
   degrees of freedom to find a meaningful improvement; K=20 leaves
   inter-macro coupling on the table.

   **Decision:** keep K=50 as the integration default.
2. **Worst-cost-cell heuristic.** Adjacency picks high-degree macros
   that CD/SA already weights heavily. An alternative is "macros
   contributing the most to current proxy" — likely produces a different
   cluster topology and may find lift the adjacency heuristic can't
   reach.
3. **--all and --ng45 generalization.** Plateau-lift results so far
   are on the IBM fast subset only. The cached placements at
   `experiments/E69_sequence_pair_search/results/placements/` cover
   only ibm01/04/09/12. Generalization tests need either rerun caches
   or integrating into E48 first and running --all directly.
