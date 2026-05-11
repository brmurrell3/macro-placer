---
id: E67
name: kjoint_hungarian
status: marginal
parent: E48
created: 2026-05-03
decided: 2026-05-05
champion_at_time: 1.08151
outcome: **ABANDONED 2026-05-05** (parallel agent ended). K=50 multi-step Hungarian re-pack. Smokes on cached ibm01/04/09/12 measured −0.11 % aggregate --fast lift over E48-equivalent (ibm04 −0.21 %, ibm09 −0.14 %, ibm12 −0.09 %, ibm01 −0.00 %). K=50 beats K=20 ablation by ~2× (−0.11 % vs −0.05 %). Saturation (rejection streak ≥ 15) consistently terminates ~30–40 clusters before budget. Integration scaffolded into E41 lane via E70. Never run on --all or --ng45. Status frozen at last known state. **Now superseded for the champion path by E74 (1.0666, ADR-012, 2026-05-05).**
champion_delta: -0.0011 (-0.11%) on --fast subset only; never aggregated to --all
graduated_to: null
superseded_by: E74 (CDLNSSAHessian, ADR-012)
---

# E67: kjoint_hungarian

## Hypothesis

K=50 Hungarian re-pack escapes the E48 plateau by joint-reassigning a
netlist-coupled cluster of 50 macros to optimal slots from a grid-based
candidate pool. The polynomial Hungarian assignment
(`scipy.optimize.linear_sum_assignment`) makes K=50 tractable where E41's
brute-force product was capped at K=3 (5^3 = 125 combos). At K=50 with
n_slots=100, Hungarian is O(K^2 * n_slots) ≈ 250k cell evaluations per solve;
with the incremental evaluator at ~10–100 μs per single-macro proxy delta,
each Hungarian step targets ~100 ms once the cost matrix is built (the cost
matrix itself is the bulk: K × n_slots ≈ 5 000 incremental delta probes,
target ~1–2 s). Hundreds of K=50 moves per hour are feasible inside the 600 s
post-E48 polish budget.

The mechanism directly attacks Wall 3 (local-move saturation): E41 K-joint
in lane 2 commits 12–58 K-tuples per bench then saturates because the
3-coupled neighbourhood is exhausted. Extending K from 3 to 50 enlarges the
joint-move scope by an order of magnitude so the post-E41 fixed point should
not be a fixed point under K=50 moves. It also has a chance at Wall 1
(infeasibility wall between E25 and E41 basins, per E65 NEB cross-section)
*if* the 50-cluster spans both basins; but the primary claim is plateau
escape, not basin bridging.

## Method

Module exposes a single step `kjoint_hungarian_step(benchmark, placement,
k=50, n_slots=100)`:

1. **Cluster select.** From the worst-cost coupled region, pick `k` hard
   movable macros by netlist-adjacency score (E39 `_adjacency_scores`,
   reused). LP-dual fallback documented but not implemented this scaffold —
   adjacency is benchmark-blind and is the recommended default.
2. **Slot enumerate.** Generate `n_slots` candidate (cx, cy) centers via
   grid-bin enumeration restricted to the cluster's axis-aligned bounding
   box (E12 `_gridbin_reinsert`-style grid with canvas-bound clamping);
   include all current cluster positions as fallbacks.
3. **Cost-matrix build.** For each (macro_i, slot_j) pair, use the
   incremental evaluator: move macro_i to slot_j, read proxy, revert. Store
   delta = proxy_after - baseline. Macros that cannot legally occupy a slot
   (overlap with the *non-cluster* background) get cost = +inf. Pairwise
   conflicts among the cluster are handled in commit, not the cost matrix.
4. **Solve Hungarian.** `scipy.optimize.linear_sum_assignment` on the
   `k × n_slots` rectangular cost matrix; returns one slot per macro,
   minimizing the linear sum of deltas (independent moves; ignores K-fold
   pairwise interaction, which the post-commit overlap check repairs).
5. **Commit & defensive revert.** Apply all `k` moves to the evaluator.
   If `compute_overlap_metrics` reports any non-zero overlap OR proxy did
   not improve vs baseline, revert all `k` moves and return the original
   placement. Otherwise commit.

Drop-in replaceable for E41's K-joint K=3 phase as the post-saturation
move generator; integration into the E48 hybrid is left for the next
session — this scaffold ships the module + smoke only.

## Kill gate

* `--fast` avg_proxy > 0.925 (E48 0.92024 + 0.5 %).
* `--ng45` ariane133 > 0.694.
* If a single Hungarian step takes > 10 s on ibm01 SDF init, the per-bench
  budget makes the move generator infeasible — kill before running --all.

## Generalization check

Cluster selection by netlist adjacency (E39's reusable `_adjacency_scores`)
is benchmark-blind: it weighs nets by 1 / (|net| - 1), the same weighting
SDF and the proxy use. If --all reveals an IBM-vs-NG45 imbalance, fall back
to LP-dual significance (deferred — E64 LP infrastructure is currently
deleted; would need to be reinstated). NG45 ariane133 is the canary as
usual.

## Outcome

**Status 2026-05-04 01:0X:** scaffold + V2 fix + plateau-lift +
multi-step + K-size sensitivity all verified. Integration into E48
hybrid is next.

### Timeline

* **2026-05-03 18:0X — scaffold + V1 smoke** (`smoke.py`). Joint commit
  on ibm01 SDF init: 8 jointly-applied moves → 16 overlaps → defensive
  revert → no improvement.

* **2026-05-03 21:0X — V2 sequential-commit fix** (`smoke_v2.py`).
  Same input: 8 pending → 2 committed (6 skipped by per-move legality)
  → 0 overlaps → proxy 1.195324 → 1.182850 (−1.04 %), accepted.
  `commit_if_better_sequential` resolves Risk b (pairwise blindness);
  `kjoint_hungarian_step(..., commit_mode='sequential')` is the
  default. V1 preserved as `commit_mode='joint'`.

* **2026-05-03 23:5X — V2 plateau-lift one-shot** (`smoke_plateau.py`).
  Used cached E25 / E41 placements at
  `experiments/E69_sequence_pair_search/results/placements/`. Single
  V2 step per cached placement; 5/8 accepts, avg lift on accepts
  −0.16 %. Critical observation: lift is on E25 lane (E48-irrelevant
  on the 4 fast benches, where E41 wins 3/4); E41 lane has ~50 / 50
  no-ops (cluster already polished by K-joint K=3).

* **2026-05-04 00:0X — multi-step with cluster diversification**
  (`smoke_multistep.py`). Added `cluster_seed: Optional[int]` to
  `select_cluster` (random sample of K from top-(K · pool_multiplier)
  by adjacency, RNG-seeded). N=50 steps with seeds 0..49 on the
  *winning lane* per bench:

  | bench | win lane | proxy_in | proxy_out | Δ | Δ% | accepts |
  |-------|---|----:|----:|----:|----:|----:|
  | ibm01 | E25 | 0.89464 | 0.89463 | -0.00001 | -0.00 % | 1 |
  | ibm04 | E41 | 1.01526 | 1.01313 | -0.00213 | **-0.21 %** | 2 |
  | ibm09 | E41 | 0.83298 | 0.83157 | -0.00141 | **-0.14 %** | 7 |
  | ibm12 | E41 | 1.20866 | 1.20779 | -0.00087 | **-0.09 %** | 5 |

  Aggregate -0.110 %; verdict by rule MARGINAL. But **all three
  E41-winning benches show meaningful lift** (-0.09 % to -0.21 %); the
  aggregate is diluted by ibm01 E25 (no slack). Saturation, not budget,
  ends every loop (15-rejection early stop fires at steps 24-40).

* **2026-05-04 01:0X — K=20 ablation** (`smoke_multistep_k20.py`).
  Same setup with K=20 instead of 50; aggregate -0.053 % (~2× worse
  than K=50). K=20 finds same accept count but each accept is smaller;
  cluster too small to span coupled regions. **K=50 is the right
  size for integration.**

### Net E48-hybrid impact estimate

E41 wins 12 of 17 IBM benchmarks. If the fast-subset E41 lift (avg
-0.15 % across the 3 E41-winners ibm04 / 09 / 12) generalizes to all
12 E41-winners, the E48 average drops by ~0.10 % (12 / 17 · -0.15 %),
taking 1.08151 → ~1.0805. Real but small. Integration cost is low
(one new phase appended to the E41 pipeline, ~1 day of code +
verification).

### Next session's first task

Build `experiments/E70_kjoint_hungarian_integrated/` (numbering shifted
from prior plan because E69 sequence-pair has been claimed). The
variant adds a 5th phase to the E41 lane only:

```
DPO init → CD → LNS → SA-v2 → K-joint K=3 → K-joint Hungarian K=50-multi → validate
```

Multi-step driver: loop with `cluster_seed = 0..N`, accept the new
placement on each accepted step, early-stop on 15 consecutive
rejections OR 600 s budget. E25 lane unchanged. Hybrid wrapper picks
min(E25, E41-with-new-phase).

Validation: --fast → --ng45 → --all in that order. Kill gate same as
E68 (--fast > 0.92024 + 0.5 %, --ng45 > 0.6922).

## Pointers

* Code: `code/kjoint_hungarian.py` (the move module), `code/smoke.py`
  (one-step ibm01 correctness check).
* Reused from E39: `_adjacency_scores`, `_is_legal_2d_excluded`,
  `_ktuples_pairwise_legal` (`experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py:447,103,468`).
* Reused from E12: grid-bin slot enumeration pattern
  (`submissions/cd_lns_gridbin/placer.py:134-180`,
  `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py:147-200`).
* Research source: `docs/research_principles_for_walls.md` lines 522–574
  (§3 Rank 1).
* Related findings: `e65_infeasibility_wall.md` (memory) — basins separated
  by infeasibility wall, motivates non-local feasibility-respecting moves.
