# Navigator Evaluation Pipeline

Last updated: 2026-04-23

---

## Overview

The navigator evaluates candidate moves through a two-stage pipeline:

```
proposed moves → [Screen] → [Evaluate] → acceptance decision
                  ~1-10us     ~0.1ms
                  prunes       scores
                  ~20-40%      survivors
```

Stage 1 (ClusterScreener) is implemented. Stage 2 currently uses GridSurrogate
(rho=0.17 within-benchmark); an exact incremental evaluator is proposed.

---

## Stage 1: ClusterScreener (implemented)

Multi-tier pre-projection pruning. Rejects obviously bad moves *before* the
expensive projection + surrogate pipeline (~0.5ms+).

**Files:** `submissions/polyhedra/cluster_bounds.py`, integrated in `navigator.py`.

### Screening tiers

Each tier is O(k) or O(k * avg_nets_per_macro) where k = flip count (1-5).

| Tier | Screen | Cost | What it catches |
|------|--------|------|-----------------|
| 0 | **Zobrist hash** | O(k), ns | Duplicate topologies and no-op flips |
| 1 | **Self-contradiction** | O(k^2), ns | Multi-pair moves with conflicting directions |
| 2 | **Displacement floor** | O(k), us | Moves requiring impossible position changes. Weighted by net connectivity; threshold at normalized score > 50 |
| 3 | **Net-span HPWL bound** | O(k * shared_nets), us | Same-axis reversals that provably worsen wirelength. Prunes when HPWL floor > 3% of proxy cost |
| 4 | **Axis crowding** | O(k), us | Moves that pack macros beyond 90% of canvas dimension in either axis |

### Design rationale

- **Zobrist hashing:** Navigation can revisit topologies via different flip
  sequences. XOR hashing gives O(k) incremental updates with zero false
  positives. Hash table grows with visited topologies (~1000s), negligible memory.
- **Displacement weighting by net count:** A large displacement for a
  low-connectivity macro is cheap (few nets affected). Same displacement for a
  high-connectivity macro propagates into HPWL through many nets.
- **90% canvas threshold for crowding:** Cascade repair handles some crowding,
  but beyond 90% the remaining space is usually insufficient for non-chained
  macros. Conservative to avoid false positives.
- **No surrogate delta screening:** Surrogate requires projected positions.
  Screening happens *before* projection, using only topology change and current
  positions. The tiers are bounds, not estimates.

### Reporting

`screener.report()` prints a one-line summary after navigation:

```
Screened 1247: 412 pruned (33%), dup=187, displacement=98, noop=72, hpwl=31, crowding=24
```

---

## Stage 2: Candidate Scoring

### Current: GridSurrogate

Fast (~0.1ms) incremental proxy estimator using grid-based density + RUDY
congestion + HPWL. Global Spearman rho=0.899, but **within-benchmark rho=0.17**
-- the navigator is effectively ranking candidates at random.

SP3 overnight results (8 experiments) showed surrogate accuracy improvements
don't move the score. Lesson: only 1-7 candidates per benchmark are genuinely
better, and the surrogate finds most of them despite weak ranking.

### Proposed: Incremental Real-Proxy Evaluator

Replace GridSurrogate with an exact incremental evaluator that caches per-net
and per-cell state and updates only what a move touches. Target: match surrogate
speed (~0.1ms) while returning exact proxy cost.

Motivated by Vedu Mallela's Partcl submission ("Incremental CD"), which reached
a 300x speedup on the real objective and abandoned surrogates entirely.

**Status:** PROPOSED. Not yet implemented.

#### Why our moves are incrementally cheap

| Move type | Nets touched | Cells touched |
|-----------|--------------|---------------|
| Single pair flip + LP | O(degree of moved macros) | O(footprint + nets' bboxes) |
| Macro-centered k-pair flip | O(k * degree) | same |
| Soft-pair flips | O(degree) | same |

Out of M nets, each move changes O(1). A cached evaluator gives orders of
magnitude speedup on the dominant cost term.

#### Incremental decomposition

| Component | Update strategy |
|-----------|----------------|
| **HPWL** | Per-net `(x_min, x_max, y_min, y_max)` cache; subtract old, add new. O(delta_nets * \|S_j\|). |
| **Density** | Per-cell occupancy accumulator; subtract old overlap, add new. O(moved * footprint cells). Top-10% reducer. |
| **Congestion (RUDY)** | Per-cell demand accumulator keyed by net; subtract old bbox, add new. O(delta_nets * bbox cells). Top-5% reducer. |

All three reduce to "subtract old, add new" on cached accumulators. No full
recompute per move.

#### Order-statistic subtlety

Top-k reductions (top-10% density, top-5% congestion) aren't naturally
incremental. Start with full reduce after each move — O(RC log RC) but grids
are ~100x100, so the constant is tiny. Upgrade to bucketed quantile sketch
only if this dominates.

#### Implementation sketch

```
IncrementalObjective:
    # per-net cache
    net_xmin, net_xmax, net_ymin, net_ymax : Tensor[M]
    net_hpwl                                : Tensor[M]
    # per-cell cache
    density_accum   : Tensor[R, C]
    congestion_demand : Tensor[R, C]

    total_wl, total_density_top10, total_cong_top5 : scalar

    def apply_move(moved_indices, new_positions):
        # 1. For each moved macro: remove old density, add new.
        # 2. For each net touching a moved macro:
        #     - remove old bbox from congestion_demand
        #     - recompute bbox from scratch over net members
        #     - add new bbox to congestion_demand
        #     - update net_hpwl; delta into total_wl
        # 3. Full reduce for top-10% density / top-5% congestion.

    def revert():
        # symmetric undo — store diffs on apply_move
```

#### Milestones

1. `IncrementalObjective` matching `compute_proxy_cost` bit-for-bit on 1000
   random moves. Unit test against full recompute.
2. Per-move cost <=1ms for typical pair flip on ibm10.
3. Swap into Navigator behind a flag. Compare vs surrogate at equal wall time.
4. If (3) wins, delete GridSurrogate and simplify navigator.

#### Risk / kill gate

If exact signal gives <1% improvement over surrogate at equal wall time, the
bottleneck is the move set, not the signal. Kill and redirect to hierarchical
navigation ([approach.md](approach.md) SS7C). SP3 overnight results suggest this
risk is real — but those experiments improved surrogate *ranking*, not
*accuracy*. An exact evaluator tests a stronger form of the hypothesis.

---

## Entry points

| What | File | Notes |
|------|------|-------|
| ClusterScreener | `submissions/polyhedra/cluster_bounds.py` | Stage 1: pre-projection pruning |
| GridSurrogate | `submissions/polyhedra/surrogate.py` | Stage 2 current: ~0.1ms, rho=0.17 |
| Navigator integration | `submissions/polyhedra/navigator.py` | Screen → project → evaluate → accept |
| Proxy ground truth | `macro_place/objective.py` | Source of truth for incremental evaluator |

---

## See also

- [approach.md](approach.md) SS2 -- architecture overview
- [approach.md](approach.md) SS7H -- incremental evaluator summary
- [roadmap.md](roadmap.md) Option 6c -- timeline and prioritization
