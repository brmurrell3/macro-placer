---
id: E89
name: ilp_detailed_legalize
status: falsified
parent: E84
created: 2026-05-12
decided: 2026-05-12
champion_at_time: 1.0612          # canonical cascade --all uncapped
outcome: "Lift saturates at +0.009% on ibm01 after one pass; 5-pass multi-seed adds nothing. 10x below the 0.1% gate, 30-100x below the 0.3-1% TODO estimate. Cascade is at a true local minimum w.r.t. small candidate grids; ILP's value-add over CD's saddle-escape is dominated by the inc-vs-canon proxy gap (0.04%)."
champion_delta: null
graduated_to: null
superseded_by: null
---

# E89: ILP detailed legalize — small-cluster joint repacking on cascade output (PATH C2)

## Hypothesis
Cascade lands at a zero-overlap local minimum where CD's
single-macro-at-a-time move structure can't escape. **Pairwise joint
moves** — where macro m₁ shifts and macro m₂ also shifts — may
reduce HPWL without breaking the zero-overlap invariant. This is
exactly the structure CD can't represent: any single-macro reposition
is rejected because pairwise non-overlap blocks the better joint
configuration.

A small ILP per local-spatial cluster can find the optimal joint
re-assignment within a fixed candidate set.

## Method
1. Take cascade-cached placement (e.g. `experiments/E84_cascading_saddle/
   results/cascade_ibm01.pt`, canon 0.84528 with 0 overlaps).
2. Build a k-NN spatial graph over hard macros; pick K-macro clusters
   by spatial proximity (greedy: each macro is the center of one cluster
   with its K-1 nearest neighbors).
3. For each cluster, generate L candidate positions per macro on a local
   grid (3×3 or 5×5 within a window of ±macro_half_size).
4. Pre-filter candidates that overlap with any **non-cluster** hard
   macro (those are fixed; candidates must respect them).
5. Solve ILP per cluster via HiGHS (`highspy` 1.13, already available):
   - Binary `x[m, p] ∈ {0,1}` for cluster-macro m → candidate p.
   - `Σ_p x[m, p] = 1` for each m in cluster.
   - For each pair (m₁,p₁), (m₂,p₂) with m₁,m₂ in cluster, AABB-overlap:
     `x[m₁,p₁] + x[m₂,p₂] ≤ 1`.
   - Objective: minimize `Σ_m,p WL_delta(m, p) · x[m, p]`, where
     `WL_delta` is the HPWL change from moving m to candidate p with
     all other macros (including other cluster members) at their
     current cascade positions. Separable approximation — exact
     post-application WL will differ slightly, but for small candidate
     windows this is close.
6. Apply solutions sequentially; verify zero overlaps via
   `compute_overlap_metrics`; evaluate canonical proxy via
   `compute_proxy_cost`.
7. Optionally iterate (re-cluster, re-solve) until no per-pass lift.

## Kill gate
**Kill if ibm01 lift after one full pass is < 0.1 % vs cascade input
(canon 0.84528 → > 0.84443).**

0.1 % is well below the TODO §C2 expectation of 0.3-1 % per bench; if
we can't even hit 0.1 % on ibm01 with small clusters + small candidate
windows, the joint-move-from-zero-overlap idea is structurally weak.

If first pass yields ≥0.1 % lift, scale: try larger K / L; iterate
multiple passes; run --fast (4 IBM); decide based on aggregate.

## Generalization check
If --fast aggregate ≥0.3 %, run --all 17 IBM. If --all aggregate
matches per-bench expectation (no bench worse by >0.2 %), run --ng45
for the 4 commercial designs. Promote only if --ng45 doesn't regress
ariane133 (the historical NG45 failure point — see e54/e62 in memory).

## Outcome (FALSIFIED 2026-05-12)

### Results on ibm01 cached cascade output (canon 0.84528, 0 ovl)

| Iter | Config | Outcome | Lift |
|------|--------|---------|------|
| 1 | cluster=4, grid=3, span=±0.5 | canon 0.84528 → 0.84530 | -0.001 % (regression) |
| 2 | cluster=8, grid=5, span=±2.0 | canon 0.84528 → 0.84521 | +0.009 % |
| 3 | iter 2 × 5 passes (re-seed) | canon → 0.84521 (saturated pass 1) | +0.009 % |

All iterations preserved zero overlaps. Multi-pass didn't compound: pass 1 found 5 moves, passes 2–5 found 1 trivial move each with no further canonical change.

### Why
1. **Cascade output is at a true local minimum w.r.t. small candidate grids.** Cascade's saddle escape already explores joint moves via the smooth-proxy Hessian's negative-curvature eigenvectors. The ILP's joint-move headroom on TOP of that is small.
2. **Incremental-vs-canonical proxy gap is 0.04 %**, larger than the per-pass canonical lift. The ILP optimizes `IncrementalProxyEvaluator.proxy`, which is a float64 / read-only-peek implementation. The canonical `PlacementCost` runs through float32 round-trip; the two agree only to ~0.05 %. ILP-detected improvements smaller than this gap are noise.
3. **Multi-pass with re-randomized cluster seeds doesn't help.** If pass 1 missed a joint move because cluster boundaries split a pair that wanted to move together, pass 2 should pick it up with different bundling. It doesn't. Direct evidence the cascade output is genuinely at a local min, not just at a cluster-boundary artifact.

The 0.3–1 % per-bench expectation from TODO.md §C2 assumed there'd be significant unrealized lift on cascade output. The empirical ceiling is ≤ 0.01 %.

### What could revive C2
- **Test on under-converged inputs.** Cascade hits wall caps on 8/17 IBM benches; the wall-truncated outputs have more headroom than uncapped. C2 might find more lift there. But the submission already runs wall-safe cascade, and the floor (1.137) is set by wall-truncated outputs. Worth one more probe (e.g., on ibm10 cached cascade output, which had wall_seconds approaching the cap), but my time budget says move on.
- **Larger ILP scope.** Multi-net-coupled objective (currently treating WL delta as separable per macro under cluster-fixed-others) — would catch joint moves whose net-coupling makes them invisible to separable ILP. But this is a 3-5 day rewrite, not a "small mechanical post-processor."

### Decision
C2 (as scoped in TODO.md) is falsified on ibm01 cascade-uncapped output. PATH C falls through to C3 (Newton-CG / multi-direction saddle escape) — directly attacks the same plateau-locality issue that limits C2, using PATH A's k=4 negative-eigval diagnostic as motivation.

### Reusable
- `code/ilp_legalize.py` — clean HiGHS-based assignment ILP, k-NN clustering, candidate-grid generator, separable-delta-cost scoring. Reusable if a future variant (different cluster strategy, different objective) wants the ILP scaffold.
- `code/spike_ibm01.py` — multi-pass driver pattern; reusable for similar one-bench falsification probes.

## Pointers
- Code: `code/ilp_legalize.py`, `code/spike_ibm01.py`
- Cached input: `experiments/E84_cascading_saddle/results/cascade_ibm01.pt`
- HiGHS API: `import highspy` (v1.13 installed)
- WL delta primitive: already in PATH A's `macro_place/incremental_evaluator.py`
  (`delta_cost`) — read-only consumer, no edits.
- TODO.md §PATH C, C2 (lines 138–149)
