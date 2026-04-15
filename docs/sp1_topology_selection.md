# SP1: Initial Topology Selection — Improvement Avenues

Last updated: 2026-04-15

## Current State

- SDF v5 generates initial positions optimizing density
- `extract_assignment()` picks L/R/A/B direction with largest gap per pair
- LP solves HPWL within that topology; navigator searches neighboring topologies
- **Problem:** SDF basin is a proven local minimum — 6 experiments show no local escape
- **Opportunity:** swap+LP proved 44% lower congestion EXISTS in other topologies

The initial topology determines ~95% of final congestion. Everything downstream (LP, navigation) optimizes within the basin SP1 chooses.

## Overnight Sweep Results (Apr 14-15)

6 experiments tested. **All alternative inits failed badly.** Only congestion-aware extraction completed (neutral at 1.4930).

| Approach | Status | Avg | Key finding |
|----------|--------|-----|-------------|
| Spectral topology (Approach 1) | **killed** | 1.78 | Spectral coords ignore macro sizes → overlapping clusters → legalizer scatters them |
| RePlAce extraction (Approach 2) | **killed** | — | extract_assignment() erases any congestion advantage — same topology as SDF |
| Congestion-aware extraction (Approach 3) | **best** | 1.4930 | Largest-gap is already near-optimal for shared-net pairs |
| hMETIS partitioning (Approach 4) | **killed** | 1.91 | kahypar works, but shelf packing within partitions produces terrible layouts |
| Greedy construction (Approach 7) | **killed** | 1.75 | Clustering for WL creates dense regions — fundamental WL vs density conflict |
| Boundary attraction (Approach 8) | **killed** | — | Penalty inert at safe lambda; larger lambda hurts WL |

**Critical insight:** SDF's analytical spreading produces topology that is **extremely hard to beat**. The issue isn't that SDF finds a bad topology — it's that alternative inits either (a) produce terrible density that legalizer/navigator can't recover, or (b) get mapped to the same topology after extract_assignment().

**Remaining viable approaches:** Only multilevel navigation (Approach 6) and TCG (Approach 5) remain untested. These change navigation reach, not init quality.

**Status: INIT APPROACHES EXHAUSTED.** Focus shifts to multilevel (structural change to navigation).

---

## Key Insight from Literature

The fundamental pattern across all successful macro placers is:

> **Phase A:** Global topology from CONNECTIVITY (partitioning, spectral, dataflow)
> **Phase B:** Local refinement for DENSITY (spreading, navigation)

Our pipeline inverts this: we start density-optimal (SDF) and try to navigate to better congestion. The literature overwhelmingly suggests starting connectivity-optimal and refining for density — which is exactly what our navigator is good at.

---

## Approach 1: Spectral Topology (Hall/Fiedler Vector)

**Source:** Hall 1970, "An r-Dimensional Quadratic Placement Algorithm"

**Idea:** Compute the netlist Laplacian matrix. Eigenvectors 2 and 3 (Fiedler vectors) give 2D coordinates that minimize total squared wirelength. Extract topology from these spectral positions instead of SDF positions.

**Why it's different from SDF:** Spectral placement minimizes a GLOBAL connectivity objective — it places connected macros close together, producing small net bounding boxes and thus low congestion. SDF optimizes density. These are fundamentally different topologies.

**Implementation:**
1. Build netlist Laplacian: `L = D - A` where A is the macro-macro adjacency weighted by shared net count
2. Compute eigenvectors 2,3 via `scipy.sparse.linalg.eigsh(L, k=4, which='SM')`
3. Scale to canvas dimensions
4. Legalize (greedy shelf or SDF-style)
5. Extract assignment, solve LP, navigate

**Expected outcome:** Better congestion, worse density. Navigator should recover density.

**Effort:** LOW (~2 hours). Standard sparse eigenvector computation.

**Risk:** Spectral positions ignore macro sizes — legalization may destroy the connectivity structure. Mitigation: use spectral positions only for topology extraction, then let LP find positions.

---

## Approach 2: RePlAce Topology Extraction

**Source:** GOALPlace (Agnesina et al., ISPD 2025) — "begin with the end in mind"

**Idea:** Extract pairwise L/R/A/B assignment from the RePlAce baseline positions instead of SDF positions. RePlAce's electrostatic spreading implicitly handles congestion through smooth density redistribution. Its topology may be in a better congestion basin.

**Implementation:**
1. Run RePlAce to get baseline positions (or use stored results)
2. Call `extract_assignment(replace_positions, sizes, hard_indices)`
3. Solve LP within RePlAce's topology
4. Navigate from there

**Expected outcome:** Inherits RePlAce's congestion-friendly structure. LP + navigation may improve upon RePlAce's positions within its own topology.

**Effort:** VERY LOW (~1 hour). Just need RePlAce positions as input.

**Risk:** RePlAce positions may not be available as raw data (only proxy cost reported). If positions aren't stored, need to run RePlAce or extract from the evaluation harness.

---

## Approach 3: Congestion-Aware Assignment Extraction

**Source:** Lin, Hsu, Chang (ICCAD 2008) — constraint graph macro placement

**Idea:** Instead of picking the direction with the largest gap (current `extract_assignment`), pick directions that minimize net bounding box area for high-congestion nets.

**Current logic** (`assignment.py:39-66`):
```python
best_dirs = argmax([gap_l, gap_r, gap_b, gap_a])  # Largest gap
```

**Proposed logic:**
```python
# For each pair (i,k), evaluate all 4 directions
# For each direction, estimate the change in bounding box area
# for all nets containing both i and k
# Pick direction that minimizes total net bbox area
```

**Implementation:**
1. For each pair (i,k), identify shared nets
2. For each candidate direction, estimate where macros would end up
3. Compute total HPWL change for shared nets
4. Pick direction minimizing HPWL of shared nets (proxy for congestion)
5. Fall back to largest-gap for pairs with no shared nets

**Expected outcome:** Topology that's locally optimal for net spans, not just for separation gaps. Should reduce congestion at the cost of tighter separations.

**Effort:** LOW (~2 hours). Modify `extract_assignment()`.

**Risk:** May produce topologies where LP is infeasible (tight constraints + wrong directions). Mitigation: keep largest-gap as fallback for conflicting constraints.

---

## Approach 4: hMETIS/KaHyPar Partitioning Topology

**Source:** Karypis et al. 1997; mPL6 (Chan et al., 2006)

**Idea:** Recursive bisection of the netlist hypergraph generates a hierarchical topology that MINIMIZES NET CUTS — i.e., minimizes the number of nets spanning partition boundaries. Since congestion is proportional to net bounding box area, minimizing cuts directly targets congestion.

**Implementation:**
1. Build hypergraph: macros = vertices, nets = hyperedges, weights = net fanout
2. Bisect into LEFT/RIGHT (use KaHyPar or pymetis)
3. Within each half, bisect into TOP/BOTTOM
4. Recurse until each partition has 1-4 macros
5. Convert partition tree to pairwise assignment:
   - All macros in LEFT partition are left-of all macros in RIGHT partition
   - All macros in TOP partition are above all macros in BOTTOM partition
6. Solve LP, navigate

**Expected outcome:** Topology where connected macros are in the same partition (close together), producing small net bounding boxes and low congestion. Completely different from SDF.

**Effort:** LOW-MEDIUM (~1-2 days). KaHyPar has Python bindings. The partition-to-assignment conversion needs care.

**Risk:** Min-cut partitioning doesn't account for macro sizes — partitions may be size-imbalanced, causing density problems. Mitigation: use balanced partitioning (KaHyPar supports balance constraints).

---

## Approach 5: TCG (Transitive Closure Graph)

**Source:** Lin & Chang (DAC 2001, TVLSI 2005)

**Idea:** Represent topology as two DAGs (horizontal and vertical) with transitive closure maintained explicitly. When you flip one pair's direction, TCG tells you which OTHER pairs MUST change to maintain acyclicity. This directly solves why group topology changes create infeasibilities — you're currently changing pairs without propagating transitive consequences.

**Implementation:**
1. Build H-graph and V-graph from current assignment
2. Compute transitive closure (Floyd-Warshall or incremental)
3. For any proposed pair flip, compute the minimal consistent set of cascading flips
4. Apply all consistent flips together → guaranteed feasible topology

**Why this matters:** The cascade repair in `projection.py` fixes POSITION violations, not TOPOLOGY violations. TCG fixes topology violations at the source — ensuring the constraint graph is acyclic before asking LP to solve.

**Expected outcome:** Enables large coordinated topology changes (50-200+ pairs) that are guaranteed consistent. This is the missing piece for crossing the congestion barrier.

**Effort:** MEDIUM (~3-5 days). Well-documented but requires careful O(N^2) closure updates.

**Risk:** O(N^2) storage and O(N^3) closure update may be slow for N=500+. Mitigation: sparse closure for essential edges only.

---

## Approach 6: Multilevel Topology Navigation

**Source:** mPL6 (Chan et al., 2006); Karypis/Kumar multilevel paradigm

**Idea:** Build macro hierarchy via clustering. At coarse level with K=20 super-macros, there are only K(K-1)/2=190 pairwise assignments — small enough for aggressive exploration. Large topology changes at coarse level become single pair flips between super-macros.

**Implementation:**
1. Cluster macros by net connectivity (hMETIS or modularity clustering)
2. Build coarse problem: super-macro sizes = sum of member sizes, super-nets = nets spanning clusters
3. Run current pipeline at coarse level: SDF init → LP → navigate (with exhaustive search possible at K=20)
4. Uncoarsen: expand super-macro placements to individual macros
5. Run fine-level navigation within each cluster

**Why this breaks the barrier:** At the fine level, crossing the congestion barrier requires 100+ pair flips. At the coarse level, the same structural change is 1-5 super-macro pair flips — well within navigation's reach.

**Expected outcome:** Access to fundamentally different topologies that local navigation cannot reach. This is the most promising approach for breaking the congestion barrier.

**Effort:** HIGH (~1-2 weeks). Requires clustering, coarse problem formulation, and multi-level refinement.

**Risk:** Coarse-level decisions may not survive refinement. Mitigation: iterate between levels.

---

## Approach 7: Greedy Sequential Construction

**Source:** WireMask-BBO (Shi et al., NeurIPS 2023)

**Idea:** Instead of extracting topology from simultaneous positions, BUILD it greedily: order macros by connectivity (most-connected first), place each one by choosing L/R/A/B directions that minimize RUDY congestion relative to already-placed macros.

**Implementation:**
1. Sort macros by total net weight (most connected first)
2. Place first macro at canvas center
3. For each subsequent macro:
   a. For each already-placed macro, evaluate all 4 directions
   b. Compute RUDY congestion contribution for shared nets under each direction
   c. Choose direction minimizing total congestion
4. After all macros assigned, solve LP within the resulting topology

**Expected outcome:** Topology that's greedily congestion-optimal. Different from both SDF and spectral approaches.

**Effort:** MEDIUM (~2-3 days). The O(N^2) direction evaluation per macro is the bottleneck.

**Risk:** Greedy ordering may make globally poor choices. Early placement decisions constrain later ones.

---

## Approach 8: Boundary-Attracted Initialization

**Source:** IncreMacro (Pu et al., ISPD 2024, Best Paper Candidate)

**Idea:** Push macros toward chip boundaries. This reduces routing congestion in the center (the most congested region). Modify the LP or SDF to include a boundary-attraction term.

**Implementation (LP variant):**
1. For each macro i, add penalty terms to LP objective:
   `lambda * min(x_i - x_lo, x_hi - x_i) + lambda * min(y_i - y_lo, y_hi - y_i)`
2. Linearize using auxiliary variables: `d_x = min(x_i - x_lo, x_hi - x_i)` with LP constraints
3. Objective becomes: `HPWL + lambda * sum_i (d_x_i + d_y_i)`

**Implementation (SDF variant):**
- Add boundary attraction force to SDF optimization

**Expected outcome:** Macros cluster at boundaries, leaving the center for routing. Common in commercial tools.

**Effort:** LOW (~half day for LP variant).

**Risk:** Boundary packing may conflict with density objectives.

---

## Recommended Priority

| Priority | Approach | Effort | Impact | Rationale |
|----------|----------|--------|--------|-----------|
| 1 | Spectral topology | 2 hours | HIGH | Fundamentally different basin; tests the core hypothesis |
| 2 | RePlAce topology extraction | 1 hour | MEDIUM | Zero-cost test of a known-good congestion topology |
| 3 | Congestion-aware extraction | 2 hours | MEDIUM | Low-risk modification of existing code |
| 4 | hMETIS partitioning | 1-2 days | HIGH | Connectivity-driven topology; strong literature support |
| 5 | Boundary attraction | half day | LOW-MED | Quick addition to LP or SDF |
| 6 | TCG closure propagation | 3-5 days | HIGH | Enables large consistent topology moves |
| 7 | Multilevel navigation | 1-2 weeks | VERY HIGH | Most promising for breaking congestion barrier |
| 8 | Greedy construction | 2-3 days | MEDIUM | Alternative to partitioning |

**Suggested sequence:** Start with 1+2 (same day, test two fundamentally different topologies). If either shows >5% congestion improvement, invest in 4 or 7. If neither helps, the problem isn't initial topology — focus on SP4 instead.

---

## See Also

- [approach.md](approach.md) — current architecture and congestion barrier analysis
- [sp4_congestion_lp.md](sp4_congestion_lp.md) — congestion-aware LP formulation
- [sp3_surrogate_accuracy.md](sp3_surrogate_accuracy.md) — surrogate ranking improvements
