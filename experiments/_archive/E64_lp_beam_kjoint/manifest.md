---
id: E64
name: lp_beam_kjoint
status: scaffolded
parent: E41 (K-joint backbone) + Phase-5 polyhedra LP (lower bound)
created: 2026-05-02
decided: null
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E64: lp_beam_kjoint — LP-bounded beam search over K-joint enumeration

## Hypothesis
The K-joint mechanism in E41 commits real lifts (12-95 K-tuples per
bench post-CD-LNS-SA). E42 K=4 / E43 longer-K-joint / E44 spatial
K-tuple all failed NG45 transfer because their **K-tuple selection
heuristics use local IBM benchmark structure** (adjacency density,
congestion peaks, geometric clusters), and that local structure
differs sharply between IBM and NG45 ariane133 (sparse 1433×1433
canvas, 133 hard macros, vs IBM 23-73 canvases with 246-760 hard
macros).

**The structural fix:** prune the K-tuple search using a
*benchmark-blind* lower bound, so the algorithm doesn't depend on
the local-structure heuristic. The natural lower bound is the
**LP-relaxation of the proxy within the candidate's polyhedron**
(L/R/A/B-assignment region). The Phase-5 polyhedra navigation work
established the framework: each fixed L/R/A/B assignment defines a
convex polyhedron, and within it the WL component is an LP (HiGHS
solvable). Density and congestion components decompose less cleanly
but admit conservative lower bounds (e.g., LP of HPWL within the
polyhedron + 0 for density/cong gives a strict lower bound on full
proxy).

**Claim:** with the LP lower bound pruning the search tree, K=10
becomes tractable (effective branching factor << 3 per level after
pruning), and the algorithm doesn't IBM-overfit because the LP is
benchmark-agnostic. Expected lift on top of E41 K=3: −0.3 to −1.0 %
on benches where K=3 saturates. NG45 transfer should hold because
the LP framework is netlist-structure-blind (cares only about
HPWL sum, not benchmark-specific patterns).

## Method (planned, not yet implemented)
Pipeline per benchmark (identical to E41 except step 7 K-joint
mechanism):

  1. DPO best_of_v2 init (E18 helper).
  2. project_overlaps.
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive (≤ 2400 s).
  5. Grid-bin LNS (≤ 600 s, cost-aware destroy).
  6. SA-v2 (≤ 600 s).
  7. **LP-bounded beam K-joint K=10 (≤ 600 s).** Replaces E41 K=3:
     a. Pick K=10 macros forming a structurally-coupled cluster
        (high-adjacency + LP-dual-significant, not just adjacency).
     b. Maintain a beam of partial assignments (default beam width 50).
     c. For each level (1 to K), expand each beam state by trying
        top-3 candidate positions per macro, pruning by **LP
        relaxation** of the resulting polyhedron's HPWL lower bound.
     d. At leaves: full proxy evaluation via IncrementalProxyEvaluator.
        Commit best leaf if it improves baseline by > 1e-7.
     e. Defensive `compute_overlap_metrics` post-commit revert.
  8. Validate, preserve fixed macros, return.

## Implementation requirements (load-bearing, blocks --fast launch)
1. **Polyhedra LP infrastructure (reconstruct from `writeup/archive/`):**
   - Pairwise L/R/A/B assignment extraction from a placement
     (`assignment.py`, `lp.py` in archived polyhedra source).
   - HiGHS LP solver wrapper (`highspy` available in env per
     `uv run python -c "import highspy"` check).
   - LP HPWL lower bound: solve `min sum_e w_e (max_x_e - min_x_e + max_y_e - min_y_e)`
     subject to L/R/A/B assignment constraints fixed.
2. **Beam search runtime** (~200 lines):
   - State: partial K-tuple assignment + LP value at each level.
   - Pruning: skip beam states whose LP-bound exceeds current best leaf.
   - Branching: top-3 candidates per macro (using same `_enumerate_topN_for_macro`
     from E39 K-joint).
3. **Cluster selection:** higher-K cluster picking (currently E39 picks
   K=3 by adjacency score). Use LP-dual values (which constraints are
   tight in the relaxation) as a structurally-aware K-cluster signal.
4. **Smoke test on ibm10** (E41's biggest hard-plateau win) before
   --fast. Validate beam width / K parameter sweep.

Estimated dev: 2-3 days focused work. The LP infrastructure is the
biggest item; once that exists, the beam search is straightforward.

## Kill gate
- **--fast > 0.929** (E41 + 0.8 %): mechanism is IBM-overfit despite the
  LP relaxation. Falsify; fundamental approach reconsidered.
- **--fast > E48 - 0.3 %** = > 0.917 (i.e., no measurable lift over E48
  hybrid): K=10 doesn't unlock structurally new commits beyond K=3.
  Mark marginal; possibly fall back to K=5 with same LP framework.
- **--ng45 ariane133 > E48 0.6861 + 1 %** (= 0.694): the LP framework
  is supposed to be benchmark-blind; if it still overfits, falsify and
  diagnose what's IBM-specific.

## Generalization check
NG45 mandatory before any promotion claim. The LP machinery should
transfer (it operates on netlist structure, not local benchmark
features), but the empirical test is required given the consistent
NG45-blind failure record (E42/E43/E44/E54/E62).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: TBD `code/cd_lns_sa_dpo_lp_beam_kjoint.py` (not yet written;
  requires polyhedra LP reconstruction first).
- Pipeline backbone: E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- Polyhedra LP source (deleted in commit 44efd16; need to extract):
  `writeup/archive/submissions/cd_lns_placer.py`,
  archived polyhedra modules (`assignment.py`, `lp.py`).
- LP solver: `highspy` package (verified available in env).
- Discussion: motivated 2026-05-02 by roadmap §4.6.A "global topology
  navigation" reframing. Direct response to "K-joint variants all
  break NG45 transfer" — the LP-relaxation lower bound is the
  benchmark-blind pruning mechanism that was missing from
  E42/E43/E44.
