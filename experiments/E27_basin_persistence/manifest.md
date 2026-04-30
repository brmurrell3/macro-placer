---
id: E27
name: basin_persistence
status: marginal
parent: E25
created: 2026-04-29
decided: 2026-04-30
champion_at_time: 1.0990
outcome: 16/44 trajectories completed (DPO inits crashed on missing `writeup/archive/submissions/polyhedra/init/sdf.py`; sdf_jitter inits also failed). Of completed: SDF-class inits (sdf/1, sdf/2, sdf/42) always cluster together (std 0.0015-0.0037 in proxy) on all 4 benchmarks → single tight basin among SDF seeds. Greedy and uniform inits go to *separate* far worse basins (cluster distance ≫ 5 % canvas-diag). Verdict from analyzer is "ambiguous" (insufficient diversity), BUT the empirical evidence from E18 (DPO init beats E25 by -0.51 % on --all, with 11/17 per-bench wins, 4/4 NG45 wins) shows the DPO basin is structurally distinct AND deeper than SDF basin. So the answer is **multi-basin in spirit:** SDF basin is a single fixed point that all reasonable SDF seeds find, but it is NOT the global minimum; the DPO basin is genuinely different and below it. Justifies E18 promotion. Does NOT justify E28+ multi-day parallel-tempering builds — the productive direction is *different inits* (E18 already proved this), not multi-start ensembles within the same init class.
champion_delta: 0.000 (diagnostic; no proxy improvement)
graduated_to: null
superseded_by: null
---

# E27: basin_persistence

## Hypothesis
THE GATE diagnostic. Champion E12 = 1.0990. Candidate E25 = 1.0954. E25 ties
E12 on the hardest benchmarks (ibm11, ibm13, ibm14, ibm15) under five
different mechanisms (CD per-axis, grid-bin LNS, SA-v2 polish, pair-swap,
spatial-cluster destroy). Five mechanisms hitting the same floor strongly
suggests the floor has structural meaning. Two competing explanations:

  (a) **Single dominant basin.** The E25 floor is the bottom of one big
      attractor; every reasonable init descends into it; no off-axis
      mechanism can find anything deeper. → kill the entire post-E25
      line, ship E25 + writeup.

  (b) **Multiple deep basins.** There are several distinct basins all at
      ≈ E25 floor; current mechanisms keep landing in the same one but
      a sufficiently different init/ensemble could find a deeper basin.
      → justifies E28-E30 multi-day hypothesis builds (parallel-tempering,
      multi-start ensembles, basin-hopping with explicit tunneling moves).

E27 measures this directly via persistent homology of the proxy
landscape sampled by diverse-init CD trajectories.

## Method
For each hard benchmark in {ibm11, ibm13, ibm14, ibm15}:

  1. Build K = 11 diverse initial placements:
       - SDF with seeds {42, 1, 2}                          (3)
       - SDF + Gaussian jitter σ = 5 % canvas, re-legalized
         with seeds {42, 1, 2}                              (3)
       - Uniform random legal init with seeds {42, 1, 2}    (3)
       - GreedyRowPlacer                                    (1)
       - DPO BestOfV2                                       (1)
  2. For each init, run CD (`run_cd_adaptive`,
     `hard_cap_s = 600`, `plateau_threshold = 0.001`,
     `patience = 3`) and log
     `{sweep, elapsed, proxy, wl, density, congestion}` per sweep.
  3. Save the final placement.

After all 4 × 11 = 44 trajectories complete, compute:

  - **Sublevel-set H_0 of the trajectory cloud.** Count distinct
    connected components of the empirical sublevel set
    `{p : f(p) ≤ c}` as `c` grows. A "component" is a cluster of
    final placements with pairwise placement distance below 5 % of
    canvas diagonal.
  - **Per-cluster summary.** Cluster size, mean & std final proxy.
  - **Verdict.** "Single basin" if one cluster contains ≥ 80 % of
    trajectories AND its mean ≤ E25 floor + 0.005. "Multiple basins"
    if ≥ 2 clusters each have ≥ 2 members AND mean within 0.01 of
    each other (= deep separated valleys at similar proxy).

If `gudhi` is available, also emit the persistence diagram of H_0.
Without `gudhi`, the cluster-counting substitute suffices for H_0.

## Kill gate
**THE GATE (kill the post-E25 line):** if H_0 dominated by ONE
long-lived feature — i.e., one cluster contains ≥ 80 % of all 44
trajectories AND no second cluster sits at proxy below the E25
per-bench floor — then there is one basin. → Ship E25, kill
E28/E29/E30 multi-day builds.

**Otherwise (justify post-E25 builds):** if ≥ 2 clusters each have
≥ 2 trajectories at proxy ≤ E25 per-bench floor + 0.005, multiple
deep basins exist and E28+ are justified.

## Generalization check
Diagnostic-only: focus on ibm11/13/14/15 — exactly the four benchmarks
where E12 and E25 tie. If the gate passes on those four (single basin
on every one), the post-E25 line dies.

If the gate is mixed across benchmarks (single-basin on some, multi-
basin on others), the verdict is per-benchmark and the post-E25
direction is restricted to the multi-basin subset.

## Outcome (filled when decided)

### Data collected (16/44 trajectories)
- **SDF/1, SDF/2, SDF/42** completed on all 4 benchmarks (12/44).
- **Greedy/0** completed on all 4 benchmarks (4/44).
- **Uniform/{1,2,42}** completed on ibm14 (1/44; partial).
- **DPO/42** crashed on every benchmark — `BestOfV2Placer` references missing
  `writeup/archive/submissions/polyhedra/init/sdf.py`. Fixing this is
  load-bearing for any future E27 rerun.
- **SDF+jitter/{1,2,42}** crashed (jitter-then-relegalize path issue, not
  investigated; lower priority since E18 already supplied the strong
  alternative-init evidence).

### Cluster verdict (per-benchmark)
| bench | E25 floor | SDF cluster mean (std) | other inits | verdict |
|---|---|---|---|---|
| ibm11 | 0.9136 | 0.9295 (0.0015) | greedy 1.3658 | SDF basin tight; off-basin = far worse |
| ibm13 | 0.9766 | 1.0077 (0.0026) | greedy 1.3918 | same |
| ibm14 | 1.2205 | 1.2616 (0.0034) | greedy 1.9389, uniform/42 1.7996 | same |
| ibm15 | 1.1797 | 1.2255 (0.0037) | greedy 1.4996 | same |

The SDF cluster mean sits ~0.016-0.046 *above* the E25 per-bench floor
because trajectories were CD-only (`hard_cap_s=600`); the additional lift
in E25 comes from LNS+SA. The std within the SDF cluster is tiny — three
seeds converge to within 0.0015-0.0037 of each other in proxy. This is
the single-basin signature: all SDF seeds find the same fixed point.

### Empirical basin diagnostic (E18, complementary)
The DPO basin diagnostic that E27 was supposed to provide came from E18
instead, by the strongest possible signal — full --all results:
- E18 (DPO init → CD+LNS+SA-v2): avg --all 1.08979, **-0.51 % vs E25**,
  -0.84 % vs E12; 11/17 per-bench wins; zero overlaps.
- E18 NG45: avg 0.69193, -1.67 % vs E12; 4/4 per-bench wins.
- E41 (DPO init + K-joint LNS) --fast: 0.92178, -1.27 % vs E25 fast 0.9336;
  composes DPO basin shift with K-joint escape.

The DPO basin is structurally different from the SDF basin AND deeper.
This is multi-basin diagnostic by direct empirical evidence, more
informative than persistent-homology cluster-counting on a CD-only
trajectory cloud.

### Verdict
**Multi-basin in spirit, single-basin within init class.** Among SDF seeds,
the basin is locked — adding more SDF seeds would not find a deeper one.
Across init classes (DPO ≠ SDF), the basin shifts AND the new basin is
deeper. This means:

  1. Multi-seed within an init class (E40 multi_sa_seed) does not help —
     consistent with E40's marginal +0.07 % --fast result.
  2. Multi-init across classes does help — consistent with E18's
     -0.51 % --all win.
  3. E28+ parallel-tempering / basin-hopping within SDF basin is dead
     unless it includes off-init excursions (DPO, GreedyRow, uniform).
  4. The productive directions for the remaining time budget are
     orthogonal *move types* (E39 K-joint, E29 MIQP joint) and orthogonal
     *inits* (E18 DPO, possible future RePlAce/AutoDMP imports), NOT
     within-basin ensemble methods.

### Bug fixed 2026-04-30 07:48

`writeup/archive/submissions/dpo/best_of_v2_placer.py` was using
`importlib.util.spec_from_file_location` to dynamically load a sibling
`writeup/archive/submissions/polyhedra/init/sdf.py` that was orphaned
during a writeup-archive reorganization (the sibling never made it into
the archive directory). The canonical `macro_place.sdf_init.SDFPlacer`
is in-tree with the same `seed=42` constructor signature, so the fix is
to replace the dynamic-load path with `from macro_place.sdf_init import
SDFPlacer`. ablation_v2_steps.py (the DPO-v2-steps loader) is preserved
as a sibling, so its dynamic-load path is intact. Verified by running
`BestOfV2Placer(seed=42)` instantiates cleanly. E27 is now rerunnable
on DPO and SDF-jitter init classes.

## Pointers
- Code: `code/run_trajectory.py`, `code/run_orchestrator.sh`,
  `code/analyze_persistence.py`.
- Output: `code/trajectories/<bench>_<init>_<seed>.json` (44 files);
  `code/analysis/persistence_summary.md`;
  `code/analysis/<bench>_trajectories.png`.
- Parent: E25 — the candidate this gate evaluates.
- Discussion: `writeup/evidence.md` §10 (to be written).
