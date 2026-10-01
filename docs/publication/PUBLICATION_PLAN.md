> **Archived.** Planning document for the September 2026 publication work; superseded by report v1.0.

# Thinkorplace publication plan

**Execution status, 6 September 2026:** the LaTeX working report, website summary,
17-design historical explorer, runtime extraction for both versions, and frozen
evidence are complete. New evidence includes evaluator timings, curvature and
topology diagnostics, two separately recorded 12-run engineering batches, and a
three-run final NG45 integration check. The repaired runtime passes 64 focused
tests. The controlled 51-run main comparison, its repetitions and mechanism
ablations, and website deployment remain unexecuted. IBM17's short-budget
timeouts still require a realistic-budget finalization preflight before the long
study. The sequence below remains the research roadmap; historical results and
engineering checks do not replace its controlled experiments.

Target: a rigorous public technical report, a short website article, and an
interactive placement explorer. Agreed 2026-09-06: use the Mac first; consider
rented compute only if the measured workload or a Linux/CUDA validation justifies
it. No cloud rental or publication has been initiated.

**Working title:** Thinkorplace: Diagnosing and Improving Proxy-Driven Macro Placement

**Thesis:** Fast incremental evaluation makes local search on the competition
objective practical. Differentiable models supply starting placements and
coordinated perturbations; canonical scoring filters and refines those proposals.
The report should explain which parts improve quality under a fixed resource
budget, rather than narrating every experiment in chronological order.

The current draft and research notes provide substantial source material. They
need a structural rewrite and claim audit, not merely a prose polish. Preserve
the existing draft during this work. The [post-competition assessment](../reviews/2026-09-06-thinkorplace.md)
records the known packaging, timing, topology, and score-attribution problems.

**The publication package**

| Deliverable | Reader experience | Suggested size |
|---|---|---|
| Website article | What macro placement is; what was built; the main measured result; what was learned; links to paper, code, and results | 300–500 words plus one visual |
| Technical report PDF | Methods, controlled comparisons, mechanism analysis, limitations, reproducibility | 12–16 pages of main text, with appendices as needed |
| Interactive explorer | Inspect real placements, compare methods and phases, see objective components and recorded progress | One focused responsive page/component |
| Reproducibility bundle | Exact source/environment/configuration, results, checkpoints, and commands behind each figure | Versioned data and scripts |

Generate paper tables, static publication figures, and explorer data from the same
frozen result records. Avoid manually maintaining three different sets of scores.
Provide a dated report version and a stable PDF URL; archive older report versions
when results change. A PDF can be a professional technical report without claiming
peer review or an official competition ranking for new runs.

**Paper structure**

1. **Abstract and introduction.** Define the problem, the concrete contributions,
   and the measured scope. Write the final abstract after results are frozen.
2. **Problem and evaluation.** Placement representation, legal geometry, fixed
   macros, soft clusters, canonical proxy components, resource budget, benchmark
   provenance, and the distinction between proxy quality and downstream PnR.
3. **System design.** One architecture figure; shared incremental evaluation;
   initialization and legalization; v1 cascade and v2 A/B/C pipelines. Explain
   the final algorithms before giving their development history.
4. **Experimental protocol.** Revisions, hardware/device, threads, timing boundary,
   seeds and initialization, failures, legality, and benchmark selection.
5. **Main results.** v1 versus Lane A versus the operating ensemble; per-design
   table; suite average; dispersion; runtime and resource use.
6. **Why it works.** Initializer/fidelity ablations, extra-compute controls,
   curvature versus matched random perturbation, and Hungarian contribution.
7. **Failure analysis and limitations.** Weak designs, failed/late lanes,
   legalization loss, surrogate mismatch, public-suite reuse, NG45 transfer, and
   what proxy improvements do not establish about timing or routability.
8. **Related work and conclusion.** Position the measured contribution precisely.
   Summarize supported findings and remaining questions.
9. **Appendices.** Full tables/configurations, extra diagnostics, representative
   unsuccessful experiments, implementation details, and a concise development
   chronology. Move most E-number history here.

**Claims to repair before writing the final prose**

| Existing framing | Defensible framing / required evidence |
|---|---|
| The plateau is a high-index saddle | Negative-curvature directions were observed in the surrogate at canonical local-search endpoints. A stationary-saddle claim additionally needs gradient and derivative checks; the two objectives are different. |
| Direct transplant of dimer/NEB with their guarantees | Curvature-guided perturb-and-polish, inspired by minimum-mode methods. The implementation is not a dimer or a chain-of-states NEB solver. |
| Entire objective is C∞ | Gaussian bin contributions smooth bin-boundary effects; top-k selection and clamping still introduce nonsmoothness. |
| True local minimum / algorithmic ceiling | No accepted improvement under the attempted directions and budget / best observed uncapped result. |
| 4,657× equivalent scoring speedup | Rebenchmark matched operations. Incremental `move` includes current-cost calculation, but the full canonical wrapper additionally computes hard-macro overlap metrics. Separate setup, matched component scoring, overlap validation, and end-to-end benefit. |
| 0.98387 is the shipped three-lane result | Historical standalone extended-CD result. The 0.97745 figure is an offline oracle. Current ensemble results need a fresh, completely recorded run. |
| NG45 is held out and proves generalization | NG45 was inspected during development. Call it a transfer suite; fix hierarchical pin extraction and rerun before explaining its mechanisms. |

These corrections strengthen the report. A careful empirical contribution is more
credible than an unproved theorem or a misleading leaderboard projection.

**Mac-first execution sequence**

1. **Freeze and repair the evaluation boundary.** Preserve competition source and
   historical results. Make corrected post-competition code a separately labeled
   revision. Extract package dependencies; fix global/inner deadlines and the
   hierarchical pin mapping; validate all legality conditions; ensure each
   process records its last usable result. Do not silently rewrite history.
2. **Make runs self-describing.** Add one runner that records the manifest below
   and saves per-phase placements and incumbent events. Smoke-test all entry
   points and interruption/failure paths before long jobs.
3. **Pilot on the Mac.** Use IBM01, IBM10, IBM17, and ariane133 to exercise small,
   larger, difficult, and hierarchical inputs. These are engineering preflights,
   not a basis for claiming full-suite superiority. Measure actual runtime and
   peak memory. Fix the experimental configuration before collecting headline
   results. Limit concurrent jobs so comparisons have controlled resources.
4. **Collect the first main table.** On the same machine/device and under the
   same enforced budget, run corrected v1, v2 Lane A, and full v2 A+B+C on all
   17 IBM designs. This is 51 benchmark-runs for one seed/repetition. Record
   every failure; do not average only successful designs.
5. **Replicate and explain.** Extend the main comparison to three effective
   seeds/repetitions (153 total runs, including the first 51). Run the focused
   controls below on a predeclared subset; expand effects that support headline
   claims to the full suite. Keep exploration and confirmation labels distinct.
6. **Freeze evidence, then finish the paper and site.** Generate figures and
   tables, rewrite the draft around measured results, review claims against data,
   render/inspect every PDF page, and integrate the explorer with the actual
   website. The website source/location is still to be supplied.

The seed API needs attention: default SDF uses seed 42 regardless of lane seed.
Record initial tensor hashes. Repeating the same effective initialization is a
repeatability study, not an independent-start study. If adding seed propagation,
label that as a post-competition change and preserve the original configuration.

**Compute decision**

The current Mac exposes 14 CPU cores and 36 GiB RAM. This establishes an available
local environment, not a throughput forecast. Pin CPU/MPS selection explicitly
instead of allowing every run to choose a device implicitly.

At a 3,600-second external cap, 51 runs permit at most 51 aggregate benchmark-hours
inside that cap; 153 permit at most 153. Setup, external evaluation, artifact
writing, and retries must be accounted for separately if outside the timed
boundary. Actual runtimes can be much lower. These are **benchmark-hours**, not
core-hours: allocated core-hours are the sum of allocated cores × elapsed hours.

Use the pilot's measured runtimes to estimate the schedule before renting.
Cloud is useful if repeated runs would monopolize the Mac for too long or if
Linux/CUDA portability is a desired result. It is not required merely to call the
report professional. If moving the primary study to cloud, run the comparison
variants on the same cloud configuration; do not compare a cloud v2 result against
a Mac v1 result as though only the algorithm changed. An explicit rental budget
and chosen configuration should precede any chargeable provisioning.

**Experiments that explain the result**

| Question | Controlled comparison | What to record |
|---|---|---|
| Does v2 improve on v1? | v1 / Lane A / full ABC; same environment and resources | Per-design canonical score and components, legality, elapsed time, peak memory, completion coverage |
| Do extra mechanisms beat extra compute? | ABC versus longer A and, where resources permit, three effectively diversified A runs | Quality-versus-time/resource curves; operating ensemble outcomes |
| What do B and C contribute? | Capture every lane in the same ABC run | Winner counts, gain over A, failures/timeouts, completion times; retrospective AB/AC selection labeled as such |
| Does curvature guidance matter? | Same plateau, perturbation norms, signs, trial count, legalization and polish budget: Hessian / random / shuffled Hessian / more CD | Canonical gains, rejected/infeasible proposals, eigensolver residual/status, HVP count and time |
| What does Hungarian search add? | Same start and budget: Hungarian / randomized proposal / more CD | Accepted moves, gain, time, post-polish outcome |
| Which smooth-model change matters? | Rectangle/Gaussian density; old RUDY/per-net trace; congestion on/off; CD on/off, changing one factor at a time | Scores before and after legalization/CD, runtime, sensitivity and boundary bias |
| Is the surrogate useful for proposals? | Same states and perturbations, predicted versus canonical cost differences | Move ranking, directional agreement, component errors; use multiple designs |
| Do results transfer to NG45? | Old/fixed pin extraction, with matched configurations on four public NG45 designs | Canonical results and legality; optional real PnR only if run and completed |

For primary comparisons, use the competition's arithmetic mean alongside the full
per-benchmark table. Show paired differences and seed/repetition dispersion.
With only three repetitions, show the actual points or ranges; avoid strong
significance claims from tiny samples. Public designs reused during development
are not independent samples proving broad industrial generalization.

**Required run manifest**

- Run ID, variant, source commit and dirty-tree status, constructor configuration.
- Benchmark/netlist/config hashes, evaluator and submodule revisions.
- OS, Python/PyTorch/dependencies, CPU/GPU, explicit device, thread/core limits,
  memory allocation, number of concurrent benchmark jobs.
- Nominal seeds, effective seeds, initial placement hash, final placement hash.
- End-to-end and per-phase timing, deadline, timeout/error status, lane status.
- Canonical wirelength/density/congestion/proxy at each saved incumbent.
- Shape, finite values, bounds, fixed-macro preservation, hard overlap count.
- Saved coordinates at meaningful phase boundaries and on accepted improvements;
  sampled histories for the explorer, with timestamps and exact source run IDs.

**Figures and interactive companion**

The paper should have roughly six central figures: system architecture;
per-benchmark comparison; quality versus time; paired component ablations;
surrogate versus canonical move changes; curvature/random controls with real
placement case studies. Put comprehensive lane, sensitivity, and failure tables
in the appendix when necessary. Use reproducible plotting scripts and vector
PDF/SVG output for print; do not use screenshots of browser charts as paper figures.

The website explorer should let a reader select a benchmark and compare recorded
v1/v2/lane placements. Show canonical objective components, recorded time, and
legality. Add density/congestion overlays and a checkpoint scrubber only when the
underlying data exist. Highlight a macro's movement or a selected net without
implying that simplified displayed connections are a routed design. Provide
keyboard/touch controls, reduced-motion behavior, source links, and a static
fallback. Recompute expensive metrics offline; the public page can remain static.

The initial conversation preview uses matching historical IBM01 E74→E84
snapshots: 0.8552724123 → 0.8452833891, zero recorded final hard overlaps.
It is a comparison of saved placements, not an optimization trajectory and not
the final v2 ensemble. Source coordinates and scalar histories are present in
`experiments/E74_hessian_saddle/results/hessian_ibm01.pt` and
`experiments/E84_cascading_saddle/results/cascade_ibm01.pt`.

**Website story outline**

- Two sentences introducing macro placement and why wirelength alone is not enough.
- One paragraph explaining the hybrid design: smooth proposals, exact acceptance,
  and complementary local searches.
- One small result block generated from the new frozen comparison, with hardware,
  budget, coverage, and a clear historical/post-competition label.
- Three concise lessons: objective fidelity, matched compute, and what failed.
- One explorer plus links labeled Read the technical report, View source, and
  Reproduce the results. Add author name, date, and report version at release.

**Initial primary-source reading list**

This is a starting set for precise attribution, not a completed novelty search.

- Henkelman & Jónsson (1999), [dimer method](https://doi.org/10.1063/1.480097):
  minimum-mode saddle-search inspiration and its differences from this heuristic.
- E & Zhou (2011), [gentlest ascent dynamics](https://arxiv.org/abs/1011.0042):
  clarify saddle-finding dynamics and avoid inheriting unproved guarantees.
- Liu & Yang (2017), [noisy negative-curvature descent](https://arxiv.org/abs/1709.08571):
  a closer connection for approximate Hessian-vector guidance and stationarity.
- Lu et al. (2015), [ePlace](https://doi.org/10.1145/2699873): distinguish
  electrostatic density objectives from Gaussian top-bin aggregation.
- Lin et al. (2019), [DREAMPlace](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement):
  establish analytical placement and toolkit acceleration as prior work.

The full report also needs the challenge's evaluator/benchmark publications,
RePlAce, and relevant placement/refinement literature. Review the actual final
winner implementations before making comparative-method claims; reproduced
baselines and published scores must remain separate.

**Release gate**

Every headline number traces to an immutable run and a table-generation script.
Every claimed mechanism has a matched control or is explicitly a hypothesis.
All failures are visible, benchmark coverage is complete for claimed aggregates,
and historical versus corrected results are distinguished. PDF figures are
legible at print size; the website and explorer work on mobile; paper/code/data
links resolve. No TODO markers, projected ranks, unsupported guarantees, or
unlabeled schematic trajectories remain in public content.
