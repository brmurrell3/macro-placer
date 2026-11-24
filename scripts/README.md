# Scripts

Most scripts here build or check the technical report in
[`../writeup/`](../writeup/). They stay at these paths because the frozen
records in `writeup/data/publication/` and `writeup/runs/` cite them by path.

## Study runs

| Script | What it does |
|---|---|
| `run_publication_study.py` | Runs a benchmark study from a config in `writeup/study/`, one supervised worker per design |
| `publication_worker.py` | The isolated worker for one design of a study |
| `freeze_publication_runtime.py` | Archives the solver runtime a study ran, as a hashed snapshot in `writeup/data/publication/` |
| `freeze_publication_evidence.py` | Freezes the May 2026 evidence and regenerates the report's tables and figures from it |

## Report tables and figures

Each reads saved records (frozen evidence, run records or git history) and
writes into `writeup/latex/generated/`, except the explorer, which writes its
own folder. None runs a solver.

| Script | What it renders |
|---|---|
| `render_publication_v1.py` | The report v1.0 figures and tables |
| `render_publication_reproduction.py` | The ranked-configuration reproduction |
| `render_publication_pilot.py` | The 12-run validation profile |
| `render_publication_integration.py` | The short NG45 integration runs |
| `render_publication_measurements.py` | The evaluator microbenchmarks |
| `render_publication_curvature.py` | The ibm01 endpoint-curvature diagnostic |
| `render_publication_devrecords.py` | The development-records table and codename chronology |
| `build_publication_explorer.py` | The interactive placement explorer in `writeup/interactive/` |

## Measurements and audits

Post-competition diagnostics. Each writes one frozen record.

| Script | What it measures |
|---|---|
| `measure_publication_evaluator.py` | The incremental evaluator against full proxy evaluation |
| `measure_publication_tail_bounds.py` | The top-k stability bound on the evaluator's recorded moves |
| `measure_publication_ibm_repair_equivalence.py` | Whether the post-competition input repairs change any IBM input |
| `measure_publication_topology.py`, `measure_publication_sdf_topology.py` | The input-parsing and SDF graph changes, in isolation |
| `diagnose_publication_curvature.py`, `diagnose_publication_curvature_followup.py` | Curvature at the frozen E74 ibm01 endpoint |
| `rescore_publication_e74_e84.py`, `audit_publication_e74_e84_geometry.py` | Canonical rescoring and geometry checks of the 34 frozen E74/E84 placements |
| `analyze_publication_referee_context.py` | Baseline context and numerical margins from frozen records |

## Release checks and packaging

| Script | What it does |
|---|---|
| `check_publication_paper.py` | Source and release checks for the report (`--source-only` while editing) |
| `check_publication_explorer.py` | Browser check of the placement explorer |
| `package_publication.py` | Collects the PDF and website assets; does not deploy |

## Website

| Script | What it renders |
|---|---|
| `render_website_leaderboard.py` | The chart of every ranked entry on the official final leaderboard |
| `render_placement_animation.py` | The animation of one ibm01 run used in the README and the website |

## May 2026 development tools

| Script | What it does |
|---|---|
| `bench_incremental.py` | Speed of the incremental evaluator against full proxy evaluation |
| `best_of_n_analysis.py` | Per-design best of N across experiments |
| `cd_ibm10_diagnostic.py` | Coordinate descent on ibm10 from the SDF start |
| `lp_hpwl_lower_bound.py` | A linear-programming lower bound on wirelength |

## From the upstream challenge

Benchmark conversion and OpenROAD flow tools: `convert_ibm_benchmarks.py`,
`convert_asap7_benchmarks.py`, `evaluate_with_orfs.py`,
`evaluate_shelfpack.py`, `generate_macro_placement_tcl.py`,
`run_openroad_flow.tcl`, `test_macro_placement.tcl`,
`view_placement_openroad.tcl` and the netlist patch `ariane133_lzc_patches.v`.
