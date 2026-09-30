# Thinkorplace runtime

This package extracts the active v1 and v2 solver dependencies from the research
tree. It is a corrected post-competition runtime, not the original competition
submission. `MIGRATION.json` maps source modules to their package locations;
Git commit `2ab92a33d352f0dbc4ba97d7c2bfad41ccf0ecfd` preserves the earlier source.

Import `thinkorplace.v1.placer.Placer` or `thinkorplace.v2.placer.Placer` and call
`place(benchmark)`. The entries under `submissions/thinkorplace*/placer.py` remain
checkout compatibility adapters. The package requires the canonical
`macro_place` infrastructure and its external PlacementCost dependency, plus
benchmark files discoverable through `BENCH_ROOT` or the standard external
benchmark directories. It has no runtime dependency on `experiments/` or
`writeup/`. Setuptools includes `thinkorplace*` alongside `macro_place*`.

V2 uses spawned lane processes, a shared kernel monotonic deadline, atomic
validated checkpoints, and canonical-best selection among valid candidates
produced in that run. Use `lanes="A"` for the A baseline or `lanes="ABC"` for
the operating ensemble. A fresh `checkpoint_dir` preserves intermediate phase
artifacts. Contention and deadlines can change which phases finish; there is
no guarantee of improvement over a separately run A baseline. If no usable
checkpoint exists, the run fails explicitly.

V1's `budget_seconds` controls phase allocation. Some inner operations remain
non-preemptible, so strict wall limits require an outer process supervisor
that terminates the process and can recover its validated phase checkpoints.
The publication runner supplies this boundary for both versions.

V2 phase artifacts record the nominal `rng_seed`, `effective_rng_seed` used by
the initializer, `initializer_attempt` (one-based), and `initializer_method`.
The preserved SDF wrapper uses seed 42 regardless of the lane seed; center
initialization has no effective seed. Random and DPO initializers use the lane
seed plus 100 per retry. These fields describe initialization, not independent
randomness in every downstream phase. The publication runner hashes the saved
coordinates and artifact files for evidence.

Runtime validity uses the canonical structural validator and an equivalent
vectorized hard-macro endpoint-box overlap predicate. It preserves the
canonical tensor dtype and rounding, including float32 touching-edge cases.
The publication worker also runs the original full canonical validator.

`phase_details` separates execution (`attempted`, `completed`, `skipped`,
`failed`) from selection (`accepted`, or null before selection is known).
`reason` and `error` explain skips and failures; initializer retries may include
`attempt_errors`. Acceptance means adoption within the pipeline, while the
selected checkpoint/lane identifies the returned placement. A completed phase
can have its result rejected. Disabled phases do not prevent pipeline
completion; budget skips and caught failures do. A worker that returns a valid
placement after a phase failure reports `partial_phase_completion`.
`checkpoint_recovery` identifies a selected checkpoint whose worker did not
return normally. V1 exposes flat phase details; V2 exposes them per lane in
`lane_results` and `phase_completion`. Phase checkpoint files capture the
details known when that checkpoint was written.

Focused validation: `python -m pytest test/test_submission_safety.py
test/test_thinkorplace_package.py`. The package isolation test imports against
pristine final-upstream infrastructure without either research directory.
