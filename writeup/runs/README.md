# Publication run directories

Run directories are append-only evidence. Keep failed and aborted attempts: they
document launch integration, timeout behavior, and the reasons a later run was
needed. A valid returned checkpoint may remain study-eligible while its pipeline
is partial or recovered; `pipeline_completed`, `solver_outcome`, phase details,
and worker errors must remain visible beside validity.

## Primary pilot

`pilot-20260906-final/` is the sole primary 12-cell limited-budget pilot. It uses
the frozen corrected runtime and the checked-in pilot configuration. All 12
manifest cells have immutable `result.json` records. Five returned valid, usable,
study-eligible operating results and seven ended in error; there were no external
timeouts. Only one of the five valid results completed the full pipeline. One
records partial phase completion and three record checkpoint recovery. Generate
the all-outcome publication summary with:

```sh
.venv/bin/python scripts/render_publication_pilot.py
```

That postprocessor requires all 12 records by default, retains failures and
recoveries, prevents selective averages, and summarizes measured saved-prefix
hash equality without inferring independent starts from nominal lane seeds. The
directory is limited-budget engineering evidence rather than an official-budget
study.

## Post-pilot repair validation

`repair-20260906/` contains 12 identity-matched records under the repaired source
snapshot and a three-job concurrent resource profile. Nine are valid, usable,
and study-eligible. The other three are external timeouts, all on IBM17; there
are no error-status records. Two valid results completed the full pipeline, four
record partial phase completion, and three record checkpoint recovery. Render
the separate evidence with:

```sh
.venv/bin/python scripts/render_publication_pilot.py --profile repair
```

The primary and repair directories therefore contain 24 preserved records, but
their results and aggregates remain separate. The repair run uses a different
source identity and three concurrent jobs, so it is not pooled with the primary
pilot or presented as a causal quality comparison.

## Final ariane133 integration

`final-ng45-integration-20260906/` contains the three final ariane133 variant
checks under corrected runtime archive `5aa4c719cf18`. All three completed with
valid, usable, study-eligible placements. V2 Lane A is the sole
pipeline-complete result; v1 and v2 ABC record partial phase completion. The
bounded profile allocated a 60-second solver budget and 120-second external cap
to each run, with three concurrent jobs and three CPU threads per job. The v2
entries used five analytical steps and a zero-second v2 CD cap. V1 retained its
own budget-derived phase settings.

The immutable aggregate is `../data/publication/final_ng45_integration.json`,
with its table at `../latex/generated/final_ng45_integration.tex`. These three
records are integration evidence for the final SDF graph correction. They are
not a quality comparison and are not pooled with the primary or repair matrix.
The associated final source identity and 64-test receipt are recorded in
`../data/publication/corrected_runtime_snapshot_5aa4c719cf18.json` and
`../data/publication/verification/final-tests-20260906.json`.

## Preserved preflights and aborted attempts

- `preflight-20260906/` is the early three-run IBM01 smoke. Its placements were
  valid, but it predates the final runtime extraction and outcome schema. Use it
  only as engineering history.
- `pilot-20260906-12run/` is the first 12-cell launch attempt. Every worker failed
  because the newly extracted package was absent from the worker import path.
  The complete error matrix is retained as launch-failure evidence.
- `pilot-20260906-12run-r2/` was stopped during its first cell after review found
  disagreement in the overlap predicate and inaccurate phase-completion
  reporting. `ABORTED.json`, the manifest, request, and logs record why it was
  stopped. It is not an aggregate result.

These directories must not be merged with the primary pilot or used to replace a
missing or poor primary result.

## Integration and package smokes

- `runner-integration-20260906-v2a/` is the first end-to-end supervisor/worker
  smoke after fixing direct package imports.
- `runner-integration-20260906-v2a-r2/` repeats that bounded IBM01 Lane-A test
  under the final source identity and phase schema. It verifies worker launch,
  result export, canonical validation, and checkpoint hashing.
- `../data/publication/package-smoke/` is the isolated-package IBM01 Lane-A
  smoke against pristine final-upstream infrastructure. Its provenance manifest
  hashes the result, log, and checkpoints. It is a packaging check, not a solver
  quality comparison.

Smokes cover one design with reduced work and do not contribute observations to
the 12-cell pilot.

## Further targeted evidence

Run any further repair or diagnostic in a newly named directory. Do not overwrite a primary
result, append a changed source identity to the primary pilot, or silently
substitute a repair into its aggregate. A targeted repair should name the failed
primary run it investigates, preserve its own manifest and logs, and report its
source identity, resource limits, validity, and pipeline outcome separately.

The predeclared long study configuration remains in `../study/main_ibm17.json`
and has not been completed. Do not infer a full-study result from a smoke,
targeted repair, integration check, or limited-budget pilot.
