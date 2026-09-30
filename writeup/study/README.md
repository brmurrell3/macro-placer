# Publication study runner

The runner starts every benchmark/variant in an isolated process group, applies
the `--timeout-seconds` wall-clock cap outside the solver, and writes an atomic result plus a complete
stdout log. Its manifest keys each run by the benchmark inputs, effective
constructor configuration, repository/code/evaluator identities, device, thread
allocation, and seed. Reusing an output directory resumes an identity-matched
terminal run; pass `--retry-failures` to create another recorded attempt for a
timeout or error.

Run the checked-in 12-run readiness pilot (this is limited-budget engineering
evidence, not the publication study):

```bash
.venv/bin/python scripts/run_publication_study.py \
  --config writeup/study/pilot.json \
  --output writeup/runs/pilot-20260906-12run
```

The pilot covers IBM01, IBM10, IBM17, and ariane133 (NanGate45) with v1, v2 Lane A, and v2
ABC: 12 runs total, one nominal seed, 3 CPU threads, one benchmark job, 500
smooth steps, a 30-second CD phase cap, a 180-second solver budget, and a
240-second external cap. `study_kind` and `study_notes` preserve its
limited-budget status in the manifest. The predeclared three-repetition IBM
study is in `main_ibm17.json` and must only be run after the pilot establishes
realistic runtime and memory requirements:

```bash
.venv/bin/python scripts/run_publication_study.py \
  --config writeup/study/main_ibm17.json \
  --output writeup/runs/main-YYYYMMDD
```

`--jobs` defaults to one. `--threads` is the total CPU thread allocation for one
benchmark run; the v2 solver divides that allocation across its selected lanes.
The external timeout defaults to `--budget-seconds`; set it explicitly when a
small allowance for worker startup and final canonical evaluation is intended.
The main configuration gives each solver 3,300 seconds inside a strict 3,600
second external cap, reserving five minutes for canonical evaluation and
atomic artifact export.
The runner sets common BLAS/OpenMP thread variables before the worker imports
PyTorch. Peak memory is the sampled sum of RSS for members observed in the worker
process group. The result states this scope explicitly because processes that
change process group would not be included.

Use exact constructor names for other settings:

```bash
--placer-param v2-a:lr_frac=0.004 \
--placer-param v1:periphery_alpha=0.02
```

Values are parsed as JSON when possible. `--num-steps` and `--cd-seconds` map to
the v2 parameters `num_steps` and `cd_polish_s`; they do not silently alter v1's
several distinct polish budgets. The result records the full effective
constructor dictionary. The known SDF seed 42 is also recorded separately from
lane and search seeds, so repeated runs cannot be described as independent
initializations unless the initializer changes.

Each run directory contains `request.json`, `stdout.log`, `result.json`,
`final.pt` after successful placement export, and one fresh `checkpoints/attempt-NNN/`
directory per attempt. Timeout and
error runs remain in `summary.json` with `valid=false`; overlap count is recorded
separately from full legality (shape, finite values, bounds, fixed preservation,
and hard-macro overlaps).

`valid` describes the exported placement. `usable_operating_result` requires a
valid placement returned within the external cap after the source recheck.
`study_eligible` additionally requires the frozen dependency environment to match.
A legal initializer or intermediate checkpoint recovered after a lane exception
remains in complete-suite aggregates. `pipeline_completed`, `solver_outcome`,
`used_fallback`, `phase_completion`, `degraded_lanes`, and the lane worker error
records expose that degraded execution separately. The checkpoint inventory
hashes both each artifact and its saved coordinate array and records phase and
seed metadata. Use `--fail-on-degraded` for an engineering preflight that should
return nonzero when any requested lane fails; this does not invalidate or discard
a usable checkpoint.
