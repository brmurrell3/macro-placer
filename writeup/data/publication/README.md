# Publication evidence inventory

This directory separates historical competition records, bounded post-competition
measurements, and source snapshots. Each class of evidence has a different claim
boundary. A checksum establishes the bytes that were measured or archived; it
does not turn historical results into measurements of the corrected runtime or
extend a small diagnostic to unmeasured benchmarks.

## Historical records and figures

`raw/` contains preserved copies of original result JSON. The three files at its
top level are historical full-IBM records for the v1 lineage, an earlier V3-based
v2, and the Gaussian ancestor. `raw/cascade/` contains the saved E74 and E84
per-design JSON stage records. `raw/placements/` contains 34 frozen PyTorch
placement states (E74 and E84 for each of the 17 IBM designs, 596 KiB total).
Their source paths, frozen paths, and SHA-256 values are recorded in
`historical_summary.json`. The frozen tensors make the placement figure and
interactive explorer reproducible when the original ignored experiment outputs
are absent; the corresponding frozen JSON sidecars supply the recorded stage
statistics. A clean-source verification rebuilt four historical figures and
reproduced four LaTeX tables byte-for-byte without the ignored `results/` and
`experiments/` outputs or Git history. These inputs are summarized in:

- `historical_summary.json` and `per_benchmark.csv`;
- `../../latex/generated/historical_table.tex`,
  `historical_per_benchmark.tex`, `component_table.tex`, and
  `cascade_table.tex`;
- `../../latex/figures/historical_scores.pdf`,
  `historical_components.pdf`, `cascade_gains.pdf`, and
  `cascade_placement_ibm01.pdf`.

Regenerate the preserved copies, summaries, tables, static figures, and the
competition source archive from the repository root:

```sh
.venv/bin/python scripts/freeze_publication_evidence.py
```

The script refuses to replace a frozen raw record with different bytes. If an
original ignored experiment result is absent, it reads the already frozen raw
JSON or placement tensor instead. The historical files have incomplete
configuration provenance and do not establish the score of the current
operating ensemble.

## Matched evaluator microbenchmarks

The three `evaluator_*.json` files are complete CPU component microbenchmarks.
They apply the same prescribed moves through incremental and canonical component
evaluation, verify numerical agreement, and time overlap checking separately.
They are kernel samples, not end-to-end placement speedups or suite-wide timing
estimates. Each record contains its protocol, environment, benchmark-file hashes,
measurement-source hashes, and runtime-package hashes.

The stored protocols can be reproduced into new, non-conflicting paths with:

```sh
.venv/bin/python scripts/measure_publication_evaluator.py \
  --benchmarks ibm01 --moves 8 --repetitions 3 --threads 1 --seed 42 \
  --macro-kind all \
  --output writeup/data/publication/reproductions/evaluator_ibm01_mac.json

.venv/bin/python scripts/measure_publication_evaluator.py \
  --benchmarks ibm10 --moves 3 --repetitions 2 --threads 1 --seed 42 \
  --macro-kind all \
  --output writeup/data/publication/reproductions/evaluator_ibm10_mac.json

.venv/bin/python scripts/measure_publication_evaluator.py \
  --benchmarks ibm10 --moves 3 --repetitions 2 --threads 1 --seed 42 \
  --macro-kind hard \
  --output writeup/data/publication/reproductions/evaluator_ibm10_hard_mac.json
```

The measurement script deliberately refuses an existing output path. Regenerate
the checked-in summary and LaTeX section from the three complete primary records:

```sh
.venv/bin/python scripts/render_publication_measurements.py
```

This writes `evaluator_measurement_summary.json` and
`../../latex/generated/evaluator_results.tex`.

## Frozen-endpoint curvature diagnostic

For report formatting, render the existing record without rerunning the probe:

```sh
.venv/bin/python scripts/render_publication_curvature.py
```

This updates the LaTeX fragment only. The measurement command below is for
reproducing the diagnostic, not for editing its presentation.

`curvature_diagnostics_ibm01.json` evaluates the historical E74 smooth surrogate
at the frozen `raw/placements/hessian_ibm01.pt` endpoint. It imports E74 and its dependencies from
`competition-source.tar`, checks that archive against `source_snapshot.json`, and
hashes the saved state, placement tensor, source members, and benchmark inputs.
It does not rerun optimization or legalization. Its finite differences probe a
piecewise surrogate; the observed disagreement does not isolate branch crossing
from numerical or operator issues. The record does not claim a classical Hessian
or a multi-benchmark saddle ablation.

Regenerate the bounded diagnostic and LaTeX summary:

```sh
.venv/bin/python scripts/diagnose_publication_curvature.py \
  --output writeup/data/publication/curvature_diagnostics_ibm01.json \
  --tex-output writeup/latex/generated/curvature_diagnostics.tex \
  --lanczos-hvp-cap 14 --lanczos-deadline-seconds 105 --torch-threads 1
```

## Referee follow-up checks

The separate September 7 referee checks are retained in
`curvature_followup_ibm01.json`, `e74_e84_canonical_rescore_20260907.json`, and
`e74_e84_geometry_crosscheck_20260907.json`. The curvature follow-up uses
float64 arithmetic, guarded directions, branch traces, and weighted-component
probes. It narrows the explanation of the original disagreement without
establishing a classical Hessian for the composite at the sampled endpoint.
The formatting command above reads both curvature records.

Canonical rescoring covers all 34 frozen E74/E84 endpoints. All have zero hard
overlaps; only 22 pass strict geometry validation. Twelve fail canvas bounds by
at most 7.63e-6 micrometres. Their coordinates remain unchanged, so the aggregate
scores include invalid placements. The independent geometry cross-check agrees
on all outcomes. The producers are `scripts/rescore_publication_e74_e84.py` and
`scripts/audit_publication_e74_e84_geometry.py`; the response and source hashes
are recorded under `docs/reviews/` and `verification/referee-response-20260907.json`.

## Topology diagnostic

`topology_diagnostic.json` isolates the hierarchical macro-owner parsing change
on IBM01 and ariane133. It compares two extractor functions that differ only in
the ownership expression and performs no placement optimization, scoring, or
downstream physical-design run. The JSON embeds both function bodies and hashes
the functions and benchmark inputs.

In a clean checkout where the evidence path does not yet exist, regenerate it
with:

```sh
.venv/bin/python scripts/measure_publication_topology.py
```

The script refuses to overwrite the existing evidence file. Preserve the primary
record and use a separate checkout or a newly named diagnostic for a changed
implementation.

## SDF topology diagnostic

`sdf_topology_diagnostic.json` is the immutable controlled record for the final
SDF hard-macro graph correction. It preserves the before and corrected extractor
source, source and input hashes, and measurements on IBM01 and ariane133. The
corrected extractor exactly preserves IBM01's recorded 644-edge graph and changes
ariane133 from an empty graph to 859 recorded edges. Its scope is graph
extraction only: it makes no placement, scoring, transfer, or downstream
physical-design claim.

Replay the record against the hashed inputs and active corrected extractor
without rewriting it:

```sh
.venv/bin/python scripts/measure_publication_sdf_topology.py --verify-record
```

The evidence-producing script is preserved independently at
`raw/measure_publication_sdf_topology.py`; its SHA-256 is embedded in the record
and checked by replay.

## Interactive placement explorer

The explorer in `../../interactive/` visualizes full-precision saved E74 and E84
placement tensors while emitting rounded display coordinates. Regenerate its
data and embeddable fragment with:

```sh
.venv/bin/python scripts/build_publication_explorer.py
```

This writes `../../interactive/placement-data.json` and
`placement-comparison.fragment.html`. The checked-in
`placement-comparison.html` is the standalone exported wrapper around the
fragment. The repository has no separate deploy command or deterministic script
for rebuilding that outer wrapper. Open the checked-in export directly or serve
it as an iframe; no website deployment is part of this evidence. With Playwright
and Chromium installed, verify the standalone export with:

```sh
PLAYWRIGHT_CHANNEL=chrome node scripts/check_publication_explorer.cjs
```

The browser check writes screenshots and `checks.json` under
`../../interactive/qa/`. See `../../interactive/README.md` for the interaction
and display-precision limits.

## Source snapshots and hash boundaries

`competition-source.tar` is the historical Python source closure exported from
the commit named in `source_snapshot.json`. It covers the listed competition
infrastructure, submission, experiment, and archived helper files. It excludes
benchmark inputs and the external TILOS checkout; consumers must obtain those
separately. Rebuild it as part of the historical command above.

`corrected-runtime-*.tar.gz` is a deterministic archive of the corrected
post-competition runtime. `corrected_runtime_snapshot.json` maps every included
path to a SHA-256 digest, hashes the archive, records the base and canonical
upstream commits, and names the TILOS commit. Its scope includes the extracted
runtime package, selected canonical infrastructure, compatibility entries,
runner, study configurations, and focused tests. Benchmark inputs and external
TILOS contents remain outside the archive and are hashed or revision-pinned by
the run records. Create a new corrected-runtime identity after an intentional
source change with:

```sh
.venv/bin/python scripts/freeze_publication_runtime.py
```

The historical raw JSON and `competition-source.tar` remain original evidence.
The corrected archive documents later code and must not be used to retroactively
attribute corrected behavior to those historical runs.

The final corrected source identity is
`5aa4c719cf18dc2508d9493e581fe08cc54729a88a7f4166383ea15b820ab3e3`,
recorded by `corrected_runtime_snapshot_5aa4c719cf18.json` and archived as
`corrected-runtime-5aa4c719cf18.tar.gz`. The focused-test receipt
`verification/final-tests-20260906.json` binds that archive and source identity
to its hashed log and records 64 passing tests with exit code zero. Its stated
scope is five named test modules, not a full-suite solver-quality result.

## Limited-budget pilot

The primary current pilot is `../../runs/pilot-20260906-final/`. All 12 planned
records are present: five are valid, usable, and study-eligible operating
results, while seven are errors. One valid result completed its full pipeline;
one returned after partial phases and three recovered checkpoints. There were no
external timeouts. Its manifest, per-run requests, atomic result records, logs,
placements, and checkpoint hashes are the primary evidence. Regenerate the
complete all-outcome JSON summary and LaTeX tables with:

```sh
.venv/bin/python scripts/render_publication_pilot.py
```

The renderer fails on an incomplete matrix by default. Its
`--allow-incomplete` option is only for an explicitly labelled interim artifact.
It retains errors, timeouts, partial pipelines, and checkpoint recoveries, and
only reports a matched pilot mean when every planned benchmark for that variant
has an eligible finite result. It also reports saved initializer and legalized-
descent coordinate-hash equality across ABC lanes and against standalone A,
including missing checkpoint coverage. These hashes show that nominal lane seed
labels alone do not establish distinct basin starts; they measure no prospective
shared-prefix speedup. The pilot uses a 180-second solver budget and is finite
engineering evidence; it is not an extrapolation to the predeclared publication
budget.

## Post-pilot repair validation

`../../runs/repair-20260906/` is a separate 12-cell engineering validation under
the repaired source identity. Nine records are valid, usable, and eligible;
three IBM17 records reached their external timeout. Two results completed their
full pipeline, four returned after partial phases, and three recovered
checkpoints. These observations are not pooled with the primary pilot and do not
form a causal quality comparison. Regenerate their separate artifacts with:

```sh
.venv/bin/python scripts/render_publication_pilot.py --profile repair
```

## Final ariane133 integration

`../../runs/final-ng45-integration-20260906/` contains the three final ariane133
variant checks under corrected runtime archive `5aa4c719cf18`. All three results
are valid, usable, and study-eligible. V2 Lane A completed its pipeline; v1 and
v2 ABC returned valid results with partial phase completion. The profile used a
60-second solver budget, 120-second external cap, three concurrent jobs, and
three CPU threads per job. The v2 entries used five analytical steps and a
zero-second v2 CD cap; v1 retained its separate budget-derived phases.

`final_ng45_integration.json` and
`../../latex/generated/final_ng45_integration.tex` summarize the immutable run
records and their hashes. This is bounded integration evidence, not a quality
comparison, and it is not pooled with either 12-cell matrix. The predeclared
full study remains incomplete and no complete publication-budget result is
claimed.
