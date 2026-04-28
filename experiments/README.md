# Experiments

Per-experiment manifest folders. Every experiment that has produced (or could
produce) a row in `results/experiment_log.jsonl` lives here so a future reader
can answer: *what is this, what is its status, why was it killed or promoted?*

## Naming

`E<NN>_<short_snake_case>/` — e.g. `E12_grid_bin_lns/`, `E16_tight_threshold/`.
Use the same `E<NN>` id used in `results/experiment_log.jsonl` and
`writeup/evidence.md`. Generation-level entries that don't have an `E<NN>`
(polyhedra, the DPO era) use a descriptive snake_case name instead.

## Layout

Every experiment folder contains at minimum:

```
E<NN>_<name>/
├── manifest.md          # required: status, hypothesis, kill gate, outcome
├── code/                # optional: placer source if it lives here
├── notes.md             # optional: free-form working notes
└── results/             # optional: JSON snapshots, partial runs
```

`code/` is only present for *active* experiments whose source still exists in
the repo. *Graduated* experiments leave the `code/` directory empty (or absent)
and set `graduated_to:` in the manifest to the path the implementation was
promoted to (e.g. `macro_place/incremental_evaluator.py`). *Falsified*
experiments whose code was removed during a cleanup leave `code/` absent and
the manifest cites the git ref or removal commit.

## Lifecycle

The `status:` field in the manifest's YAML frontmatter is the single source of
truth. Allowed values:

| Status | Meaning |
|---|---|
| `proposed` | Documented, not yet implemented |
| `in_progress` | Code exists; awaiting result or kill |
| `graduated` | Promoted into the library / current champion. `graduated_to:` is set. |
| `falsified` | Kill gate triggered. Manifest records the data that killed it. |
| `superseded` | Replaced by a newer experiment. `superseded_by:` is set. |
| `marginal` | Improved over baseline but below the strict kill gate; kept as a sensitivity datapoint, not promoted. |

Falsified experiments are kept *forever*. The manifest is the historical
record. Do not delete the folder — even after the code is gone, the manifest
is what stops the team from re-running the same dead end.

## Running an experiment

```bash
# Fast subset (4 benchmarks, ~10 s)
uv run evaluate experiments/E<NN>_<name>/code/<placer>.py --fast --json --hypothesis E<NN>

# Full validation (17 benchmarks, ~40 s for cheap placers; hours for CD/LNS)
uv run evaluate experiments/E<NN>_<name>/code/<placer>.py --all --json --hypothesis E<NN>

# Single benchmark
uv run evaluate experiments/E<NN>_<name>/code/<placer>.py -b ibm10 --hypothesis E<NN>
```

Always pass `--hypothesis E<NN>` so the row written to
`results/experiment_log.jsonl` is attributable.

## Lifecycle decisions (mirroring `CLAUDE.md`)

- `--fast` fails the gate → tweak parameters, try next variant.
- `--fast` passes → run `--all`.
- `--all` avg `< 1.10` → graduate. Stop and surface to human.
- `--all` avg `< 1.05` → new champion. Stop immediately.
- Best score hasn't improved by `>2%` after N variants → kill the hypothesis.

Update the manifest (`status`, `decided`, `outcome`, `champion_delta`) the
moment a status-changing decision is made.

## Template

Copy `_template/manifest.md` for a new experiment. See `_template/README.md`
for the per-field convention.
