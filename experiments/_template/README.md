# Experiment template

Copy `manifest.md` to a new `E<NN>_<short_name>/manifest.md` and fill in the
fields. Field conventions:

| Field | Convention |
|---|---|
| `id` | `E<NN>` matching `results/experiment_log.jsonl` and the `--hypothesis` flag |
| `name` | `short_snake_case`, mirrors the directory name suffix |
| `status` | `proposed | in_progress | graduated | falsified | superseded | marginal` |
| `parent` | `null` or the `E<NN>` of the parent experiment in the lineage |
| `created` | `YYYY-MM-DD` you stood the manifest up |
| `decided` | `YYYY-MM-DD` the status became terminal (`graduated/falsified/superseded`); `null` while in flight |
| `champion_at_time` | `avg_proxy` of the reigning `--all` champion when this experiment was created (e.g. `1.1055`) |
| `outcome` | best `--all` `avg_proxy` once the experiment completes (or `partial: <avg>` for an interim snapshot) |
| `champion_delta` | `outcome - champion_at_time`. Negative means improvement. |
| `graduated_to` | path the implementation was promoted to (e.g. `macro_place/incremental_evaluator.py`); only set when `status: graduated` |
| `superseded_by` | `E<NN>` of the successor; only set when `status: superseded` |

The body sections are required. Keep `Hypothesis`, `Method`, `Kill gate`, and
`Generalization check` honest at the time you write them — even if the
experiment later turns out to fail. The retroactive narrative goes in
`Outcome`.

## Naming examples

- `E12_grid_bin_lns/` — proper experiment id from the log.
- `polyhedra_navigation/` — generation-level umbrella that doesn't have a single
  `E<NN>`; the manifest's `id:` field uses the umbrella name.

## Why we keep falsified experiments

A killed experiment is a signpost. Without the manifest, the next person on
this codebase will rediscover the dead end. The manifest's `Outcome` section
exists specifically to stop that.
