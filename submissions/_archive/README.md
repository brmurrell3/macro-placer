# Submissions archive

Development placers from April to May 2026, kept because the experiment
records in [`../../experiments/`](../../experiments/) and the decision records
in [`../../docs/decisions/`](../../docs/decisions/) cite them. None of the
current entries imports from this folder.

| Path | Contents |
|---|---|
| `cd_only/`, `cd_adaptive/`, `cd_lns_gridbin/`, `cd_lns_sa*/` | The coordinate-descent and large-neighbourhood-search line that became v1. |
| `e110_*/` to `e127_*/` | Variants from the descent experiments that became v2. |
| `e155_parallel_ensemble/`, `e157_adaptive_lane_selector/` | Late-May follow-ups to v2, named `thinkorplace-v3-ensemble` and `thinkorplace-v3` at the time. Unrelated to the current v3. |
| `thinkorplace-v2-singlepipeline/` | v2 as a single pipeline, before the lane supervisor. |
| `falsified/` | Variants whose experiments failed. |
| `will_seed/` | The simulated-annealing seed placer the project started from. |
| `thinkorplace-v2-lib/` | The v2 sources before they were extracted into [`thinkorplace/v2/`](../../thinkorplace/v2/). [`thinkorplace/MIGRATION.json`](../../thinkorplace/MIGRATION.json) records each one under its pre-move path `submissions/thinkorplace-v2/lib/`. |
| `common/` | Shared modules of the competition-era placers, imported as `submissions._archive.common`. `MIGRATION.json` records them under their pre-move path `submissions/common/`. |

Code that imports an archived placer uses `submissions._archive.<name>`.
