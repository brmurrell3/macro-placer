# Submissions

| Path | Contents |
|---|---|
| [`thinkorplace-v2/`](thinkorplace-v2/) | The v2 entry point. `placer.py` delegates to `thinkorplace.v2` in the [`thinkorplace/`](../thinkorplace/) package; `lib/` keeps the pre-extraction sources that `thinkorplace/MIGRATION.json` records as origins. The root [`placer.py`](../placer.py) loads this entry. |
| [`thinkorplace/`](thinkorplace/) | The v1 entry point. `placer.py` delegates to `thinkorplace.v1`. |
| [`common/`](common/) | Compatibility modules from the competition code: runtime-safety helpers and shared placer modules. `thinkorplace/MIGRATION.json` records which ones the package was extracted from. |
| [`examples/`](examples/) | Reference placers from the challenge. |
| [`_archive/`](_archive/) | Earlier placers, cited by the experiment manifests in [`../experiments/`](../experiments/). |

Evaluate the v2 entry on one design:

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py -b ibm01
```
