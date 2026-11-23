# Submissions

One folder per version. Each holds a short `placer.py` entry point that the
challenge's `evaluate` command loads, plus a README on the method. The solver
code for all three versions is in the [`thinkorplace/`](../thinkorplace/)
package.

| Entry | Solver code | What it is |
|---|---|---|
| [`thinkorplace-v3/`](thinkorplace-v3/) | [`thinkorplace/v3/`](../thinkorplace/v3/) | Post-competition recipe and the best of the three: many screened seeds, three chains (one warm from the supplied placement), stacked exact search. |
| [`thinkorplace-v2/`](thinkorplace-v2/) | [`thinkorplace/v2/`](../thinkorplace/v2/) | The ranked competition entry (ninth, mean proxy 0.9974): smooth descent, legalisation, exact coordinate descent. The root [`placer.py`](../placer.py) loads it. |
| [`thinkorplace-v1/`](thinkorplace-v1/) | [`thinkorplace/v1/`](../thinkorplace/v1/) | The first version: two starting methods, three local searches and a curvature cascade, about 53 minutes per design. |

Shared modules (the incremental evaluator, coordinate descent, the SDF
initialiser, runtime safety) are in [`thinkorplace/common/`](../thinkorplace/common/).

Evaluate a version on one design:

```bash
uv run evaluate submissions/thinkorplace-v3/placer.py -b ibm01
uv run evaluate submissions/thinkorplace-v2/placer.py -b ibm01
uv run evaluate submissions/thinkorplace-v1/placer.py -b ibm01
```

Other folders:

| Path | Contents |
|---|---|
| [`examples/`](examples/) | Reference placers from the challenge. |
| [`_archive/`](_archive/) | Development placers cited by the experiment records in [`../experiments/`](../experiments/), the pre-extraction v2 sources (`thinkorplace-v2-lib/`) and their shared modules (`common/`). Not used by any entry above. |
