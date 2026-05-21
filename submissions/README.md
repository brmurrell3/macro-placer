# thinkorplace submissions

## `thinkorplace/` — 2026-05-13 submission

```bash
uv run evaluate submissions/thinkorplace/placer.py --all --json
```

Or via Docker:

```bash
./eval_docker/run_eval.sh thinkorplace submissions/thinkorplace/placer.py
```

## `thinkorplace-v2/` — 2026-05-21 submission

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py --all --json
```

Or via Docker (this is the default entry point the root `placer.py` launcher points to):

```bash
./eval_docker/run_eval.sh thinkorplace placer.py
```

## `thinkorplace-v3/` — 2026-05-21 follow-up

Adaptive selector that dispatches to v2-extCD or E138 based on the
benchmark's hard-macro density. Offline projected EPYC --all = 0.98143
vs v2-extCD's 0.98387 (−0.25 %).

```bash
uv run evaluate submissions/thinkorplace-v3/placer.py --all --json
```

## `thinkorplace-v3-ensemble/` — 2026-05-21 final submission

2-lane PARALLEL ensemble (v2-extCD + E138 saddle escape) with subprocess
plateau-pick. Each bench runs both lanes in parallel subprocesses; master
picks the canonical-better placement. The plateau-pick provides strict
≥ v2-extCD safety on every bench (never worse than baseline).

Verified EPYC ibm17 smoke: **1.17653** (vs v2-extCD 1.18269 = −0.52%,
strongest single-bench lift since v2-extCD itself).

Offline projection EPYC --all from oracle of v2-extCD + E138 per-bench
runs: **0.98064** (−0.33 % vs v2-extCD).

**Currently the root `placer.py` launcher's target.**

```bash
uv run evaluate submissions/thinkorplace-v3-ensemble/placer.py --all --json
```
