# thinkorplace submissions

## `thinkorplace/` — 2026-05-13 submission

Original cascade placer.

```bash
uv run evaluate submissions/thinkorplace/placer.py --all --json
```

Or via Docker:

```bash
./eval_docker/run_eval.sh thinkorplace submissions/thinkorplace/placer.py
```

## `thinkorplace-v2/` — 2026-05-22 submission

3-lane parallel ensemble (V4 + Gaussian density base; lane B adds bounded
Hessian saddle escape; lane C adds K=50 Hungarian joint permutation).
Master spawns 3 subprocesses per bench and picks the canonical-best.

Self-contained: all our experiment code lives in `submissions/thinkorplace-v2/lib/`.
Only `macro_place.*` (competition infra) and standard deps are imported.

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py --all --json
```

Via Docker (root `placer.py` launcher points here):

```bash
./eval_docker/run_eval.sh thinkorplace placer.py
```
