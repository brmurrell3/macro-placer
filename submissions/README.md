# thinkorplace submissions

## `thinkorplace/` — 2026-05-13 submission

```bash
uv run evaluate submissions/thinkorplace/placer.py --all --json
```

Or via Docker:

```bash
./eval_docker/run_eval.sh thinkorplace submissions/thinkorplace/placer.py
```

## `thinkorplace-v2/` — upcoming submission

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py --all --json
```

Or via Docker (this is the default entry point the root `placer.py` launcher points to):

```bash
./eval_docker/run_eval.sh thinkorplace placer.py
```
