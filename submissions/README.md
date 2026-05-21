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
