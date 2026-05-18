# Submission

**Entry placer:** [`cd_lns_sa_cascade_dp_lane/placer.py`](cd_lns_sa_cascade_dp_lane/placer.py) (`CDLNSSACascadeDPLanePlacer`)

## Run inside partcl `eval_docker`

The placer has cross-directory imports (it uses other lanes under `experiments/`
and `submissions/`). To make those resolve, mount the **whole repo** alongside
the placer, then point the launcher at it:

```bash
# From the partcl repo (with eval_docker/ checked out):
./eval_docker/run_eval.sh thinkorplace \
    /path/to/macro-place-challenge-2026/submit/placer.py \
    /path/to/macro-place-challenge-2026 \
    /path/to/macro-place-challenge-2026/submit_deps/dreamplace_install
```

`submit/placer.py` is a thin launcher that adds `/submission/repo` to
`sys.path`, then delegates to `cd_lns_sa_cascade_dp_lane`. The third mount
arg makes a bundled DREAMPlace install available at
`/submission/dreamplace_install`; the placer auto-discovers it (no
`DREAMPLACE_ROOT` env var needed).

## Run outside Docker (development)

```bash
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --ng45 --json
```

If `DREAMPLACE_ROOT` env var is set to a compatible DREAMPlace install
(or if `submit_deps/dreamplace_install/` exists in this repo), the
placer adds a third "DP-polished" init lane alongside SDF and DPO,
landing at IBM 1.0665 / NG45 0.6809 (verified 2026-05-14). Without
DREAMPlace, the placer falls back to two lanes (Option A behavior:
IBM 1.0782 / NG45 0.6810).

## Layout

- `cd_lns_sa_cascade_dp_lane/placer.py` — main 3-lane placer
- `cd_lns_sa_cascade/placer_adaptive.py` — 2-lane fallback (Option A)
- `cd_lns_sa/placer.py`, `cd_lns_sa_cascade/placer.py` — component lanes (E25, cascade saddle)
- `../submit/placer.py` — thin launcher for eval_docker
- `../submit_deps/dreamplace_install/` — bundled DREAMPlace (when present)
