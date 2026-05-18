# Submission

**Entry placer:** [`cd_lns_sa_cascade_dp_lane/placer.py`](cd_lns_sa_cascade_dp_lane/placer.py) (`CDLNSSACascadeDPLanePlacer`)

## Run

```bash
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --ng45 --json
```

Per-bench wall cap: 60 minutes. Output: zero overlaps on all 17 IBM + 4 NG45.

## Optional DREAMPlace lane

If `DREAMPLACE_ROOT` environment variable is set to a DREAMPlace install path,
the placer enables a third initialization lane (DREAMPlace GP + full polish)
alongside the SDF and DPO lanes. Without DREAMPLACE_ROOT, the placer falls back
to the two-lane configuration ([`cd_lns_sa_cascade/placer_adaptive.py`](cd_lns_sa_cascade/placer_adaptive.py)) which has no
external dependencies.
