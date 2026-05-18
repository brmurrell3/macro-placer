# Submission

See [`../SUBMISSION.md`](../SUBMISSION.md) at the repo root for the
canonical run-it-yourself guide.

**Entry placer:** [`cd_lns_sa_cascade_dp_lane/placer.py`](cd_lns_sa_cascade_dp_lane/placer.py)
(`CDLNSSACascadeDPLanePlacer`).

The repo-root [`../placer.py`](../placer.py) is a thin launcher that
delegates to this placer; it handles the `eval_docker` mount/path issues.

## Direct run (development)

```bash
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --ng45 --json
```

## Verified results

| Mode | IBM | NG45 | Overlaps |
|---|---:|---:|---:|
| 2-lane (bundled CPU DP install or no DP) | **1.07820** | **0.68102** | 0 |
| 3-lane (CUDA-enabled DREAMPlace) | 1.06650 | 0.68086 | 0 |

The bundled `submit_deps/dreamplace_install/` is a CPU-only build; the
placer's GPU-required DP lane fails gracefully and the 2-lane fallback
takes over. Supply a CUDA DREAMPlace via `$DREAMPLACE_ROOT` for the
1.06650 number.

See [`../SUBMISSION.md`](../SUBMISSION.md) for full details and resource
expectations.
