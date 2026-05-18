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
| 3-lane (with DREAMPlace) | **1.06650** | **0.68086** | 0 |
| 2-lane fallback | 1.07820 | 0.68102 | 0 |

See [`../SUBMISSION.md`](../SUBMISSION.md) for full details, optional
DREAMPlace mount instructions, and resource expectations.
