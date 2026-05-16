# PATH B B-R0' — Quick TLDR (2026-05-12 session)

## What happened

Original PATH B was killed 2026-05-11 on basin-only data: "DP basin
1.25 >> cascade-capped 1.08 on ibm10, structural mismatch, can't be
bridged." I verified the test that was actually run: it was DP-without-
polish vs cascade-with-polish. The fair comparison (DP + same polish
budget as cascade gives its SDF/DPO inits) was never done.

I ran the fair test on cloud (lambda.ai 129.213.18.245). Driver:
`experiments/E91_dp_full_polish/code/dp_full_polish.py`.

## Results (4 of 5 still running mid-pipeline at session end)

| Bench | DP basin | + legal | + CD | + LNS | + SA | + cascade | Cascade ref | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ibm01 | 1.011 | 1.047 | 0.876 | 0.875 | 0.875 | **0.862** | 0.85 (uncapped) | +1.4 % |
| ibm09 | 0.954 | 0.955 | 0.799 | 0.796 | running | | n/a | ~ |
| ibm10 | 1.215 | 1.444 | 1.105 | 1.099 | 1.099 | running | 1.0775 (capped) | +2.0 % |
| ibm12 | 1.442 | 1.477 | 1.149 | 1.147 | 1.146 | running | 1.3031 (capped) | **−12.0 %** |
| ibm14 | 1.494 | 1.523 | 1.263 | 1.261 | 1.261 | running | 1.2919 (capped) | **−2.4 %** |
| ibm17 | 1.543 | 1.572 | 1.313 | 1.312 | running | | 1.4546 (capped) | **−9.7 %** |

**Headline:** 3 of 4 hardest autopsy benches show DP+CD-LNS-SA (no
cascade saddle yet) already *beats* cascade-capped, often by large
margins (ibm12 −12 %, ibm17 −10 %). The original "structural
mismatch" claim is empirically falsified.

## Files

- `SUMMARY.md` — full results table + interpretation
- `NEXT_STEPS.md` — decision tree
- `check_results.sh` — pull cloud results + summarize
- `submissions/cd_lns_sa_cascade_dp_lane/placer.py` — hybrid placer ready
  to run `--all` (E25 + E41 + DP lanes + cascade saddle)
- `experiments/E91_dp_full_polish/code/dp_full_polish.py` — B-R0' driver
- `experiments/E92_dp_tilos_rudy/code/diff_rudy.py` — B-R1 prototype
  (differentiable RUDY for adding congestion to DP's obj_fn; gated on
  hybrid result; lower priority given how well stock DP+polish does)

## B-R4 status

FALSIFIED. Cascade-init → DP perturb (iter 10 or 100, lr 0.005) →
proxy diverges, every macro pair overlaps. DP's gradient destabilizes
canonical-optimal placements. Driver kept for future retry with
lr=1e-5 or non-Nesterov optimizer.

## Implication for first place

DP+polish lifts ibm12 and ibm17 by 9-12 % vs cascade-capped. Applied
across 17 benches via hybrid placer, expected `--all` aggregate lift
of ~3-5 % vs cascade-capped (current 1.137). Hybrid target: ~1.08-1.10
aggregate. Leaderboard top ~1.01. Still gap, but closer than current.

For actual first place: need either (1) better-than-stock DP via B-R1
(custom loss), or (2) a different method we haven't tried, or (3) the
leaderboard top doing nothing we can't replicate.

## To resume

```bash
bash experiments/E91_dp_full_polish/check_results.sh  # snapshot status
```

When all benches complete, evaluate hybrid:

```bash
ssh ubuntu@129.213.18.245
cd ~/macro-place-challenge-2026
DP_DOCKER_IMAGE=dreamplace:custom DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
  DP_USE_GPU=0 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json --hypothesis E91hybrid
```
