# CDLNSSACascadeDPLanePlacer — Cascade + DP basin lane

**Status:** Built but NOT yet validated on `--all`. Use only after
verifying ibm10/14/12/17 B-R0' results land within 5% of cascade.

## What it does

Same as `submissions/cd_lns_sa_cascade/placer.py` but adds a third
init lane (DREAMPlace via Docker subprocess) with **the same polish
budget as E25 and E41** (~660 s CD + ~200 s LNS + ~200 s SA each).

This is the fix to the bug in
`submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py`, which
gave DP only 60 s brief CD cleanup. The "DP can't compete" autopsy
conclusion was based on that unfair comparison.

## Setup (cloud — lambda.ai 129.213.18.245)

DREAMPlace is built in Docker at `~/DREAMPlace_cpu/install` (image
`dreamplace:custom`). CPU build (CUDA build hit nvcc11.0/compute_86
issue; CPU is fast enough — ibm10 DP = 12 s).

```bash
ssh ubuntu@129.213.18.245
cd ~/macro-place-challenge-2026

# Single bench:
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  DP_USE_GPU=0 DP_DOCKER_IMAGE=dreamplace:custom \
  DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
  uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py -b ibm10 --json

# --all (17 IBM):
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  DP_USE_GPU=0 DP_DOCKER_IMAGE=dreamplace:custom \
  DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
  uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json --hypothesis E91hybrid
```

If `DREAMPLACE_ROOT` is not set, the placer falls back to standard
cascade (E25 + E41 plateau + cascade saddle, no DP lane).

## Expected behavior (based on B-R0' partial data)

Per-bench, the plateau pick is best of `{E25, E41, DP-polished}`. On
benches where:

- **DP basin lands in a DP-favored valley:** plateau picks DP-polished.
  Observed on ibm12 (DP+CD alone = 1.149 vs cascade-capped 1.303 =
  −11.8 %) and ibm14 (1.263 vs 1.292 = −2.2 %).
- **SDF or DPO basin is better:** plateau picks E25 or E41 as usual.
  Likely on ibm01, ibm10 where DP+polish remained +1-2 % above cascade
  in B-R0'.

Cascade saddle escape runs on the plateau, so the final placement is
the best of {E25, E41, DP+polish, cascade-saddled-best-plateau}.

## Budget allocation (3300 s total)

- E25 (SDF init + polish): ~1060 s (CD 660 + LNS 200 + SA 200)
- E41 (DPO init + polish + K-joint): ~1155 s (CD 660 + LNS 165 + SA 165 + Kj 165)
- DP (DP + legalize + polish): ~1150 s (DP 30 + legal 60 + CD 660 + LNS 200 + SA 200)
- Cascading saddle: remaining ~800 s

Tight. Need to verify the 60-min cap is respected on partcl EPYC.

## When NOT to run

- Without `DREAMPLACE_ROOT` set: equivalent to standard cascade. Use
  `submissions/cd_lns_sa_cascade/placer.py` directly.
- On benches where E91 B-R0' showed DP+polish > cascade by >5 % (none
  observed so far).

## Data + diagnostic

See `experiments/E91_dp_full_polish/`:
- `SUMMARY.md` — table of B-R0' results across benches
- `NEXT_STEPS.md` — decision tree for follow-up work
- `check_results.sh` — pull cloud results + summarize
