# Hybrid Cascade + DP-lane Placer (scaffold)

This is a sketch — only build if E91 B-R0' results say DP is competitive.

## What it does

Same as `submissions/cd_lns_sa_cascade/placer.py` but with a third init
lane: DP global placement via Docker subprocess. All three init lanes
(SDF, DPO, DP) get equal polish budget (~20% each of total budget for
CD, 6% each for LNS, 6% each for SA). Best-of-3 plateau pick, then
cascading saddle escape.

## Why the falsified `cd_lns_sa_hessian_dp/placer.py` failed

It gave DP only 60s `min_time=30, hard_cap=60` brief CD cleanup vs
E25/E41's ~660s each. The plateau pick was therefore always E25 or
E41 (DP basin 1.21 vs polished E25 ~0.88 isn't a fair fight). The
revised version below gives DP equal polish.

## Adapting from cd_lns_sa_cascade/placer.py

Change Phase 3 between E41 and cascade saddle:

```python
# Phase 3: DREAMPlace (optional, with full polish budget)
dp = None
dp_proxy = float('inf')
if deadline is not None and (deadline - time.time()) < (B * 0.35):
    log(f"  Phase 3: DP SKIPPED (only {deadline - time.time():.0f}s left)")
else:
    log("  Phase 3: DREAMPlace (with full polish)")
    dp_raw = _try_run_dreamplace(benchmark, plc, log)
    if dp_raw is not None:
        dp_polished, _stats = _full_polish_pipeline(
            dp_raw, benchmark, plc,
            cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.06,
            sa_budget_s=B * 0.06, log=log,
        )
        dp = dp_polished
        dp_proxy = float(compute_proxy_cost(dp, benchmark, plc)["proxy_cost"])

# Phase 4: plateau pick = best of {E25, E41, DP}
candidates_plateau = [(e25_proxy, e25, "E25"), (e41_proxy, e41, "E41")]
if dp is not None:
    candidates_plateau.append((dp_proxy, dp, "DP"))
candidates_plateau.sort(key=lambda c: c[0])
plateau, plateau_label, plateau_proxy = candidates_plateau[0][1], candidates_plateau[0][2], candidates_plateau[0][0]
```

## Budget allocation under 3300s total

- E25: CD 660s + LNS 200s + SA 200s = 1060s
- E41: CD 660s + LNS 165s + SA 165s + Kj 165s = 1155s
- DP: DP 30s + legalize 60s + CD 660s + LNS 200s + SA 200s = 1150s
- Cascade saddle: remaining ~800s

Tight. Worth measuring whether E25 + E41 + DP all running serially fits
in 55 min on partcl EPYC. If not, reduce per-lane CD cap to ~500s.

## Code reuse

The polish pipeline is implemented in
`experiments/E91_dp_full_polish/code/dp_full_polish.py::run_full_polish_on_init`
— factor that out into a module the hybrid placer can import.

The DP Docker subprocess is in `experiments/E91_dp_full_polish/code/dp_full_polish.py::run_dp_docker`
— or use the cleaner version in `submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py::_try_run_dreamplace` (which has the proper graceful fallback if DREAMPLACE_ROOT isn't set).

## Run command (cloud)

  DP_DOCKER_IMAGE=dreamplace:custom DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    uv run evaluate submissions/cd_lns_sa_cascade_dp_lane/placer.py --all --json

