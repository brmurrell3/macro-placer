# cd_lns_sa_cascade_dp_multi_lane

**Status:** experimental (UNVALIDATED on `--all`).
Created 2026-05-13. Replaces single-DP with K=4 multi-config DP in the
DP lane of `cd_lns_sa_cascade_dp_lane`.

## What it does

Same as `submissions/cd_lns_sa_cascade_dp_lane/placer.py` (best-of-{E25, E41, DP}
with cascade saddle), but the DP lane:

1. Runs DREAMPlace K=4 times with diverse configs:
   - `baseline_auto` (current single-DP default; matches B-R0')
   - `dw_low` (density_weight = 2e-5, ×0.25)
   - `dw_high` (density_weight = 4e-4, ×5)
   - `seed_2024` (different random seed for basin diversity)
2. Greedy-legalizes each, applies `extended_legalize` fallback for 1-10
   residual overlaps.
3. Selects the basin with lowest `legalize_proxy` (ovl=0 required).
4. Polishes the selected basin via the same CD+LNS+SA pipeline.

## Why

E96 probe (2026-05-13) showed:
- DP basin quality is **highly stochastic** even with the same config
  (same `random_seed`, same target_density). Basin proxy can swing 2×
  between runs (e.g., dw_low on ibm10: probe 1.33, polish 2.48).
- Different benches prefer different configs:
  - ibm10 (high macro density): dw_low or dw_high
  - ibm12 (natural density): baseline_auto
  - ibm14 (low density clamped): dw_low or dw_high
- The legalize step has a big cost on ibm10 (+0.23 proxy) when DP basin
  is crowded — multi-config picks looser DP configs that legalize cleaner.

## Expected lift

vs single-DP B-R0':
- ibm10: 1.395 (new) vs 1.444 legal (B-R0') → polished ~1.04 (new) vs 1.095 (B-R0').
  Expected: **-5%** (and beats cascade-capped 1.0775 by -3%).
- ibm12: tied (baseline_auto already wins).
- ibm14: 1.454 (new) vs 1.523 legal (B-R0') → polished ~1.18 (new) vs 1.243 (B-R0').
  Expected: **-5%** (and beats cascade-capped 1.292 by -8.7%).
- ibm17: probe pending; expected similar pattern.

Aggregate `--all` lift on top of B-R0' hybrid: estimated **-0.5 to -1.5%**.

## Cost

- DP basin generation: K=4 × ~150s = **+600s** wall per bench (vs ~25s for single-DP).
- Budget rebalanced 2026-05-13: DP-lane share 22% → 30%. E25 22% → 18%, E41 24% → 20%.
  Cascade saddle ~28% (was 29%). Total 96%, leaving headroom for project_overlaps overhead.
- DP polish (CD+LNS+SA) gets ~390s within 30% DP lane (after 600s K=4 basin).
- Hybrid total wall: should fit within 3300s budget (~55 min).

## How to run (cloud)

```bash
ssh ubuntu@129.213.18.245
cd ~/macro-place-challenge-2026

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \\
  DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \\
  DP_DOCKER_IMAGE=dreamplace:custom \\
  DP_USE_GPU=0 \\
  DP_NUM_THREADS=8 \\
  uv run evaluate submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py \\
    --all --json --hypothesis E96multilane
```

## Files

- `placer.py` — the placer class
- `README.md` — this file
- Depends on:
  - `experiments/E96_multi_config_dp/code/multi_dp_basin.py` (K-config selector)
  - `experiments/E91_dp_full_polish/code/extended_legalize.py` (legalize fallback)
  - `experiments/E76_dreamplace_integration/code/{tilos_to_bookshelf, bookshelf_to_pt, macro_legalizer}.py`
  - `experiments/E84_cascading_saddle/code/cascading_saddle.py`
  - `experiments/E74_hessian_saddle/code/hessian_saddle.py` (transitively)
  - `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`
  - `submissions/cd_lns_sa/placer.py`

## Notes

- **DO NOT modify `submissions/cd_lns_sa_cascade_dp_lane/placer.py`.** That
  variant is used by PATH B's overnight Phase 4 hybrid `--all` validation.
- This variant is purely additive — if it doesn't validate, fall back to
  the single-DP variant.
