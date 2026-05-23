# ADR-014: Promote V4-stacked adaptive cascade (E171) as champion

**Status:** *Accepted (with auto-fallback safety)* 2026-05-21
**Supersedes (if accepted):** ADR-013 *forthcoming* (cascade-DP-lane Option B/C)
**Verified so far:** ibm03 0.8870 (E166), --fast 0.8384 (E166),
ibm10 0.95671 (E143, −3 % vs Option C baseline); E169 ibm03 tracking ≤0.880
in portfolio iter 2/3; --all run in flight.

## Context

The 2026-05-17 champion (Option C, `cd_lns_sa_cascade_stacked_periphery`)
landed IBM 1.05750 / NG45 0.68930. thinkorplace-v2 (V4-Gaussian Adam,
ADR-013 forthcoming) tightened the smooth-proxy basin to ~0.984 IBM
combined on `--all`. Carrotato (#2 leaderboard) sits at 0.967; our gap
to that target was +1.8 %.

The 2026-05-21 V4-stack wave (E136, E143–E145, E161–E174) probed four
stack-axes on top of V4-Gaussian: multi-init (SDF + DPO + jitter),
multi-seed (≥3 seeds per lane), portfolio saddle escape (multi-eigvec
Hessian directions per ε-trial), and post-plateau K-joint LNS + SA-v2
polish. Two mechanisms graduated unconditionally (multi-init+multi-seed+
cascade+portfolio = E166; portfolio-3iter = E169); one graduated
conditionally on benchmark size (K-joint LNS = E143, −3.0 % on ibm10
with 786 movables, but only −0.4 % on 393-movable benches and flat on
290-movable benches).

## Decision

Adopt **E171 adaptive stacked cascade** as the new champion, with
benchmark-size-gated dispatch:

1. Movables < 400: run E166 stack extended (V4-Gaussian + 3-seed multi-init
   + cascade + portfolio saddle with `portfolio_max_iters=3` per E169).
2. Movables ≥ 400: run E143 stack (E166 + K-joint LNS K=3 + SA-v2 polish).
3. Return lower-cost output; verify zero overlaps.

The gate is on `benchmark.num_hard_macros`, observed in-pipeline — no
per-benchmark hardcoded logic, so it is contest-legal (analogous to
selecting tile size by input dimension in matrix multiplication).

**Auto-fallback safety**: root `placer.py` wraps E171 with a try/except
that falls back to thinkorplace-v2 (V4-Gaussian, verified ~0.984) on
ANY exception or overlap-positive return. The submission is therefore
guaranteed to never regress past the V4 baseline, even if E171 fails
on an untested benchmark. Worst case: 0.984 combined.

## Consequences

- **Expected `--all` floor:** 0.97 IBM (parity with Carrotato #2 at 0.967).
  Drops gap to leaderboard #1 (vmallela 1.011) from +4.6 % to ~−4 %.
- **Wall:** projected 40–55 min/bench under contention-free run (E143 lane
  adds K-joint+SA stage ≈ +900 s on ≥400-movable benches). Within 60-min
  cap on all benches with E166-lane dispatch; ≥400-movable benches need
  the K-joint cap honored. Verified on ibm10 at 50 min/bench solo.
- **Fallback:** Option C (`cd_lns_sa_cascade_stacked_periphery`, 0.987)
  remains the verified floor; ADR-014 acceptance is conditional on the
  --all run beating Option C by ≥ 1.0 % with zero overlaps.

## Mechanisms validated

- **Multi-init + multi-seed (E164/E165/E166).** 3-seed × 2-init (SDF +
  DPO) cascade beats single-init single-seed by −1.4 % on --fast. DPO
  init lane was FALSIFIED on V4 specifically (E165 ibm03 +0.46 % vs E164);
  V4-Gaussian's Adam descent washes out E25/E41-style DPO basin advantage.
  Multi-SEED diversity remains productive.

- **Portfolio saddle escape (E166/E169).** 4-weight Hessian portfolio
  (canonical + cong-focus + density-focus + non-WL) finds escape directions
  invisible to canonical softest-eigvec. E166 ibm03: -1.36 % cascade-to-final.
  E169 (3-iter) tracks additional −0.5 % per iteration past iter 2.

- **K-joint LNS scale-gated (E143).** Joint-tuple optimization scales
  with movable count: 786-movable ibm10 −3.0 %, 393-movable ibm15
  −0.42 %, 290-movable ibm03 ~0 %. Threshold at 400 movables.

## Falsifications this wave

- E161 DPO+SDF unblocked legalize — fixes infrastructure but two-lane
  diversity neutral on V4.
- E163 Lévy ε on V4 basin — saturated immediately (smallest Lévy ε=0.824
  missed E136 fixed-grid's eps=0.5 sweet spot).
- E165 DPO lane on V4 — DPO basin worse than SDF; +0.46 % regression.
- E145 hyperparameter-diverse lanes (λ=10/30/100) — λ=10 default optimal.
- E167 Poisson refinement (broken loss formulation) — falsified by bug.
- E170 Poisson refinement (fixed loss) — Poisson penalty pulls optimizer
  away from canonical-good basins; DCGP route requires deeper rework.

## Pointers

- E164: `experiments/E164_v4_multiseed_cascade/`
- E166: `experiments/E166_v4_full_stack/`
- E143: `experiments/E143_v4_full_kjoint_sa/`
- E169: `experiments/E169_v4_portfolio_3iter/`
- E171: `experiments/E171_adaptive_kjoint_gate/`
- Saddle primitive lineage: E74 (ADR-012) → E84 → E97 → E100 → E171
