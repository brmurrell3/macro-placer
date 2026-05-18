---
id: E111
name: population_annealing
status: in_progress
parent: E25
created: 2026-05-18
decided: null
champion_at_time: 1.05750
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E111: population_annealing — replica resampling SA replacement

## Hypothesis

Single-chain SA-v2 (`submissions/cd_lns_sa/placer.py::run_sa_polish_v2`)
saturates on hard benches because the post-cascade landscape is
spin-glass-like (E100 found multiple residual soft Hessian directions
across weight vectors). Population annealing (Hukushima & Iba 2003,
PRE 67 056111) provably escapes glassy landscapes via systematic
weight-based resampling of N parallel replicas at each temperature step.
No smooth-proxy gradient — canonical proxy only, via the E1 incremental
evaluator's 4657× speedup.

## Method

- N = 24 replicas (each owns an `IncrementalProxyEvaluator` on a deep-copy
  of the cascade+portfolio plateau placement).
- Pre-ladder diversification: 100 Metropolis steps at T₀ per replica
  with independent RNG.
- Temperature ladder: T₀ = 1e-3 → T_f = 1e-6 over 40 geometric steps.
- Per step: each replica runs 500 Metropolis sweeps on the
  `axis_breakpoints` move set (same as SA-v2). Acceptance:
  `exp(-Δproxy_canonical / T_k)`.
- Resample at T_k → T_{k+1} with weights
  `w_i ∝ exp(-(1/T_{k+1} - 1/T_k) · (E_i - E_min))`
  and systematic resampling (Baumgartner & Wang 2013, PRE 87 033303)
  conserving N exactly. Replicas with count 0 die; count ≥ 2 clone.
- Track global best across all replicas × all temperatures; restore at
  end via `evaluator.move` per macro.

Wall budget: 600s (matches `run_sa_polish_v2`'s slot inside the cascade
pipeline). Expected load on ibm10-class: 145s sweep + ~140s clone
overhead. Cloud-only — falls back to single-chain SA-v2 if A100
unavailable (detected via DREAMPLACE_ROOT + ssh).

Integration: monkey-patch `run_sa_polish_v2` from the v2 composer when
env `MPC_V2_H2=1`. `run_pa_polish` signature-compatible with `**_unused`
swallowing SA-only kwargs.

## Kill gate

1. **Smoke (May 18 EOD)**: replica build + one ladder step on ibm03
   must complete without OOM and improve canonical proxy on at least
   one replica.

2. **`--fast` lift (May 19 EOD)**: `MPC_V2_H2=1` `--fast` aggregate
   beats re-measured Option C `--fast` by ≥ 0.5%. Below 0.5% = drop
   H2 from v2.

3. **Overlap**: zero overlaps required.

4. **NG45 sanity**: ariane133 must not regress > +1% vs Option C 0.6755.

## Generalization check

- IBM `--all` ≤ 1.046 on cloud A100.
- NG45 `--ng45` ≤ 0.690.
- Wall: each bench under 60-min/bench cap.
- Cross-platform: AWS c6a.4xlarge EPYC `--fast` smoke + ariane133.

## Outcome

[Empty until decided.]

## Pointers

- Code: `code/pa_core.py`, `code/pa_replica.py`, `code/pa_resample.py`
- Replaces: `submissions/cd_lns_sa/placer.py::run_sa_polish_v2`
- Integration: `submissions/cd_lns_sa_cascade_v2/placer.py`
- Cloud: mpc-cloud, OCI A100-SXM4-40GB at 132.145.135.39
- References: Hukushima & Iba 2003 PRE 67 056111; Baumgartner & Wang
  2013 PRE 87 033303; Wang 2015 PRE 92 063307.
- Parent: E25 CDLNSSA (provides the SA-v2 baseline being replaced).
