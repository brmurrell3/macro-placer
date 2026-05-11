---
id: E66
name: hardware_probe
status: graduated
parent: E48 (champion)
created: 2026-05-03
decided: 2026-05-03
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **PARTCL REGRESSION RISK PARTIALLY DE-RISKED — bench-specific pattern (updated 2026-05-03 22:30 EDT after ibm12 probe).** Three probes under thread-limit (OMP/MKL/OPENBLAS/VECLIB/NUMEXPR=1) produced better-or-tied scores than historic --jobs 4 baselines: ibm01 0.89004 vs 0.89234 (−0.26 %), ibm10 1.00408 vs 1.00961 (−0.55 %), ibm12 1.20595 vs 1.20639 (−0.04 %, effectively tied). **The contention-vs-compute boundary is bench-specific.** ibm10 was contention-bound: under thread-limit CD plateau-exits at 1829s (vs cap at 2400s) with 30 % budget headroom recovered. ibm12 is truly compute-bound: under thread-limit CD STILL caps at 2400s on both lanes, last-sweep Δ=+0.00012 (E25) / +0.00019 (E41), still descending when wall fires. Per-sweep wall on ibm12 is ~173 s thread-limited — same scale as ibm10's 166 s thread-limited — but ibm12 needs 14+ sweeps to plateau where ibm10 needed 11. **Implication for partcl:** the "switch to --jobs 1 in production" tradeoff helps contention-bound benches (ibm10-class) but does not buy headroom on truly compute-bound benches (ibm12-class). The complementary defense is the §4.5 work-bounded refactor (E68): looser soft cap (3000s vs 2400s) gives compute-bound benches the 3-4 extra sweeps they need to plateau. **CAVEAT:** thread-limit on M3 Max removes parallel-job thread-contention; it is NOT a faithful "slower per-core server CPU" simulation. A real partcl-class test would require server hardware or `cpulimit`-style throttling. The remaining concern — partcl's per-core clock being significantly slower than M3 Max — is partially addressed by E68's looser caps but ultimately requires AWS c7i / c7g verification before submission.
champion_delta: not applicable (defensive measurement)
graduated_to: null
superseded_by: null
---

# E66: hardware_probe — measure regression risk on partcl-class hardware

## Hypothesis
The partcl evaluation hardware is reportedly "many cores, lower clock speed"
(server CPU class — e.g., AMD EPYC ~2.45 GHz base vs M3 Max P-cores ~4.0 GHz).
The user observed that the leaderboard's "Incremental CD+LNS" entry by vmallela
(verified 1.1172 on user-class hardware) reportedly drops to ~12th place when
re-evaluated on partcl's hardware. **Our champion E48 may have similar
budget-bound risk.**

The mechanism by which lower per-core clock causes score regression: phases
that hit hard caps (run for full budget without plateau exit) make fewer
internal iterations on slower hardware, so they converge less and produce
worse placements.

## Pre-existing evidence (from existing E41 / E25 --all logs)

Parsed `experiments/E41_dpo_kjoint/run_all.log` and
`experiments/E25_lns_sa_compose/run_all.log` for per-phase exit reasons:

### E41 (lane 2 of E48 hybrid) phase profile across 17 benches
| Phase | Plateau exits | Cap exits | Risk |
|---|---:|---:|---|
| CD (cap=2400s) | 12 | 5 | **HIGH** — 5/17 benches hit cap; on slower HW more will |
| LNS (budget=600s) | 17 (saturate ≤397s) | 0 | LOW — plateau-bound |
| SA-v2 (budget=600s) | 14 (lift=+0.00000 — best-so-far holds CD output) | 0 | LOW — saturated |
| K-joint (budget=600s) | 3 (≤340s) | 14 | **MEDIUM** — budget-bound by design; commits fewer K-tuples on slower HW |

### E25 (lane 1 of E48 hybrid) phase profile across 17 benches
| Phase | Plateau exits | Cap exits | Risk |
|---|---:|---:|---|
| CD (cap=2400s) | 15 | 2 | LOWER — only ibm12 + ibm17 cap |
| LNS, SA-v2 | all plateau / saturated | 0 | LOW |

### Cap-hitters (E41 CD, hardware-fragile benches)
By proxy match against `docs/results.md` per-bench table:
- proxy 1.02741 → likely ibm10 (1.0096 final) — CD ran 11 sweeps to cap
- proxy 1.20771 → ibm12 (1.2064 final) — CD ran 11 sweeps to cap
- proxy 1.19697 → ibm14 (1.1995 final) — CD ran 11 sweeps to cap
- proxy 1.14762 → ibm16 (1.1440 final) — CD ran 10 sweeps to cap
- proxy 1.34104 → ibm17 (1.3324 final) — CD ran 8 sweeps to cap

**These are the largest IBM benches.** Pattern: macro count scales with
sweep cost (each sweep is O(N) macros × O(B) breakpoints). On slower HW,
each sweep takes proportionally longer → fewer sweeps in 2400s → worse
proxy at cap exit.

## Method (probe)

### Phase 1 — Thread-limited reproduction (in flight)
Run E48 hybrid on **ibm01** (smallest IBM bench, ~50 min on M3 Max baseline)
under `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1`. Compare to historic
ibm01 baseline 0.89234 (from `E48_hybrid_all` log row).

Thread-limit doesn't perfectly simulate "lower clock per core, more cores",
but reduces per-bench parallelism in NumPy / MPS / OpenBLAS-backed kernels.
First-order proxy for "the per-core compute fence is tighter."

### Phase 2 — Big-bench cap stress (planned, conditional on Phase 1 result)
If Phase 1 shows score holding within 0.5 % of 0.89234, run **ibm17**
(hardest cap-hitter, proxy 1.3324) under same thread-limit. ibm17 is where
regression would manifest most strongly. Expected wall: ~3-4 hr.

### Phase 3 — Mitigation pilot
If regression is confirmed in Phase 2, test:
- **Higher CD plateau threshold** (0.001 → 0.002): easier to exit, frees
  budget for cap-hit benches. Tradeoff: less precise plateau detection;
  may exit while still descending.
- **Tighter CD hard cap** (2400 → 1800 s): forces earlier transition to
  LNS / SA. Tradeoff: less CD convergence on big benches.
- **Drop K-joint K=3 phase** entirely on cap-bound benches: K-joint is
  always budget-bound (commits 12-58 K-tuples in 600s); skipping it loses
  -0.001 to -0.008 per bench but releases 600s wall.

## Kill gate
N/A (defensive — measuring risk, not breakthrough).

## Generalization check
None required — the question is hardware-invariance, not benchmark transfer.

## Final state

### Phase 1 — ibm01 thread-limit probe (completed 2026-05-03 14:58 EDT)

| Lane | Proxy | Notable |
|---|---:|---|
| E25 (SDF init pipeline) | **0.89004** | (E48 winner) |
| E41 (DPO + K-joint) | 0.91207 | |
| **E48 = min(E25, E41)** | **0.89004** | Total wall 2849 s |

vs historic E48 ibm01 = 0.89234. **−0.26 % LIFT under thread-limit.**

Source: `results/probe_t1_ibm01.log`,
`results/CDLNSSAHybridPlacer_20260503_145800.json`.

### Phase 2 — ibm10 cap-hitter thread-limit probe (completed 2026-05-03 16:48 EDT)

| Lane | Proxy | Notable |
|---|---:|---|
| E25 (SDF init pipeline) | 1.04711 | |
| E41 (DPO + K-joint) | **1.00408** | **CD plateau-exit at 1829s, NOT cap** |
| **E48 = min(E25, E41)** | **1.00408** | Total wall 6588 s |

vs historic E48 ibm10 = 1.00961. **−0.55 % LIFT under thread-limit.**

E41 lane phase exits:
- CD: plateau-exit at sweeps=11, wall=1829.8s, proxy=1.02719 (historic --jobs 4 hit cap at 2400s, proxy=1.02741 — same quality, 23 % less wall).
- LNS: plateau-exit at samples=10, wall=363.9s, proxy=1.00956.
- SA-v2: lift_vs_init=+0.00134, best=1.00822.
- K-joint: 1 pass to budget cap, committed=72, proxy=1.00075.

Source: `results/probe_t1_ibm10.log`,
`results/CDLNSSAHybridPlacer_20260503_164855.json`.

### Phase 3 — second cap-hitter probe ibm12 (completed 2026-05-03 22:28 EDT)

| Lane | Proxy | CD exit | Notable |
|---|---:|---|---|
| E25 (SDF init) | 1.20797 | **cap at sweeps=14, 2400s, Δ_last=+0.00012** | LNS plateau (s=4); SA lift +0.00000 |
| E41 (DPO + K-joint) | **1.20595** | **cap at sweeps=14, 2400s, Δ_last=+0.00019** | LNS plateau (s=5); K-joint 1 pass / 40 commits / Δ=−0.00104 |
| **E48 = min** | **1.20595** | both lanes capped | Total wall 7031 s |

vs historic E48 ibm12 = 1.20639. **−0.04 % under thread-limit (effectively tied).**

**Pattern reversal vs ibm10.** Where ibm10 plateau-exited at 1829s with
30 % CD headroom recovered, ibm12 still caps both lanes at 2400s. Per-
sweep wall is similar (ibm12 ~173 s vs ibm10 ~166 s thread-limited) but
ibm12 needs more sweeps to reach plateau (last-sweep Δ still above
plateau threshold even under serial execution). **ibm12 is truly
compute-bound**, not contention-bound.

Source: `results/probe_t1_ibm12.log`,
`results/CDLNSSAHybridPlacer_20260503_202832.json`.

### Phase 4 — slow-clock simulation (NOT RUN)

The original Phase 3 plan (mitigation pilot — try higher CD plateau threshold,
tighter cap, drop K-joint on cap-bound benches) was deferred. The ibm12
probe shows the right defense is **looser soft caps** (per §4.5), not
aggressive plateau detection — the existing plateau threshold doesn't
fire on ibm12 even at 14 sweeps because Δ remains above 0.001 longer
than budget allows. E68 (`experiments/E68_workbounded_refactor/`)
implements the §4.5 prescription: CD soft cap 2400→3000s, LNS 600→900s,
SA 600→900s, K-joint 600→900s, with work-bounded primary termination.

The remaining hardware concern — actual slower per-core clock on partcl —
requires real server hardware or `cpulimit` throttling, neither of which
is locally available. Recommended follow-up: run an AWS c7i (Intel Xeon ~2.5 GHz)
or c7g (Graviton3 ~2.6 GHz) instance for a faithful partcl-class test before
submission.

## Pointers
- E41 source log (used for pre-existing analysis):
  `experiments/E41_dpo_kjoint/run_all.log`
- E25 source log: `experiments/E25_lns_sa_compose/run_all.log`
- Champion: `submissions/cd_lns_sa_hybrid/placer.py`
- Per-bench historic E48 results: `docs/results.md` §CDLNSSAHybridPlacer
  (E48) section.
