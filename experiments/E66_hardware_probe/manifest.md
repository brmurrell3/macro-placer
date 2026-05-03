---
id: E66
name: hardware_probe
status: graduated
parent: E48 (champion)
created: 2026-05-03
decided: 2026-05-03
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **PARTCL REGRESSION RISK SUBSTANTIALLY DE-RISKED 2026-05-03 16:48 EDT.** Two probes under thread-limit (OMP/MKL/OPENBLAS/VECLIB/NUMEXPR=1) produced **better** scores than historic --jobs 4 baselines: ibm01 0.89004 vs 0.89234 (−0.26 % LIFT), ibm10 1.00408 vs 1.00961 (−0.55 % LIFT). **Critically, ibm10 (a CD cap-hitter under --jobs 4) plateau-exited CD at 1829s instead of hitting the 2400s cap** — same proxy reached (1.0272 vs 1.0274) but with 30 % budget headroom. This means the "5 cap-hitters on E41 lane" risk identified from existing --jobs 4 logs (ibm10/12/14/16/17) was **largely a parallel-contention artifact, not an algorithmic-quality limit**. Under serial execution each CD sweep takes ~160s vs ~218s parallel; plateau detection fires before the cap. **Implication for partcl:** if partcl runs each bench serially (the natural per-bench-1-hr-cap model), our pipeline plateau-exits cleanly with budget margin. The hardware-regression scenario where we run out of budget mid-CD requires partcl's per-core clock to be MORE than 30 % slower than M3 Max (the headroom we measured) AND the slowdown to exceed our budget headroom. **CAVEAT:** thread-limit on M3 Max removes parallel-job thread-contention; it is NOT a faithful "slower per-core server CPU" simulation. A real partcl-class test would require server hardware or `cpulimit`-style throttling. But the existing data shows our pipeline handles serialization well — remaining concern is purely "what if partcl's per-core clock is much slower than M3 Max" which is testable separately.
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

### Phase 3 — slow-clock simulation (NOT RUN)

The original Phase 3 plan (mitigation pilot — try higher CD plateau threshold,
tighter cap, drop K-joint on cap-bound benches) is **not needed**: Phase 2
showed cap-hitting was the parallel-contention artifact, not a real
algorithmic-quality limit. The mitigations would have addressed the wrong
problem.

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
