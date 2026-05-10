---
id: E79
name: hardware_portability
status: superseded
parent: E74
created: 2026-05-05
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: E83
---

# E79: hardware_portability

## Hypothesis

E74 (CDLNSSAHessian) runs ~96 min/bench on M3 Max for ibm01 (E25 ~25 min +
E41 ~30 min + Hessian ~48 min). The judging hardware is AMD EPYC 9655P with
a hard 1-hr cap. E74 already exceeds the cap on M3 Max; on slower per-core
EPYC the gap widens. Three §Derisk mitigations together should bring wall
to ~48 min/bench, fitting the 60-min cap with headroom:

1. **Parallel E25 ⊥ E41** — ~25 min saved (max(25, 30) ≈ 30 min vs 25+30=55 min).
2. **Compressed Hessian** — n_eigvecs=1 (was 2), polish_budget=180s (was 240s).
   Saves ~30 min: 12 polishes @ 240s → 6 polishes @ 180s. ~80 % of lift on
   ibm01 came from eig0, so eig1 is largely redundant.
3. **Hardware probe** — at startup, scale phase budgets to hardware speed
   so judges' hardware gets proportionally more time per phase if slower.

## Method

- Workers run as subprocesses via `subprocess.Popen` (avoids Windows pickle
  issues with ProcessPoolExecutor + spawn). Each worker loads the benchmark
  by name, runs E25 or E41, writes the result as `.npy`, exits.
- Worker normalizes Windows backslash paths to forward slashes when calling
  `load_benchmark` directly (avoids modifying `macro_place/loader.py`).
- Hessian parameters: `n_eigvecs=1`, `eps_values=(0.3, 1.0, 3.0)`,
  `polish_budget=180.0`.
- Hardware probe: 50 iterations of 512×512 float32 matmul. M3 Max baseline
  = 1.2 s. Scale ∈ [0.5, 1.5].
- Work-bounded streak termination (LNS 5-streak, SA 1000-move, K-joint
  30-tuple) deferred — requires patching cd_core inner loops. Implemented
  as a follow-up if wall still exceeds cap after (1) + (2) + (3).

## Kill gate

- proxy > 1.0800 on --fast (regression vs E74 1.0666 > 1 %) → falsified.
- Any overlap on any benchmark → falsified.
- Wall > 60 min on ANY --fast benchmark → falsified.

## Generalization check

- `--all` proxy ≤ 1.0800 (within ~1.3 % of E74 1.0666).
- `--ng45` proxy ≤ 0.700 (within ~2.7 % of E74 0.6813).
- All 17 IBM + 4 NG45: zero overlaps.

## Outcome (partial — first --fast run, 2026-05-05)

| Bench | E79 proxy | Wall (min) | vs E74 wall | Status |
|---|---:|---:|---:|---|
| ibm01 (smoke) | 0.86417 | 65 | −32 % vs E74 96 min | ✅ |
| ibm01 (--fast) | 0.8820 | 57 | better | ⚠ proxy regressed +2.1 % vs smoke (probe over-shrank polish to 90s) |
| ibm04 | 0.9928 | 61 | within cap | ✅ |
| ibm09 | 0.8438 | 65 | over cap (5 min) | ⚠ |
| ibm13 | 0.9455 | **78** | **+18 min over 60-min cap** | ❌ Kill-gate failure |
| **AVG** | **0.9160** | — | — | proxy gate ✅ (< 1.0800), wall gate ❌ |

### Root causes

1. **Probe over-shrunk polish on faster-than-M3 hardware.** Original floor
   `0.5` reduced polish from 180s → 90s on this hardware, costing −2.1 %
   proxy on ibm01 with negligible wall savings.  *Fixed*: floor → 1.0.
2. **E25/E41 use full phase budgets on hard benches.** ibm13 spent 67 min
   in parallel E25⊥E41 (each lane ≈ 70-min budget — CD 2400s + LNS 600s +
   SA 600s + KJoint 600s).  Without §Derisk Mitigation #3 (work-bounded
   streaks), ibm13 cannot fit the 60-min cap.

### Decision

E79 status remains `in_progress` pending follow-ups:

- **E80** (next): port §Derisk Mitigation #3 (work-bounded streaks) into
  `run_lns_gridbin` / `run_sa_polish_v2` / `run_kjoint_lns`.
- Re-run E79 `--fast` with corrected probe floor to confirm proxy returns
  to smoke-test level (~0.864 on ibm01).
- E79 is **partial-success**: parallelism + compressed Hessian save
  ~30 min on benches that already plateau early, but largest benches
  need work-bounded termination too.

## Pointers

- Code: `code/cd_lns_sa_hessian_fast.py`, `code/_worker.py`.
- Results: `results/`.
