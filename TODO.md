# TODO — temporary

**Status:** scratch doc, delete after the deadline. Not part of the
canonical docs (those are `CLAUDE.md`, `docs/roadmap.md`,
`docs/results.md`, ADRs).

## Pinned state — current champion

| | |
|---|---|
| Submission | `submissions/cd_lns_sa_hessian/placer.py` — **CDLNSSAHessian (E74)** |
| ADR | 012 *Accepted* 2026-05-05 (supersedes 011) |
| Verified `--all` (17 IBM) | **1.0666** = −1.38 % vs E48, −4.53 % vs leaderboard 1.1172, −26.8 % vs RePlAce |
| Verified `--ng45` (4 designs) | **0.6813** = −1.57 % vs E48 |
| Critical NG45 bench | ariane133 **0.6641** = −3.21 % vs E48 0.6861 (breaks failure point that killed E42/43/44/54/62) |
| Mechanism | E48 hybrid plateau → smooth-proxy autograd Hessian via `torch.autograd.functional.hvp` → Lanczos `eigsh(which="SA")` smallest-algebraic eigvecs → ±ε saddle perturbation → CD polish |
| Biggest lifts (per-bench) | ibm02 −7.13 %, ibm01 −3.86 %, ibm15 −1.73 % (E61V2-layered), ibm07 −1.67 %, ibm06 −1.86 % |
| Last commits | `2db85de` (E74 promotion), `86eb136` (E67/68/70 abandon-cleanup) on `main` |
| Memory | `e74_hessian_champion.md` |

**Submission policy:** E74 IS the submit target. E48 is a fallback only if
E74 has a regression on partcl hardware that we can't fix in time.

## Queue (priority order)

| # | Task | Wall | Expected lift | Notes |
|---|---|---|---|---|
| 15 | **E77 sharper E74** (k=5 eigvecs, finer ε on ibm12-18) | ~9 hr `--jobs 4` | +0.2–0.5 % | replace per-bench result if lower |
| 16 | **E78 layered E61V2 + E74** on remaining tied benches (ibm08, 14, 16, 17, 18) | ~30 hr | +0.1–0.3 % | pattern from ibm12/15 worked |
| 17 | **Hardware portability** (see §Derisk below) | ~1 day | defensive | must finish before submit |
| 18 | **Tier 2 ORFS verification** scoping | ~1 day | gates $20k Grand Prize | read SCORING.md |
| 19 | **Innovation Award writeup** at `writeup/paper.md` | ~3 days | $4k | Henkelman/Jónsson NEB applied to placement is novel |

## §Derisk — hardware portability strategy

**The risk.** Champion runs on M3 Max P-cores (~4 GHz, 16-core). Judges
evaluate on AMD EPYC 9655P (~3 GHz per-core, 16 cores + 100 GB RAM,
RTX 6000 Ada GPU available). Per-bench hard cap is **1 hour end-to-end**.

**Critical concern.** Canonical ibm01 smoke wall = **96 min on M3 Max**
(E25 ~20 min + E41 ~48 min + Hessian saddle ~28 min). **This is already
over the 60-min cap.** On slower per-core AMD EPYC the gap widens.

The leaderboard top entries fit in cap (Cezar 55 min/bench, vmallela
40 min, Shoom 42 min, MTK 37 s on GPU). We need to compress ours.

### Mitigations, ordered by impact and cheapness

#### 1. **Parallelize E25 ⊥ E41** in the placer (highest impact, ~1 day dev)

Currently `cd_lns_sa_hessian/placer.py` runs Phase 1 (E25) sequentially,
then Phase 2 (E41), then Phase 3 (Hessian). E25 and E41 are independent —
they can run in parallel via `concurrent.futures.ProcessPoolExecutor` or
`multiprocessing.Process`. Saves ~25–30 min/bench. Brings ibm01 from 96
to ~70 min — still tight but feasible if Hessian compresses too.

Risk: each subprocess imports torch + DPO smooth proxy, doubling memory
footprint. ~2× peak (~400 MB → ~800 MB). M3 Max 36 GB and partcl 100 GB
both handle this trivially.

#### 2. **Compress the Hessian phase** (high impact, ~hours dev)

Current: `n_eigvecs=2 × 2 signs × 3 ε values × 240 s polish each = ~48 min`.
Most lift on ibm01 came from `eig0 sign=-1 eps=1.0` and `eig1 sign=+1 eps=3.0`.
**Action**: drop to `n_eigvecs=1, eps={0.3, 1.0, 3.0}, polish_budget=180 s`
= 6 trials × 180 s = **18 min** Hessian. Need to validate this preserves
most of the −3.86 % ibm01 lift.

Risk: drops some bench-specific lifts. Mitigate by **per-bench cached
config**: run the full search once (we have the data); for benches where
the win came from a non-default eigvec / sign / eps, hardcode a per-bench
list. **Actually do not do this** — competition rules forbid per-benchmark
hardcoded logic. Instead: the eigsh return is rank-ordered, so eig0 with
sign sweep covers ~80 % of cases.

#### 3. **Adopt E68 work-bounded termination** (defensive, ~hours dev)

E68 (parallel agent's, abandoned) implements §4.5 of the roadmap: phase
plateau-/saturation-streaks as primary termination, wall caps as soft
secondary. **The lane-variance regression E68 saw was a separate issue**
(seed noise in the lanes); the work-bounded mechanism itself is sound.

Specifically, port these knobs into our champion (NOT the soft-cap
loosening — keep the same wall caps to stay under 1 hr):
- LNS: terminate on 5 consecutive non-improving samples (vs current 1)
- SA-v2: terminate on no improvement in last 1000 moves (vs wall-only)
- K-joint: terminate on 30 K-tuples with no commits

This protects against slower hardware where the same wall = fewer
sweeps. The streak metric is hardware-invariant; if we plateau early on
fast hardware AND late on slow hardware, the streak still fires.

#### 4. **Drop K-joint K=3 phase** in E41 (saves 10 min/bench)

E41 includes a K-joint K=3 polish (~10 min budget). Hessian saddle escape
finds deeper minima than K-joint K=3 does, so the K-joint phase is
largely redundant when Hessian comes after. **Action**: smoke-test E41
without K-joint on ibm01/04/12, verify proxy doesn't regress, drop the
phase from E41 inside our champion.

Risk: K-joint may help on benches where Hessian doesn't (ibm09 lift was
small). Test with both configurations.

#### 5. **Hardware-probe at startup** (highest portability, ~1 hr dev)

At placer init, run a fixed-work benchmark (e.g., 1000 incremental-
evaluator moves on a synthetic placement). Measure wall. Scale internal
phase budgets inversely:

```python
calibration_wall = run_calibration_benchmark()  # measure 1k moves
expected_wall = 5.0  # M3 Max baseline in seconds
scale = max(0.5, expected_wall / calibration_wall)
self.cd_budget = 2400 * scale
self.lns_budget = 600 * scale
# etc.
```

This adapts caps to whatever hardware the judges run on. If partcl is
2× slower per-core, all caps relax to 2× without changing the algorithm.

Risk: scales DOWN on faster hardware too — could under-budget. Cap the
scale at `[0.5, 1.5]` so we don't over-shrink.

#### 6. **Verify on throttled CPU** (validation, ~hours)

Before submitting, simulate AMD EPYC's slower per-core via:
```bash
OMP_NUM_THREADS=1 nice -n 19 uv run evaluate submissions/cd_lns_sa_hessian/placer.py -b ibm01
OMP_NUM_THREADS=1 nice -n 19 uv run evaluate submissions/cd_lns_sa_hessian/placer.py -b ibm17  # largest
```

Alternatively, on a Linux box run via `cpulimit -l 50 -- uv run evaluate`.

**Goal**: confirm proxy on throttled run is within 1 % of full-power
proxy on each bench. If proxy regresses more than that, mitigations
(1)–(5) need more aggressive tuning.

### Sequence to ship

1. Apply (1) parallelize E25⊥E41.
2. Apply (2) compress Hessian.
3. Smoke test on M3 Max — verify total wall ≤ 50 min/bench.
4. Apply (3) work-bounded streaks.
5. Apply (5) hardware probe.
6. Run (6) throttled-CPU verification.
7. If proxy holds, submit E74.
8. If proxy regresses, revert to E48 fallback (`submissions/cd_lns_sa_hybrid/placer.py`,
   verified 1.08151 / 0.6922).

Steps 1–2 alone should bring per-bench wall down from ~96 min to ~50 min,
leaving 10 min headroom for hardware variability. Steps 3–5 are
belt-and-suspenders.

## Reusables left in tree (don't re-compute)

- Cached E25 + E41 placements: `experiments/E69_sequence_pair_search/results/placements/`
  (ibm01, ibm04, ibm09, ibm12)
- E61V2-fresh: `experiments/E75_fresh_e61v2_wave/results/` (ibm12, ibm14, ibm15)
- Per-bench E74 outputs: `experiments/E74_hessian_saddle/results/` (all 17 IBM + layered ibm12/15)
- Validator (loads cached): `submissions/cd_lns_sa_hessian/loader_placer.py`

## Delete this doc when

- E74 successfully submitted, OR
- E74 superseded by something even better (E77 / E78 / DREAMPlace / etc.)

Whichever comes first.
