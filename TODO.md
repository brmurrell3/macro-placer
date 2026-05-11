# TODO — temporary

**Status:** scratch doc, delete after the deadline. Not part of the
canonical docs.

Deadline: May 21, 2026 (~10 days).

## SESSION HANDOFF — 2026-05-11 01:39 PDT

### What's currently running

- **Local M3 Max**: `pid 66167` running ibm01 wall-safe smoke (budget=3300s),
  log at `/tmp/walltight_smoke/ibm01_b3300.log`. Saddle phase, 1 trial
  completed (0.87196, NEW BEST vs plateau 0.89614). 11 trials remaining.
  Expected completion ~22:00 PDT. Final SMOKE_IBM01 line is the result.
- **Cloud OCI A100** (mpc-cloud / 132.145.135.39): no python processes.
  Earlier smokes completed. DREAMPlace installed at `/opt/DREAMPlace/install`
  (built with ABI=1).

### What was verified this session

1. **E84 cascading verified at canonical 1.0612** across 17 IBM, zero
   overlaps. Loader at `experiments/E84_cascading_saddle/code/loader_placer.py`.
   Cached .pt files at `experiments/E84_cascading_saddle/results/cascade_ibm*.pt`.
   Result JSON: `results/CascadingLoader_20260510_205559.json`.
2. **Wall-safe pipeline works end-to-end** (smoke ibm03 b=600s: 1.00480,
   598s, zero overlaps, deadline triggered cleanly after 10/12 trials).
3. **DREAMPlace integration mechanically functional** (subprocess uses
   `/usr/bin/python3`, integer-scaled Bookshelf, legalize_flag=0). But
   output has 30-44 overlaps that CD polish (60s) can't fully fix → DP
   lane skips itself. **Decision: deprioritize DP, focus on cascade.**

### 5 LOCAL COMMITS PENDING PUSH (sandbox blocked `git push`)

```
76bbca4 DREAMPlace integration: int-scale Bookshelf, system python3, CD polish
cac929d Cloud driver: set OPENBLAS/OMP/MKL_NUM_THREADS=8
48348a5 Add wall-safe cascade README + cloud --all driver; update CLAUDE.md
78d8f81 Wall-safe E74 + cascade variants with budget_seconds enforcement
8b40bc5 Add cd_lns_sa_hessian_dp — champion + optional DREAMPlace lane
```

Run `git push origin main` manually to sync to cloud.

### Next session priority order

1. **Verify ibm01 wall-safe smoke result** — read `/tmp/walltight_smoke/ibm01_b3300.log`
   tail for SMOKE_IBM01 line. Compare to cached E74 0.85527.
2. **Push commits** (above).
3. **Kick off cloud --all wave**: `ssh mpc-cloud "bash ~/macro-place-challenge-2026/run_cloud_walltight_all.sh cascade 4 cascade_walltight_$(date +%s)"`
   — runs wall-safe cascade variant across all 17 IBM with budget=3300s
   each, jobs=4 parallel, OPENBLAS=8. ~5-6 hr wall.
4. **Kick off local --all wave** (in parallel): `uv run evaluate submissions/cd_lns_sa_cascade/placer.py --all --jobs 4 --json --hypothesis cascade_walltight_local`. ~5-6 hr wall.
5. **Compare aggregates**: cascade wall-safe vs E48 baseline 1.08151.
   If beats E48 → promote E84 cascading via ADR-013 (in `docs/decisions/`).
6. **NG45 verification**: `uv run evaluate submissions/cd_lns_sa_cascade/placer.py --ng45 --jobs 4 --json`.

## Champion lineage state

Current champion (live): E74 CDLNSSAHessian, 1.0666 verified canonical.

Next champion candidate: **E84 cascading saddle escape, verified canonical 1.0612.**
Wall-safe variant: `submissions/cd_lns_sa_cascade/placer.py`. budget_seconds=3300s default.

Fallback if cascade fails wall: E48 CDLNSSAHybrid 1.08151.

## State (PRE-SESSION)

Two candidates currently in the tree:

| Candidate | `--all` | `--ng45` | Wall fits 60-min cap? | Notes |
|---|---:|---:|---|---|
| **E74 CDLNSSAHessian** | **1.0666** | **0.6813** | ❌ ~96 min worst-case | Proxy-best but DQs on partcl 1-hr-per-bench cap |
| **E84 cascade (verified)** | **1.0612** | (not yet run) | ❌ 8/17 over 55min unbudgeted; wall-safe variant pending | -0.51% vs E74; needs wall-safe --all validation |
| **E83 CDLNSSAHessianClock** | 1.0859 | (not run) | ⚠️ 17/17 fit on Windows; 5/17 within 1-min margin | Wall-safe; +0.41% worse than E48 |
| E48 CDLNSSAHybrid (prior fallback) | 1.08151 | 0.6922 | ❌ ~130 min sequential | Same wall problem as E74 |

**E74 wall-safe variant added 2026-05-11**: smoke ibm03 b=600s validated
1.00480/598s/zero-ovl/deadline-clean. Full ibm01 b=3300s in progress.
**Cascade wall-safe variant ready** at `submissions/cd_lns_sa_cascade/placer.py`.

## Critical-path open items

### 1. **Wall-safe E74 with mid-loop time enforcement** (priority #1)

Modify `submissions/cd_lns_sa_hessian/placer.py` to enforce a hard
end-to-end deadline inside the Hessian saddle escape loop:

- At placer entry, set `deadline = start + budget_seconds` (default
  3300 s = 55 min, leaves 5 min for harness overhead).
- Before each ±ε polish trial, check `time.time() < deadline`.
  If not, return best-so-far without further trials.
- Inside `run_cd_adaptive` polish, pass through a `deadline` arg
  so CD exits early on any sweep boundary past the deadline.
- Also enforce on Phase 1 (E25) and Phase 2 (E41) — currently each
  has its own multi-step caps that can stack to >60 min before
  Hessian even starts.

Expected: ibm01 96 min → ≤ 55 min by clipping the Hessian trial
list when budget exhausted. Some benches will see fewer ε trials
than full E74 run; aggregate proxy will rise slightly but should
stay well below E48.

Smoke gate: ibm01 + ibm17 (largest) both finish ≤ 55 min with
proxy within 0.5 % of full E74 cached result.

Done when: all 17 IBM walls under 55 min on this machine AND
aggregate proxy ≤ E48 1.08151.

### 2. **Throttled-CPU verification** (priority #2 — gates submission)

Before submitting, simulate slower per-core hardware to confirm
proxy holds under wall pressure. Two options:

```bash
# Option A: single thread (forces serial work)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  uv run evaluate submissions/cd_lns_sa_hessian/placer.py --all --json

# Option B (Linux): cpulimit
cpulimit -l 50 -- uv run evaluate submissions/cd_lns_sa_hessian/placer.py --all
```

Goal: every bench under 60 min, every bench within 1 % of
full-power proxy. If proxy regresses > 1 % on any bench, the
budget caps in (1) are too tight — relax CD plateau patience
or extend `run_cd_adaptive` `min_time_s` floor.

### 3. **E80 work-bounded streaks** (priority #3 — defensive)

`experiments/E80_work_bounded_streaks/` is `in_progress`. Port the
saturation streak counters into the wall-safe E74 placer from (1):

- LNS: terminate after 5 consecutive non-improving samples
- SA-v2: terminate after 1000 moves without improvement
- K-joint: terminate after 30 K-tuples with no commits

This is hardware-invariant — the streak fires whether the platform
is fast or slow. Composes with (1)'s wall enforcement: phase exits
on streak OR deadline, whichever first.

### 4. **NG45 verification of the wall-safe build** (priority #4)

Once (1) and (2) pass IBM, run `--ng45` and confirm:

- All 4 designs ≤ 60 min
- ariane133 ≤ 0.6900 (current E74 was 0.6641; allow some regression
  from wall-clipping but must not regress to E48's 0.6861)
- ariane136, mempool_tile, nvdla ≤ E48 reference

Done when: `--ng45` avg ≤ 0.69 with zero overlaps.

### 5. **Submission package + form** (priority #5)

Once (1)–(4) green:
- Confirm `submissions/cd_lns_sa_hessian/placer.py` is the entry.
- Check `SETUP.md` for any submission-format requirements.
- Fill out the Google form (link in `README.md`).
- Keep `submissions/cd_lns_sa_hybrid/placer.py` as private fallback.

## Lower-priority / can drop

- **Tier 2 ORFS verification scoping** — if we make top-7 by Tier 1,
  we're automatically considered for Tier 2 ($20k Grand Prize).
  Worth reading `SCORING.md` to know what failures look like, but no
  build work needed. Skip unless time after (1)–(5).
- **Innovation Award writeup** ($4k) — paper writeup at
  `writeup/paper.md`. Worth ~3 days end-of-deadline if (1)–(5) ship.
  Describe the mechanism: smooth-proxy autograd Hessian on the local
  proxy minimum + Lanczos eigvec + ε-step + CD polish. Frame as
  applying transition-state methods (well-developed in
  chemistry/materials) to combinatorial placement; cite Henkelman &
  Jónsson 2000 NEB literature for the mathematical foundation.

## Settled / no further work

- E77 sharper Hessian — marginal, no incremental lift.
- E78 layered E61V2+E74 — marginal.
- E79 hardware_portability — superseded by E83.
- E81 cd_only_saddle — falsified.
- E82 hybrid_dispatcher — falsified.
- E83 clock_aware — marginal at 1.0859; doesn't beat E48 in
  isolation. Body of work absorbed into the wall-safe E74 plan above.

## Reusables (don't recompute)

- Cached E25 + E41 placements: `experiments/E69_sequence_pair_search/results/placements/` (ibm01/04/09/12).
- E61V2-fresh outputs: `experiments/E75_fresh_e61v2_wave/results/` (ibm12/14/15).
- Per-bench E74 outputs (full Hessian wave): `experiments/E74_hessian_saddle/results/` (all 17 IBM + layered ibm12/15).
- Validator (loads cached): `submissions/cd_lns_sa_hessian/loader_placer.py`.

## Delete this doc when

E74 (or its wall-safe successor) is submitted, or May 21 passes.

## Daily progress log

- 2026-05-05 — E74 promoted (ADR-012); cached wave verified 1.0666
  `--all`, 0.6813 `--ng45`.
- 2026-05-05/06 — E77 / E78 / E79 / E80 / E81 / E82 / E83 derisk wave;
  E83 clock-aware fits 60-min cap (1.0859, marginal).
- 2026-05-10 — TODO consolidated; wall-safe E74 plan is critical path.
- (next entries here as work lands)
