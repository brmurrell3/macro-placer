# TODO — temporary

**Status:** scratch doc, delete after the deadline. Not part of the
canonical docs.

Deadline: May 21, 2026 (~11 days).

## State

Two candidates currently in the tree:

| Candidate | `--all` | `--ng45` | Wall fits 60-min cap? | Notes |
|---|---:|---:|---|---|
| **E74 CDLNSSAHessian** | **1.0666** | **0.6813** | ❌ ~96 min worst-case | Proxy-best but DQs on partcl 1-hr-per-bench cap |
| **E83 CDLNSSAHessianClock** | 1.0859 | (not run) | ⚠️ 17/17 fit on Windows; 5/17 within 1-min margin | Wall-safe candidate; **+0.41 % worse than E48 prior champion** |
| E48 CDLNSSAHybrid (prior fallback) | 1.08151 | 0.6922 | ❌ ~130 min sequential | Same wall problem as E74 |

**There is no shippable candidate that beats E48 today.** E74 doesn't
fit the cap; E83 fits the cap but is worse than E48. The 11-day work
plan below is structured to fix this.

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
