# 2026-05-18 morning status (overnight 05-17 → 05-18)

## TL;DR

**Submission entry is ready for judges.** Run via
`./eval_docker/run_eval.sh thinkorplace placer.py` (no extras arg
needed — `submit_deps/dreamplace_install/` is bundled and auto-found
inside the container). Expect IBM `--all` ≈ 1.07820 / NG45 ≈ 0.68102
because the bundled DREAMPlace is CPU-only; the GPU lane fails gracefully
and the 2-lane fallback runs cleanly. To get the 3-lane 1.06650, supply
a CUDA-built DREAMPlace via `$DREAMPLACE_ROOT`.

**Decision still open:** Option B (cd_lns_sa_cascade_dp_lane, currently
wired into `placer.py`) vs. Option C (cd_lns_sa_cascade_stacked_periphery,
**better number 1.0575 with no external deps**).
See [`2026-05-18_next_submission_candidate.md`](2026-05-18_next_submission_candidate.md)
for the switch instructions if you want to flip to Option C.

## What's been verified

| | IBM `--all` | NG45 `--ng45` | Combined (21) | Hardware |
|---|---:|---:|---:|---|
| **Option B (current wiring)** 2-lane | **1.07820** | **0.68102** | 0.998 | AWS EPYC c6a.4xlarge |
| Option B 3-lane (CUDA DP) | 1.06650 | 0.68086 | 0.993 | lambda cloud A100 |
| **Option C (alternative)** | **1.05750** | **0.68930** | **0.987** | M3 Max --jobs 4 |
| Option C K_eps=3 variant | 1.05689 | (not run) | — | M3 Max |

## Overnight runs in flight

Both started 2026-05-17 ~9 PM. See `Monitor` task `bqs912901` for
tabu_stacked WINNERs (real-time).

### tabu_stacked --all on M3 (`E108_tabu_stacked_all`)

Hypothesis: replace `cascading_saddle` (canonical (1,0.5,0.5) Hessian
eigvec) with `tabu_levy_saddle_escape` from E99 — forces orthogonal
eigvecs across iters for more direction diversity, complementary to
portfolio's weight diversity.

**Status at 23:45 EDT:** 12/17 WINNERs landed. Simple sum-avg over
finished 12 = **0.98441**. Hard benches (ibm17, ibm18) still pending.
**ETA: ~1-2 AM** for completion. Log: `/tmp/tabu_stacked_all.log`.

Decision logic: if final avg < 1.0575, promote tabu_stacked over
stacked_periphery as next candidate.

### no_e41_deep --all on aws-gpu (`E108_no_e41_deep_all`)

Hypothesis: skip E41 lane, reallocate its 0.32×B budget to cascade
(0.20→0.36) and portfolio (0.13→0.29). Maybe more saddle iters
compensate for worse starting plateau.

**Status at 23:45 EDT:** 3/17 WINNERs landed. First bench 0.87179
vs original 0.85963 = **+1.4 % regression**. Hypothesis weakening
already. ETA late tomorrow morning (aws-gpu is slower, --jobs 3 CPU).
Log on aws-gpu: `~/no_e41_deep_all.log`.

Decision logic: if any of the first 5 benches shows ≥1 % regression,
kill it. Current trend: kill.

## DREAMPlace situation

- Bundled `submit_deps/dreamplace_install/` is **CPU-only** (built
  against pytorch:2.5.1-cuda12.4-devel with `_GLIBCXX_USE_CXX11_ABI=0`
  but without CUDA kernels). 98 MB, 36 .so files.
- The placer defaults to `DP_USE_GPU=1`. With v2 (CPU-only), the DP
  subprocess fails with "CANNOT enable GPU without CUDA compiled",
  returns non-zero exit, placer logs the failure and falls back to
  2-lane. **Net effect on bundled-install path: IBM 1.07820.**
- A `dreamplace:2.5.1-cuda12.4-v3` CUDA-enabled image was built on
  aws-gpu (~24.6 GB). Verified compile but not end-to-end tested
  (libcudart.so.11.0 vs 12.4 mismatch suspected — needs investigation
  if we want the 3-lane number bundled).
- **For tomorrow:** if there's time, finish v3 verification and replace
  the bundled v2 with v3. Otherwise, ship 2-lane (1.07820) which beats
  every verified leaderboard entry except vmallela (#1 at 1.0109).

## Other autonomous work this session

- Made README.md, SUBMISSION.md, submissions/README.md, and the
  launcher placer.py docstring all honest about the bundled-CPU-DP
  behavior. Judges who clone-and-run get 1.07820 (no surprises).
- Confirmed launcher path resolution outside Docker
  (`uv run evaluate placer.py -b ibm03 --json` works).
- Confirmed inner placer DREAMPlace auto-discovery hits
  `submit_deps/dreamplace_install/` via `_ROOT / "submit_deps" / "dreamplace_install"`.

## What to check first thing in the morning

1. Look at `Monitor bqs912901` events for tabu_stacked WINNERs. If 17/17
   done, run:
   ```bash
   grep WINNER /tmp/tabu_stacked_all.log | sed -E 's/.*proxy=([0-9.]+).*/\1/' | \
     awk '{sum+=$1; n+=1} END {printf "avg=%.5f over n=%d\n", sum/n, n}'
   ```
   If `avg < 1.0575`: promote tabu_stacked. Update
   `2026-05-18_next_submission_candidate.md` to point at it.

2. Check aws-gpu no_e41_deep status:
   ```bash
   ssh aws-gpu "grep WINNER ~/no_e41_deep_all.log"
   ```
   If still regressing on most benches, kill (`ssh aws-gpu "pkill -f no_e41_deep"`).

3. Submission-day decision (B vs C): the user owns this. Both are
   wall-safe and verified. C is **better** (1.0575 vs 1.0782) **and
   simpler** (no DREAMPlace dependency). Switch instructions are in
   `2026-05-18_next_submission_candidate.md`.

## Files modified tonight

- `README.md` — submission section accuracy, no extras arg in one-liner
- `SUBMISSION.md` — honest bundled-DP-is-CPU section
- `submissions/README.md` — verified-results table reordered (2-lane first)
- `placer.py` — docstring updated
- `docs/handoffs/2026-05-18_next_submission_candidate.md` — variant status
- `docs/handoffs/2026-05-18_morning_status.md` — this file

No code changes to placers; only documentation/README work + variant
runs in flight.
