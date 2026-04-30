# Codebase gotchas (hard-won)

Code-level discoveries surfaced during the post-CD experiment phase. These
are not in the public docstrings but bit us at least once each. Recorded
verbatim from the retired `findings.md` so they don't get lost.

1. **Hardcoded `external/MacroPlacement/Testcases/ICCAD04` paths in 3 files**
   would crash on NG45 evaluation. Fixed via
   `macro_place/bench_paths.find_benchmark_dir()` (tries `$BENCH_ROOT`,
   `ICCAD04`, `NG45`, and an `rglob` fallback).

2. **Evaluate harness loads placers via `spec_from_file_location`** which
   doesn't put repo root on `sys.path`. All experiment placers must add
   `sys.path.insert(0, str(_ROOT))` before importing from `submissions.*`.

3. **Incremental evaluator's `revert()` is single-step only** — only undoes
   the LAST move. For multi-move sequences (pair-swap, multi-destroy LNS),
   revert manually via inverse moves.

4. **Legality-check eps must point AWAY from `compute_overlap_metrics`'s zero
   tolerance.** `compute_overlap_metrics` (in `macro_place/objective.py`) flags
   any positive overlap area as a violation — it has no tolerance. If a
   custom legality check uses `dx < min_dx - eps` with a positive eps to
   "tolerate float noise", it will admit placements with up to `eps × eps`
   overlap area that the validation harness will reject. Bit us in E39 K-joint
   on `ibm07` (eps=1e-4 → "1 overlap, area 0.0000" crash on bench 7/17 of
   `--all`). Fix: use `dx < min_dx + eps` (strict separation REQUIRED, not
   tolerated) with a small positive eps like 1e-9, AND add a post-commit
   `compute_overlap_metrics` defensive revert. See
   `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py` for the
   pattern. Inherited by E41 via import.
