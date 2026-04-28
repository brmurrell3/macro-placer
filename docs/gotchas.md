# Codebase gotchas (hard-won)

Code-level discoveries surfaced during the post-CD experiment phase. These
are not in the public docstrings but bit us at least once each. Recorded
verbatim from the retired `findings.md` so they don't get lost.

1. **`benchmark.net_nodes` is empty** — known loader bug, GitHub issue #70.
   Net connectivity must be read from `plc.nets` by parsing pin parent names
   (`"module/pin"` → `module` is the parent macro). See
   `sdf_init.py:_extract_edges` and `cd_pair_swap.py:_build_candidate_pairs`.

2. **Hardcoded `external/MacroPlacement/Testcases/ICCAD04` paths in 3 files**
   would crash on NG45 evaluation. Fixed via
   `macro_place/bench_paths.find_benchmark_dir()` (tries `$BENCH_ROOT`,
   `ICCAD04`, `NG45`, and an `rglob` fallback).

3. **Evaluate harness loads placers via `spec_from_file_location`** which
   doesn't put repo root on `sys.path`. All experiment placers must add
   `sys.path.insert(0, str(_ROOT))` before importing from `submissions.*`.

4. **Incremental evaluator's `revert()` is single-step only** — only undoes
   the LAST move. For multi-move sequences (pair-swap, multi-destroy LNS),
   revert manually via inverse moves.
