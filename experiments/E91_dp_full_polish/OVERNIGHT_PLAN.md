# Overnight Queue Plan (launched 2026-05-13 01:20 UTC)

Cloud: lambda.ai 129.213.18.245. Master log: `/tmp/overnight_master.log`.
Phase logs: `/tmp/overnight/<phase>_<bench>.log`.

## Sequence

### Wait for in-progress (~01:20-02:00 UTC)
- 6 hybrid IBM: ibm01/09/10/12/14/17 with hybrid placer (PRE-fix DP config — used hardcoded target_density=0.85). Provides first hybrid `--all` data points.
- ariane133 v4 (auto-adaptive DP config, rule-compliant).

### Phase 1: IBM auto-config validation (~02:00-03:00 UTC)
- Re-run B-R0' on ibm10/ibm12/ibm14/ibm17 with auto-adaptive DP config.
- Verifies no regression vs original B-R0' (which used hardcoded 0.85).
- 4 benches in parallel, 4 threads each.

### Phase 2: NG45 generalization (~03:00-04:00 UTC)
- B-R0' on ariane136, mempool_tile, nvdla.
- Verifies the ariane133 generalization holds on the rest of NG45.

### Phase 3: remaining 11 IBM benches (~04:00-06:30 UTC)
- B-R0' on ibm02/03/04/06/07/08/11/13/15/16/18.
- 3 batches of 4 benches each, parallel.
- Completes the IBM B-R0' picture.

### Phase 4: hybrid --all (~06:30-11:30 UTC)
- Full hybrid placer on all 17 IBM + 4 NG45 = 21 benches.
- Batches of 4. Each bench ~50 min wall.
- This is the canonical `--all` aggregate for hybrid vs cascade.

## Rule compliance

All DP configs use **auto-adaptive** target_density derived from
`macro_density = sum(macro_area) / canvas_area` × 1.5, bounded
[0.40, 0.85]. Single formula applied to every input — no per-bench
branching.

## Failure handling

- If a bench crashes / overlaps remain: hybrid placer's
  `_polish_dp_basin` returns `(None, inf)` so DP lane is excluded
  from plateau pick. Bench still produces a valid placement via
  E25/E41 lanes.
- Budget violations: cascading_saddle_escape now tracks avg iter
  wall and skips next iter if predicted to exceed budget.
