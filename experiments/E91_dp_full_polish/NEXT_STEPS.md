# Next steps after B-R0' results land

## Step 1 (immediately after each B-R0' run completes):
Sync results from cloud:
```bash
rsync -az ubuntu@129.213.18.245:~/macro-place-challenge-2026/experiments/E91_dp_full_polish/results/ experiments/E91_dp_full_polish/results/
```

## Step 2 (decide based on B-R0' results):
Look at ibm10/12/14/17 final proxies. Three possible verdicts:

### Verdict A: DP+full-polish ≤ cascade-capped on ≥3 of 4 benches
**Action:** the autopsy was definitively wrong. Build hybrid placer
with DP as 3rd lane. ~1 day engineering, then `--all` run on lambda.ai
to verify aggregate. Expected lift: 0.5-1.5% on top of E48 hybrid.

### Verdict B: DP+full-polish within +3% of cascade on ≥2 benches
**Action:** marginal lane — useful as diversity for hybrid but not a
solo win. Build the hybrid anyway for diversity. Implement B-R1
(TILOS-RUDY) as the diagnostic for what's holding DP back on the
remaining bench(es).

### Verdict C: DP+full-polish > cascade by >5% on most benches
**Action:** autopsy holds, polish is not sufficient. Skip hybrid.
Commit to B-R1 → B-R2 → B-R3 sequence (modify DP loss ops).

## Step 3 (build hybrid placer if Verdict A or B):

Scaffold: copy `submissions/cd_lns_sa_cascade/placer.py` to
`submissions/cd_lns_sa_cascade_dp/placer.py`. Add Phase 3 between
E41 and saddle escape that:

1. Tries `_try_run_dreamplace` (adapt from
   `submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py:68-201`)
2. Runs the polish pipeline on DP init (CD + LNS + SA) with budget
   `B * 0.20` for CD, `B * 0.06` each for LNS/SA — same as E25/E41.
3. Adds the DP-polished plateau to the candidate list.
4. Cascading saddle on the best plateau (same as cascade).

Key change vs the falsified placer: equal polish budget for DP lane.

## Step 4 (B-R1 if needed):

Patch `~/DREAMPlace_cpu/install/dreamplace/PlaceObj.py` to add a
congestion term to obj_fn. Use either:
- `RudyDiff` from `experiments/E92_dp_tilos_rudy/code/diff_rudy.py`
  (pure torch, autograd-friendly, prototype)
- Or DP's built-in `rudy.Rudy` if its CUDA kernel exposes a backward
  (verify in `dreamplace/ops/rudy/src/`).

Weight: start at `congestion_weight = density_weight * 0.5` to match
canonical 1.0/0.5/0.5 weights.

## Step 5 (B-R2/B-R3 if needed): see E93 / E94 manifests.

