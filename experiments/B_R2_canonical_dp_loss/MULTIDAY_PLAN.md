# Multi-day plan to close gap to vmallela 1.011

**Started:** 2026-05-15 03:55 UTC. **Deadline:** 2026-05-22 03:00 EDT.

Realistic target: **1.02-1.05 aggregate** over 6 days. Sub-1.011 still hard but tightens gap from −5.2% to −0.5 to −3%.

## Phase 1 — Multi-seed --all (TONIGHT, in flight)

**Launched:** 2026-05-15 03:55 UTC. Queue script: `/tmp/multiseed_all_overnight.sh`.

Multi-seed GPU DP basin selection (N=5 per bench, pick best legalized) → polish → cascade saddle. 17 IBM + 4 NG45 in batches of 2 parallel.

**ETA done:** ~10 AM EDT 2026-05-15 (10 hr from launch).

**Validation gate:** aggregate vs hybrid baseline 1.067.
- ≥0.5% lift → continue Phase 3
- 0% to 0.5% → continue but reduce confidence
- regression → pivot to pin-aware HPWL alone

**Expected lift:** −0.5 to −1.5%. ibm10 standalone showed −0.7% but ibm12 regressed +2.0%; full 21-bench picture unknown.

## Phase 2 — Day 1 morning (May 15 AM)

Aggregate Phase 1 results. Update `SUMMARY.md` and this plan. Decide:
- (A) Continue Phase 3 if Phase 1 confirms lift
- (B) Pivot to pin-aware HPWL standalone if not

Estimated 1-2 hr of analysis + decision.

## Phase 3 — Pin-aware HPWL (Day 2, May 16)

**Hypothesis:** Real pin offsets vs center-pin approximation gives DP a truer HPWL gradient, lifting basin quality 0.5-1.5%.

**Implementation:**
1. Modify `experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py`:
   - Use `benchmark.macro_pin_offsets[i]` instead of `(0, 0)` in `_write_nets`
   - Convert pin offsets to bookshelf-format relative coords
2. Run pin-aware multi-seed --all on cloud (10 hr)
3. Aggregate.

**Validation gate:** Phase 3 aggregate vs Phase 1 aggregate.
- ≥0.5% lift → continue Phase 4
- < 0.5% → skip Phase 4 (use Phase 1 placement)

## Phase 4 — Deeper cascade saddle (Day 3-4, May 17-18)

**Hypothesis:** k=10 Lanczos eigvecs + max_iters=20 + finer ε grid finds deeper minima than current k=2, max_iters=5.

**Implementation:**
1. Modify `experiments/E84_cascading_saddle/code/cascading_saddle.py`:
   - Default k=2 → k=10 in hessian_saddle params
   - max_iters=5 → 20 (with budget-aware exit unchanged)
   - eps_values=(0.3, 1.0, 3.0) → (0.1, 0.3, 0.7, 1.5, 3.0, 5.0)
2. Need extra budget — may require dropping E25 or E41 lane on benches where they consistently lose
3. Run all-improvements-stacked multi-seed --all (~12-15 hr due to deeper saddle)

**Validation gate:** Phase 4 aggregate vs Phase 3 aggregate.
- ≥0.5% lift → continue
- < 0.5% → revert to Phase 3 placement

## Phase 5 — Submission packaging (Day 5, May 19)

**Build Dockerfile** that:
- Uses base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (challenge default)
- Installs DREAMPlace with all required ops
- Includes our patched PlaceObj (if using B-R2 — currently NOT in plan)
- Mounts the placer code

**Verify wall-cap compliance** on all 21 benches: ibm17 was the historical violator; need to confirm fix holds with deeper saddle/multi-seed/pin-aware combined budget.

**ORFS feasibility check** (for Tier 2 Grand Prize $20K):
- Identify whether our NG45 placements pass ORFS flow without timing/area violations
- If passing → $20K shot is real
- If failing → optimize for ORFS feasibility (different objective weighting)

## Phase 6 — Day 6 (May 20): Buffer

React to any unexpected issues. Final tuning. Decide submission strategy:
- If Phase 4 aggregate ~1.02-1.04: submit that
- If only ~1.05-1.06: still better than current PATH A 1.077; submit

## Phase 7 — Day 7 (May 21): Submit

Submit via `https://forms.gle/YDRtYV5Vq68SZgKW9` before 23:59 PT = 02:59 EDT.

## Coordination with other agent

Other agent (cascade_*_levy lineage) may continue running E97/E100 variants. They have produced:
- E100_dual_levy: 1.073 17-IBM, ariane133 = 0.644 (best NG45 result so far)
- E100_portfolio_v2: 1.072 17-IBM

If Phase 4 stacks well with their work, combine: hybrid+multi-seed+pin-aware+deeper-saddle for IBM, dual_levy for NG45. **Not rule-compliant as separate placers** but could be a single placer that conditionally routes per-bench-feature.

## Files I'll touch (no conflicts with other agent)

- `experiments/B_R2_canonical_dp_loss/code/cascade_dp_multiseed_placer.py` (mine)
- `experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py` (will fork to new file for pin-aware)
- `experiments/E84_cascading_saddle/code/cascading_saddle.py` (will fork)
- Cloud `~/DREAMPlace_cpu/install`, `~/DREAMPlace_gpu/install`

## Files I won't touch

- `submissions/cd_lns_sa_cascade*/placer*.py` (other agent's submission lineage)
- `TODO.md`, `CLAUDE.md`, `README.md`
- `macro_place/` core library
