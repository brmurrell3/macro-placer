---
id: E40
name: multi_sa_seed
status: marginal
parent: E25
created: 2026-04-29
decided: 2026-04-30
champion_at_time: 1.0990    # E12 CDLNSGridBin
fast_outcome: 0.93295 (--fast); -0.07 % vs E25 fast 0.9336; per-bench wins on ibm04 (-0.07 %), ibm09 (-0.20 %), ibm13 (-0.18 %); loss on ibm01 (+0.16 %); zero overlaps. Multi-SA-seed best-of-4 finds tiny lift over E25 single-seed but doesn't compound — fork 1 (seed=42, same as E25) often is the best fork. SKIPPING --all (marginal lift not worth ~14 hr wall).
outcome: marginal (--fast +0.07 % win); skipped --all
champion_delta: -0.07 %
graduated_to: null
superseded_by: null
---

# E40: multi_sa_seed

## Hypothesis
E25's pipeline (CD-adaptive + grid-bin LNS + SA-v2 polish) is *deterministic*
through CD and LNS (modulo `lns_seed`), but SA-v2 is genuinely stochastic
through `sa_seed`. The post-LNS state is identical across seeds; only the
SA chain diverges. Forking 4 SA seeds {42, 1, 2, 3} from the *same saved
post-LNS state* and keeping the best across forks should improve per-bench
proxy by 0.05–0.2 % on benches where SA-v2 found lift in E25 (ibm01,
ibm04, ibm08, ibm09, ibm10). The cost is cheap because CD + LNS — the
expensive deterministic phases — are shared across all 4 forks; only SA
gets multiplied (4× 600 s = 2400 s vs E25's 600 s).

If SA-v2 is *insensitive* to seed (i.e., the chain converges to the same
basin from any seed because T₀ = 5e-4 is too low to escape), then
multi-seed shows no lift over E25. The hardest benches (ibm12–18) where
E25 SA found nothing should also show no multi-seed lift — confirming the
plateau is robust to the move set, not the seed.

## Method
Per-benchmark pipeline (3600 s legal cap is *not* respected — multi-seed
SA exceeds it; this is a research probe, not a contest-legal placer):

1. SDF init.
2. Project overlaps.
3. Build `IncrementalProxyEvaluator`.
4. **CD phase** (`run_cd_adaptive`, `cd_hard_cap_s=2400`,
   `cd_plateau_threshold=0.001`) — UNCHANGED from E25.
5. **LNS phase** (`run_lns_gridbin`, `lns_budget_s=600`,
   `destroy_frac=0.05`, `destroy_cap=30`, `lns_seed=42`) — UNCHANGED from E25.
6. **Snapshot** `saved_placement = evaluator.placement.detach().clone()`.
7. **SA-v2 fork loop** for `sa_seed` in `[42, 1, 2, 3]`:
   a. Restore evaluator to `saved_placement` by walking per-macro `move()`
      for any index whose current position differs from saved (mirrors the
      SA-v2 best-restore pattern in E25 — keeps V/H congestion cache in
      sync; direct mutation of `evaluator.placement` would desync).
   b. Run `run_sa_polish_v2(seed=sa_seed, ...)`.
   c. Record `(sa_seed, final_proxy, final_placement_clone)`.
8. **Pick best fork** (lowest final proxy). Walk evaluator state from
   current → best fork's placement via `move()` (same pattern).
9. Validate (zero overlaps), preserve fixed macros, return.

All hyperparameters global. `sa_seeds` is a list, default `[42, 1, 2, 3]`.

Walls: CD ≤ 2400 s + LNS ≤ 600 s + 4 × 600 s SA = 5400 s/bench worst-case
≈ 1.5 hr/bench. `--all` ETA ≈ 16 hr (E25 base 10 hr + 3 extra SA forks ≈
6 hr). Over the 17 hr legal cap — research probe only.

## Kill gate
- **Multi-seed has no lift:** if avg `--fast` ≥ E25 fast 0.9336, kill —
  SA-v2 is seed-insensitive at T₀ = 5e-4 and the basin is locked.
- **Pipeline regression:** if avg `--fast` > E12-random 0.9372, kill —
  there's a state-restoration bug (forks not actually starting from the
  same post-LNS state, or evaluator desync).

## Generalization check
- Any single `--fast` benchmark improves over E25 → queue `--all`.
- If `--fast` improvement is concentrated on benches where E25 SA already
  found lift (ibm01/04/09), that confirms seed variance hypothesis.
- If improvement is broad (including ibm13/17/etc.), the seed effect is
  larger than expected — investigate whether LNS-then-SA needs better
  cooling schedule.

## Outcome (filled when decided)
[Empty until decided. Then: result, decision, rationale.]

## Pointers
- Code: `code/cd_lns_sa_multi_seed.py` (defines `CDLNSSAMultiSeedPlacer`).
- Inlines E25's primitives (`run_cd_adaptive`, `run_lns_gridbin`,
  `run_sa_polish_v2`, etc.) — self-contained.
- Parents: E25 (champion candidate at 1.0954).
- Discussion: probes seed-variance bound on SA-v2; the cheapest possible
  test of "is SA-v2's basin determined by post-LNS state alone, or does
  the seed matter?".
