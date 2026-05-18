# 2026-05-17 morning handoff — NEW CHAMPION 1.0575

## TL;DR

`submissions/cd_lns_sa_cascade_stacked_periphery/placer.py` is the new
champion: **IBM 1.0575 / NG45 0.6893 / combined 0.987**. Beats Option B
by **−0.85 % IBM** and **−0.6 % combined**, with **no external
dependencies** (no DREAMPlace needed). Committed at 08f2af9.

| | New (C) | Option B | Option A |
|---|---:|---:|---:|
| IBM `--all` | **1.0575** | 1.06650 | 1.07820 |
| NG45 `--ng45` | 0.6893 | 0.68086 | 0.68102 |
| Combined (21) | **0.987** | 0.993 | 1.003 |
| Deps | none | DREAMPlace optional | none |

## Architecture

Cascade-then-portfolio sequential stacking:

1. **E25 lane** (SDF → CD → LNS → SA-v2, with new `project_overlaps`
   fallback before the final validity check)
2. **E41 lane** (DPO → CD → LNS → SA-v2 → K-joint)
3. **Plateau pick**: best of {E25, E41} (with `try/except` so if either
   lane crashes on residual overlaps the placer continues with whatever
   succeeded — fixes the NG45 nvdla case)
4. **Phase 3a: cascade_saddle** (canonical eigvec of (1, 0.5, 0.5)
   weighted-sum Hessian, 1 iter at 0.20 × B budget)
5. **Phase 3b: portfolio_saddle** (3 non-canonical weights: (1, 0, 1)
   cong-focus, (1, 1, 0) density-focus, (0, 1, 1) non-WL — accepted by
   canonical proxy)
6. **Periphery wrapper** (α=0.01 push toward edges + CD polish,
   strict-conservative accept — every push REJECTed in the --all run,
   wrapper acts purely as safety)

## Why it works

The previous direct portfolio_saddle from E25/E41 plateau (committed
at 88f6cc9 by another agent at 1.07404) wasted iters on canonical
direction (which cascade does better). Stacking cascade FIRST lets
portfolio focus iters on TRULY non-canonical directions where E100
spike showed the lift comes from (-0.92 % avg lift on hard benches).

## Per-bench results (IBM)

vs cached cascade uncapped (1.0612). 8 wins / 5 ties / 4 small losses
(none > +1.7 %). Detailed table in commit message 08f2af9.

## Failed parallel experiments (overnight 2026-05-16/17)

- **Xplace integration**: 5-config sweep (default, dw=5e-4, dw=1e-3,
  dw=1e-2, mixed_size=False, target_density=0.4). All produced
  catastrophic raw output (proxy 3.16-8.33, 209-72142 overlaps). Polish
  reduced ibm01 from 3.16 → 1.41 (vs SDF-cascade 0.85) — fundamentally
  bad basin from HPWL-only GP. K-Reorder patch-out allowed full
  pipeline to complete but didn't fix basin quality. **Verdict**:
  Carrotato's 0.967 via Xplace must use patched Xplace internals we
  don't have; not viable for our remaining time.

- **E101 levy_curvadapt --all** (aws-gpu): 1.0902 (worse than
  stacked_periphery by +3.1 %). Per-iter σ = β/√|λ_min| adaptive
  Lévy didn't help.

- **E108 critical_net_sa smoke** (M3): TIE on cached cascade plateau.
  Net-degree² weighting of SA macro-selection doesn't escape SA
  saturation. Marginal value on pre-plateau states (not tested).

## Still running (you'll see results when you wake)

- **M3 K_eps=3 variant --all** (started ~04:00, ETA ~08:00): Tests
  whether more Lévy magnitude diversity per portfolio weight gives
  marginal lift beyond 1.0575. Log: `/tmp/stacked_periphery_k3_all.log`.

## Compute state

- **aws-gpu**: SHUT DOWN at ~03:50 to stop $1.21/hr burn (E101 was its
  last job). To revive: `aws ec2 start-instances --instance-ids
  <id>`. Need fresh `aws login` first.
- **aws-cpu**: Another agent ran `cascade_levy --all` overnight; should
  be done. We didn't touch it.
- **M3**: Currently running K_eps=3 --all.

## Action items before submission day (May 21)

- [ ] Pick Tier-1 entry: Option C (`cd_lns_sa_cascade_stacked_periphery`)
      is recommended given combined 0.987 < Option B 0.993 and no DP
      dependency. Verify it runs cleanly on partcl judge box.
- [ ] If K_eps=3 variant (running now) is significantly better (say
      < 1.05), promote that instead.
- [ ] Drain `writeup/paper.md` TODO(prose) markers — paper draft
      already substantial per other agent's commits.
- [ ] Decide on Tier-2 ORFS per per-design strategy in
      `docs/handoffs/2026-05-16_tier2_orfs_findings.md`.

## Key files (champion)

- `submissions/cd_lns_sa_cascade_stacked_periphery/placer.py` — entry
- `submissions/cd_lns_sa_cascade_stacked/placer.py` — inner cascade→portfolio
- `submissions/cd_lns_sa/placer.py` — E25 with `project_overlaps` fix
- `experiments/E84_cascading_saddle/code/cascading_saddle.py` — cascade saddle
- `experiments/E100_weight_portfolio_saddle/code/portfolio_saddle.py` — portfolio saddle

## Memory updates

- `memory/champion_2026_05_17_stacked_periphery.md` — new champion details
- `memory/MEMORY.md` — index updated at top

## Submission packaging (added 2026-05-17 ~09:00)

Per partcl `eval_docker/Dockerfile` + `run_eval.sh` (GitHub PR #38):
- Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- Mounts: placer's parent dir at `/submission` (ro)
- Extra args mounted at `/submission/<basename>` (ro)
- Flags: `--gpus all`, `--memory 64g`, `--cpus 16`, `--network none`, `timeout 7200`
- Judges have: RHEL9, AMD Turin 24c, RTX A6000 40GB, CUDA driver ≥590

Run command (from official `run_eval.sh`):
```bash
./eval_docker/run_eval.sh team_name path/to/placer.py path/to/extras_dir
```

Our champion has cross-dir imports (`experiments/`, `submissions/cd_lns_sa`,
etc). The placer's parent-only mount won't include those. Solution shipped:
**`submit/placer.py` is a launcher** that:

  1. Discovers `/submission/repo` (mounted via run_eval.sh extras arg)
  2. Adds it to `sys.path`
  3. Imports + re-exports `CDLNSSACascadeStackedPeripheryPlacer`

Submission command:
```bash
./eval_docker/run_eval.sh thinkorplace path/to/repo/submit/placer.py path/to/repo
```

**DREAMPlace not needed** for our submission. Option C beats Option B
(1.0575 vs 1.06650) without DREAMPlace. If the team later wants to ADD
DREAMPlace as a 4th lane to stacked_periphery, would need to bundle a
pre-built install dir (compatible with PyTorch 2.5.1 + CUDA 12.4) as
another extra mount, then update placer to check for `DREAMPLACE_ROOT`.

## Overnight iteration after 1.0575 champion

- **K_eps=3 variant --all**: 1.0569 (−0.06% vs 1.0575). Marginal win
  within noise. Wins 9 benches, loses 7. Not worth promoting.
- **tabu_stacked --all** (running on M3, ETA ~12:30 PM): E99
  tabu_levy_saddle replaces standard cascading_saddle for orthogonal
  eigvec diversity across cascade iters. Hypothesis: direction
  diversity complementary to portfolio's weight diversity.
- **no_e41_deep --all** (queued, M3 after tabu): Skips E41 lane,
  reallocates 0.32×B to cascade (0.20→0.36) + portfolio (0.13→0.29).
  Doubles saddle budget.

If neither variant beats 1.0575 by >0.5%, ship Option C as is.
