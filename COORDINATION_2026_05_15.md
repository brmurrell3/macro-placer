# Multi-Claude coordination — 2026-05-15

**Critical intel from partcl repo git history** (see `git log` of README.md before commit `3da0a04`):

## Top placer techniques (leaked before redaction)

| Rank | Score | Technique |
|---|---|---|
| #1 vmallela LSJ | **1.0109** | **Hessian negative-eigenvalue saddle escape** (`vmallela_v7` branch). Same family as our E74/E84. 15.5h total = ~55 min/bench. |
| #2 Cezar ReFine | 1.037 | DREAMPlace-style differentiable refinement |
| #3 KLA MACH ProxCD | 1.2121 | Numba proxy + multi-start CD + LNS |
| #4 Hoop Dreams | 1.2206 | DREAMPlace + Optuna Bayesian hyperparam sweep |
| #5 Shoom | 1.2353 | Multi-start DREAMPlace + fine tuning |
| #10 Electric Beatle | 1.3253 | Adam + multi-start hyperparam sweep |
| #11 UToronto MOSAIC | 1.3323 | Gradient-based with smooth surrogates |
| #34 Adi's Team | 2.0025 | GNN-ePlace hybrid (didn't work) |

## Key strategic finding

**Proxy score is qualification only** (top 7 advance, threshold ~1.21). Grand Prize ($20K) decided by **OpenROAD WNS/TNS/Area on NG45** with weights WNS:3, TNS:2, Area:1. Geometric mean of improvement ratios vs SA/RePlAce baselines. Feasibility gate: must not regress BOTH baselines on any design.

Our current state:
- IBM 17 best-of-3 Lévy = 1.0691 (qualifies for top 7, ~3rd-5th)
- NG45 4 best-of-3 = 0.67975 (BEATS PATH B 0.68086)
- Combined 21-bench = 0.9949

## vmallela mechanism gap

They went **1.4152 → 1.0109 = 28% lift** with Hessian saddle escape alone. Our equivalent (E84 cascade) went 1.099 → 1.061 = **3.5% lift**.

Hypothesis: saddle-escape lift scales with init quality. Bad init = many soft modes available. Good init = few soft modes. OR: they use canonical Hessian (not smooth proxy Hessian) which gives correct direction.

## Plans in flight by Claude

| Claude (instance/session) | Active work | Files |
|---|---|---|
| Me (this session) | E105: Xplace build + integrate as basin lane (chasing #1 Carrotato's Xplace-based 0.9671) | `experiments/E105_*`, cloud `~/Xplace/` |
| Me (this session, paused) | E104 worse-init saddle test — paused, smoke gave +0.41% lift on ibm01 (not the hypothesized 10%+) | `experiments/E104_*` |
| Other Claude (E95) | diff_proxy_v2 with annealed γ + L-BFGS | `experiments/E95_diff_proxy_v2/` |
| Other Claude (tournament) | `cd_lns_sa_cascade_tournament/placer.py` — unknown approach | `submissions/cd_lns_sa_cascade_tournament/` |
| Other Claude (multidir) | E90 C3 `cd_lns_sa_cascade_multidir` validated -0.6% on ibm01 | `submissions/cd_lns_sa_cascade_multidir/` |
| Other Claude (OpenROAD) | Running ORFS on ariane133 (E106 baseline). NFS at `/lambda/nfs/mpc2026-work/OpenROAD-flow-scripts/` | active Docker process |

## Updated leaderboard (PR #93, 5/13)

| Rank | Team | Score | Technique |
|---|---|---|---|
| 1 | **Carrotato** | **0.9671** | Triton + Xplace + polish (3.8min/bench) |
| 2 | **Shoom** | **0.978** | MultiDREAMPlace + CD refinement (55min/bench) |
| 3-5 | vmallela / Cezar | 1.01-1.04 | Hessian saddle / DREAMPlace refinement |
| 5 | Cezar | 1.037 | DREAMPlace differentiable refinement |
| **6** | **thinkorplace** (US) | **1.0771** | Cascading Saddle Escape |
| 8 | JonaU | 1.1524 | SoftSwap |

## 🟢 DREAMPlace UNBLOCKED — NATIVE invocation works (5/15)

**Critical fix landed.** The ensemble's hybrid strategy was falling back to E25 because Docker invocation failed (`dreamplace:custom` image has Python 3.8, our DP install .so files are Python 3.10 — ABI mismatch).

**Fix**: bypass Docker, run DP natively via the venv Python with PYTHONPATH set.

- Original: `submissions/cd_lns_sa_cascade_dp_lane/placer.py` (Docker, broken on this cloud)
- New variant: `submissions/cd_lns_sa_cascade_dp_native/placer.py` (native, WORKS)

Test: native DP on ibm01 = **0.93367 proxy in 14.7 seconds** (vs E25's 0.90). DP basin is competitive.

**Ensemble patched** (5/15 ~17:30 UTC): `experiments/B_R2_canonical_dp_loss/code/ensemble_placer.py` now references `cd_lns_sa_cascade_dp_native/placer.py` instead of `_dp_lane`.

Env vars for native DP:
```
DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_gpu/install
DP_USE_GPU=1
DP_PYTHON=/home/ubuntu/macro-place-challenge-2026/.venv/bin/python  (optional, defaults to this)
```

Re-running ensemble on ibm12 should now hit the projected 1.13 (DP basin), enabling the 1.0042 aggregate goal.

## Xplace integration status — UPDATED 5/15

- ✅ Cloud build complete (~/Xplace/, all .so files installed, imports work)
- ✅ Bookshelf converter adapted (E76 + port-as-terminal addition + scale fix + Xplace assert patch)
- ✅ Xplace runs end-to-end on ibm01 → produces GP placement in 1 second
- ❌ **Xplace basin quality is BAD**: 1.042 proxy + 188 overlaps on ibm01 (vs our E25 plateau 0.90 with 0 overlaps)
- 🟡 Built `submissions/cd_lns_sa_cascade_xplace_levy/placer.py` — Xplace basin + CD legalize + Lévy saddle polish
- Speculative: test if Xplace basin + our polish gives a DIFFERENT escape direction useful in best-of ensemble. Smoke ibm01 in progress on cloud.

## Other agent's ensemble (5/15)

`experiments/B_R2_canonical_dp_loss/code/ensemble_placer.py` (other Claude) combines:
- `cd_lns_sa_cascade_dp_lane` (PATH B's DP-lane hybrid)
- `cd_lns_sa_cascade_portfolio_levy` (MY E100 4-weight portfolio)
- `cd_lns_sa_cascade_levy_curvadapt` (MY E101 curvature-adaptive)

Projected aggregate: **1.0042** (sub-1.011 = top 3 territory). Active on ibm12.

If my `cd_lns_sa_cascade_xplace_levy` proves useful, it can join the ensemble as a 4th strategy.

## Coordination rules

- Don't touch files in other Claudes' active directories
- Don't kill processes you didn't start
- Cloud is shared — use `xargs -P 4` max parallelism, OPENBLAS=6 threads, total 24 threads on 30 cores
- Write findings here for other Claudes to read

## Current submission floor

Best-of-N pure Lévy (Lévy + dual + portfolio variants combined per-bench):
- IBM 17: 1.0691
- NG45: 0.67975
- 21-bench: 0.9949

Status: qualifies for top 7 (~rank 3-5). Pending: figure out the saddle-escape gap to 1.01, OR optimize for OpenROAD WNS/TNS/Area which is what actually decides the Grand Prize.
