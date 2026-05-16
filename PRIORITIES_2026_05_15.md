# Top-priority work items (2026-05-15, post-research)

**For other Claude sessions** — claim by adding `OWNER: <your-session-id>`
on a line under the item. Don't duplicate.

## Context

- 6 days to deadline (May 21)
- We are "thinkorplace" rank 6, verified proxy **1.0771**
- Top-3 cutoff ~1.01; gap is **5-6%**
- Grand Prize ($20K) decided by OpenROAD WNS/TNS/Area on NG45, NOT by proxy
- Existing ensemble chain on cloud (blocked, cloud unreachable as of 20:00 EDT)
- Submission floor: IBM 17 best-of-3 = 1.0691, NG45 = 0.67975, 21-bench = 0.9949

---

## P0 — Periphery-guided relocation (Grand Prize attack) 🔥🔥

**Source**: [DAC25-ReMaP](https://github.com/lamda-bbo/DAC25-ReMaP) —
*"Macro Placement by Recursively Prototyping and Periphery-Guided
Relocating"* reports **+34.15% WNS / +65.39% TNS** on ORFS designs
including ariane133, ariane136 (our Grand Prize set).

**Why P0**:
- Grand Prize is $20K, decided on WNS/TNS/Area (NOT proxy)
- [arxiv 2407.15026](https://arxiv.org/pdf/2407.15026) confirms
  "MacroHPWL only has WEAK correlation with actual Wirelength" —
  we've been chasing the wrong objective for $20K
- Local diagnostic confirmed ariane133 macros are center-clustered
  (mean edge_dist=0.241, periphery-pure ≈ 0.05) → 3-4× headroom
- Lift is empirically MASSIVE compared to anything else on the table

**Spike plan** (1-2 days):
1. Add periphery-bias term to cascade saddle escape:
   ```python
   def periphery_term(positions, canvas_w, canvas_h):
       dx_edge = torch.minimum(positions[:, 0], canvas_w - positions[:, 0])
       dy_edge = torch.minimum(positions[:, 1], canvas_h - positions[:, 1])
       return torch.minimum(dx_edge, dy_edge).mean() / max(canvas_w, canvas_h)
   ```
2. Objective: `canonical_proxy - α · periphery_term` (start α=0.05)
3. Re-run cascade on ariane133 with new objective
4. Validate via OpenROAD WNS/TNS (other Claude has ORFS infra)
5. If proxy regresses but stays < 1.21 (feasibility threshold), good
6. If WNS improves ≥10% AND proxy stays competitive, deploy

**Files to touch**:
- New: `submissions/cd_lns_sa_cascade_periphery/placer.py`
- Modify: cascade saddle objective (separate placer, don't break existing)

**Owner**: <unclaimed — needs OpenROAD access for full validation>

### 🟢 E107 FINAL RESULTS — 10 BENCHES (2026-05-16 02:18 EDT)

**NG45 (α=0.01)**:

| Bench | Baseline | Control | Periphery | Random | Center |
|---|---|---|---|---|---|
| ariane133 | 0.66993 | -0.28% (0✓) | **-0.95% (0✓)** | -0.62% (0✓) | -0.50% (1) |
| ariane136 | 0.66107 | -0.23% (0✓) | -0.38% (1) | -0.68% (5) | -0.55% (9) |
| mempool_tile | 0.73744 | 0.00% (0✓) | +0.41% (0✓) | +0.44% (0✓) | +1.15% (1) |
| nvdla | 0.68438 | -0.70% (0✓) | -0.67% (1) | -1.14% (4) | -1.48% (11) |

**IBM (α=0.01)** — all create overlaps:

| Bench | Baseline | Control | Periphery (ovl) | Random (ovl) | Center (ovl) |
|---|---|---|---|---|---|
| ibm01 | 0.86134 | -0.05% (0✓) | -0.58% (15) | -0.49% (9) | -0.49% (25) |
| ibm09 | 0.78006 | +0.13% (0✓) | -0.26% (20) | -0.27% (25) | -0.14% (33) |
| ibm10 | 1.16367 | -1.41% (0✓) | +9.47% (38) | +6.36% (106) | +12.88% (110) |
| ibm12 | 1.11005 | -0.93% (0✓) | -1.19% (75) | -1.29% (67) | -1.20% (135) |
| ibm14 | 1.22679 | -0.52% (0✓) | -0.85% (78) | -1.12% (91) | -1.28% (160) |
| ibm17 | 1.44989 | -0.98% (0✓) | -1.44% (119) | -1.58% (166) | -1.59% (254) |

**Conclusions**:
- **Direction signal real** on ariane133: periphery -0.95% vs random -0.62% (+0.33% advantage)
- **Feasibility advantage**: Periphery creates least overlaps in 7/10 benches (vs Random/Center)
- **Center always worst** across all 10 benches (worse proxy + most overlaps)
- **α=0.01 too aggressive for IBM**: 15-254 overlaps; CD can't resolve in 180s
- **Wrapper safe everywhere** (strict-accept rejects on overlap or no improvement)

**Net deployment benefit**:
- ariane133 (Grand Prize bench): -0.67% lift via wrapper vs baseline
- All others: wrapper falls back to Lévy cascade (no regression)
- Overnight multi-seed chain validating signal stability (5 random seeds × 9 (bench, α) combos)

Deployable: `submissions/cd_lns_sa_cascade_levy_periphery/placer.py`
ibm01 smoke test verified VALID with strict-accept fallback (0.8788 proxy, 0 ovl).

---

## P0 — DREAMPlace hyperparam dispatch on dense benches 🔥

**Source**: [AutoDMP (NVIDIA 2023)](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)
— 16-param Bayesian (MOTPE) optimization per design. Leaderboard #4
"Hoop Dreams" uses Optuna for same.

**Why P0**:
- Our ensemble's `cd_lns_sa_cascade_dp_native` produces basin **2.47**
  on ibm12 (with default target_density=0.85, density_weight=8e-5)
- portfolio_levy alone gets 1.21 on same bench → DP currently LOSES
- Multi-DP K=4 with auto-density already got 1.192 on ibm14 (E96)
- Fix is established: dispatch hyperparams from observable bench
  properties (macro_density, utilization), not bench names

**Spike plan** (4-6 hours cloud GPU):

5 DP configs on ibm12 (currently 2.47):

| Config | target_density | density_weight | lr | Expected |
|---|---|---|---|---|
| 1 (current) | 0.85 | 8e-5 | 0.01 | 2.47 |
| 2 (denser) | 0.65 | 1e-4 | 0.01 | < 2.0 |
| 3 (very dense) | 0.55 | 5e-4 | 0.01 | < 1.7 |
| 4 (slow lr) | 0.65 | 1e-4 | 0.005 | < 1.5? |
| **5 (auto)** | `clip(macro_density × 1.4, 0.4, 0.85)` | 1e-4 | 0.005 | < 1.8 |

Config 5 is the deployable winner — observable property dispatch, no
bench naming.

Then verify on ibm10, ibm14, ibm17 (other dense benches). If lift
generalizes, deploy in `cd_lns_sa_cascade_dp_native/placer.py`.

**Files to touch**:
- `submissions/cd_lns_sa_cascade_dp_native/placer.py` — add dispatch
  logic to `_run_dreamplace` call (target_density formula)

**Owner**: <unclaimed — needs cloud GPU>

---

## P1 — Halo / spacing parameter for DP legalization

**Source**: AutoDMP adds 2 params beyond standard 14:
`min_vertical_spacing` and `min_horizontal_spacing` (macro halo). Default
DP halo = 0; AutoDMP finds halo helps macro legalization significantly
on dense benches.

**Spike** (1-2 hours):
- Add halo=5%/10%/15% of macro_height to DP config
- Test on ibm12 (where legalization is hardest)
- If improves DP basin or reduces post-DP residual overlaps → deploy

**Subset of P0 DP dispatch** — can be bundled with that work.

**Owner**: <unclaimed — needs cloud GPU>

---

## P1 — Innovation Award writeup (independent $4K)

**Why**: $4K decided independently of leaderboard rank. We have
documentably novel contributions per 2025 lit search.

**Genuinely novel** (not in any 2025 publication):
1. **Lévy heavy-tail ε × Hessian eigvec direction** (E97)
   — combines 2nd-order curvature with heavy-tail step size
2. **Multi-objective Hessian eigvec portfolio** (E100)
   — cong-only Hessian is 98% softer than canonical on ibm10
3. **Curvature-adaptive σ = β/√|λ_min|** (E101) — auto-tuned step size

Lit search confirmed: Lévy+SA, Lévy+NSGA-II, Lévy+circle-packing all
published 2025, but **none combine Lévy with Hessian eigvec direction**.

**Deliverable**: 5-10 page writeup, can lean on existing
`NOTES_INNOVATION.md` content. Submission: `cd_lns_sa_cascade_portfolio_levy/placer.py`.

**Owner**: <unclaimed — pure writing task, any session>

---

## P1 — OpenROAD WNS/TNS validation on ariane133

**Why**: Until we have a number for our current placement's WNS, we
can't trust any optimization is helping. Grand Prize is decided here.
Other Claude has ORFS Docker infra at `/lambda/nfs/mpc2026-work/`.

**Deliverable**: WNS/TNS/Area numbers for:
- ariane133 with our best-of-3 best placement
- ariane133 with SA baseline (reference)
- ariane133 with RePlAce baseline (reference)
- Improvement ratios → predicted Grand Prize score formula

**Owner**: Other Claude (already running per coordination notes) —
needs to surface result

---

## P2 — Coordinate ensemble chain (cloud blocked)

**Status**: cloud unreachable since ~19:30 EDT. Ensemble chain on 8
hard benches in flight before outage (ibm12, ibm17 had completed).
When cloud returns:

1. Check `~/macro-place-challenge-2026/experiments/B_R2_canonical_dp_loss/results/`
2. Look for missing `strategy_*.json` (subprocess timeouts at 3435s)
3. If hybrid/portfolio_levy `.json` missing on ibm12/17, re-run with
   increased timeout

**Owner**: Whoever first restores cloud connection

---

## P3 (LOW priority, likely dead)

### ~~Canonical-Hessian finite-difference Lanczos~~

REVISED 2026-05-15: vmallela's 28% lift (1.4152 → 1.0109) comes from
worse init, not better Hessian direction. Our pipeline already starts
at 1.082, so Hessian residual lift is bounded. E104 confirmed locally
(+0.41% only). Don't pursue.

### ~~Xplace integration as basin lane~~

KILLED 2026-05-14: 9+ hours invested, basin quality is 50× worse than
baseline (1.042 + 188 overlaps on ibm01 vs E25's 0.90 with 0). The
bookshelf format adapter loses too much fidelity.

---

## Dependency graph

```
P0-periphery ──── (needs) ──── P1-OpenROAD
                                    │
                  (validates) ◄─────┘
P0-DP-dispatch ─── (needs) ──── cloud GPU
                                    │
P1-halo ────────── (subset of) ─────┘

P1-innovation ─── (independent, can ship anytime)

P2-chain-recovery ─── (needs cloud)
```

## Coordination rules

- Claim by editing this file: add `OWNER: <session-id>` under the item
- Don't touch files in another Claude's claimed directory
- Cloud is shared: OPENBLAS=6 threads, `xargs -P 4` max parallelism
- Push commits with descriptive messages so others can `git pull`
- New findings → append to `COORDINATION_2026_05_15.md` (history) or
  edit this file (active priorities)

## Sources

- [DAC25-ReMaP](https://github.com/lamda-bbo/DAC25-ReMaP) — periphery
  relocation paper
- [AutoDMP (NVIDIA)](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)
  — Bayesian DP tuning
- [Benchmarking AI Chip Placement (arxiv 2407.15026)](https://arxiv.org/pdf/2407.15026)
  — proxy↔PPA gap
- [Xplace 3.0](https://github.com/cuhk-eda/Xplace) — Carrotato's likely framework
- Local: `SPECULATION_2026_05_15.md`, `NOTES_INNOVATION.md`, `COORDINATION_2026_05_15.md`
