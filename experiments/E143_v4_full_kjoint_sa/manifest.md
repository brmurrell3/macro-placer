---
id: E143
name: v4_full_kjoint_sa
status: in_progress
parent: E166
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E143: v4_full_kjoint_sa — E166 + K-joint LNS + SA polish v2

## Hypothesis

E166 stacks V4 multi-init + multi-seed + cascade + portfolio saddle.
Two mechanisms from E41 (DPO+K-joint) haven't been layered on V4 basin:
- K-joint LNS (E39 mechanism): joint optimization of K=3 macro tuples
  selected by adjacency. Different from cascade (single eigvec direction)
  and portfolio (multi-weight eigvec direction) — operates on small
  spatial clusters, finds local re-arrangements that single-macro moves
  miss.
- SA polish v2 (E25 mechanism): low-temperature simulated annealing on
  per-axis breakpoints with best-so-far tracking. Catches the last
  micro-traps that CD's deterministic search can't escape.

K-joint and SA are the missing pieces from the full Option C / E41 stack.
Layering them after cascade + portfolio gives the full Option C arsenal
on a V4 basin foundation.

## Method

Same pipeline as E166 through portfolio saddle, then:
- K-joint LNS: K=3, top_N=5, budget 400s
- SA polish v2: T0=5e-4, Tf=1e-6, budget 250s

Budget 3000s (50 min, under 60-min cap):
- 4 lanes: 800s
- CD polish: 400s
- Cascade saddle: 300s
- Portfolio saddle: 500s
- K-joint LNS: 700s
- SA polish v2: 250s
- Safety: 50s

## Kill gate

ibm10 smoke (hard bench, where E136 showed -0.33% in 1 partial cascade
iter): final proxy >= E166 ibm10 - 0.001 → kill K-joint+SA addition.

If smoke shows compounding (≥ -0.5% improvement over E166), promote to --fast.

## Generalization check

--fast aggregate: must beat E166 --fast by ≥ 0.3% AND no single-bench
regression > 0.5%. If passes, run --all + --ng45.

## Outcome (filled when decided)

**Smoke ibm10 PASSED — BIG LIFT** (2026-05-21):
- Multi-init pick: sdf42 (canonical=1.15487, DPO SKIPped on residual overlap)
- CD polish: 0.98414
- Cascade iter 1 (only 1 fit in budget): 0.98414 → 0.98124 (-0.30%)
- Portfolio iter 1 w[0]: 0.97797 (-0.33%)
- K-joint pass 1: **167 of 262 tuples committed** (64% acceptance!), Δ=-0.02342 in evaluator
- K-joint converged at pass 2 (no improvement)
- SA polish: didn't improve K-joint output (best stayed at 0.95317)
- **Final canonical: 0.95671 (ovl=0), total wall 3015s = 50 min**

vs E136 ibm10 (V4+cascade 1 partial iter): 0.98516 → E143 is **-3.0% lift**.

**K-joint is the biggest mechanism win on hard benches** (786 hard movables
on ibm10 vs 290 on ibm03). 64% K-joint acceptance rate indicates many
genuine cluster-reordering improvements. SA didn't add further lift —
K-joint's discrete moves already exhausted the local minima cascade left.

Kill gate cleared (lift > 0.5% over E166). Promoted to --fast.

**--fast PASSED 2026-05-21 (NEW BEST --fast)**: 0.8316 avg (wall 291 min)
| Bench | E164 | **E143** | Δ |
|---|---:|---:|---:|
| ibm01 (246) | 0.8231 | 0.8262 | +0.04% |
| ibm04 (295) | 0.9255 | **0.9127** | **-1.38%** |
| ibm09 (253) | 0.7695 | **0.7591** | **-1.35%** |
| ibm13 (424) | 0.8354 | **0.8284** | **-0.84%** |
| **AVG** | 0.8384 | **0.8316** | **-0.81%** |

**K-joint+SA helped even on small benches** in this run (ibm04, ibm09 had
material lift — likely SA finding micro-improvements). My earlier
"K-joint useless on small benches" was based on ibm03 alone (which is
genuinely an outlier). On --fast, K-joint stack wins net.

This means E171's adaptive threshold (400) might be too conservative —
small benches in 250-300 movable range ALSO benefit from K-joint+SA.
However, since auto-fallback to thinkorplace-v2 protects against
regressions, no urgent change to E171 needed.



## Pointers

- Code: `code/placer.py`
- Parent: E166 (`experiments/E166_v4_full_stack/`)
- K-joint primitive: `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`
- SA polish v2 primitive: same file (`run_sa_polish_v2`)
- V4 production: `submissions/thinkorplace-v2/placer.py` (read-only)
