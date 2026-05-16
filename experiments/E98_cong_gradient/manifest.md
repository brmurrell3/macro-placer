---
id: E98
name: cong_gradient
status: falsified
parent: E84
created: 2026-05-13
decided: 2026-05-13
champion_at_time: 1.078
outcome: cong_polish ibm14: canon 1.20630 → 1.28861 (+6.8%) during L-BFGS descent; 17 residual overlaps post-project; fallback to init
champion_delta: +6.8% (per-bench, on test bench ibm14)
graduated_to: null
superseded_by: null
---

# E98: cong_gradient — Differentiable congestion-gradient polish

## Hypothesis
Congestion is 80%+ of proxy on hard benches (ibm10/12/14/17). Direct gradient
descent on the smooth congestion term should find polish moves that CD's
discrete swap moves miss. Optimize ONLY the congestion term (not full proxy),
with overlap + boundary Lagrangian penalties, via L-BFGS with strong-Wolfe
line search.

## Method
Pipeline: cascade plateau → L-BFGS gradient descent on
`(congestion + λ·overlap_penalty + λ·boundary_penalty)` with λ annealed
geometrically → project_overlaps → CD-adaptive recovery polish.

Gradient extraction:
`cong_grad = (total_smooth_full - total_smooth_no_cong) * 2.0` (since
total_smooth = wl + 0.5·density + 0.5·cong).

## Kill gate
On ibm14 cascade plateau (canon=1.206), if 60-step L-BFGS run gives final
canon ≥ init, kill.

## Outcome (FALSIFIED 2026-05-13)

L-BFGS descent on smooth congestion + λ·overlap on ibm14 cascade plateau:

| Step | Loss | Cong (smooth) | Canon (canonical) | λ_overlap |
|------|-----:|--------------:|------------------:|----------:|
| 0    | 4.41 | 4.41 | **1.20630** | 1000 |
| 5    | 4.35 | 4.34 | 1.21597 | 1212 |
| 10   | 4.25 | 4.18 | 1.28939 | 1468 |
| 15   | 4.26 | 4.18 | **1.28861** | 1778 |

Smooth congestion descended by 5% (4.41 → 4.18), but **canonical proxy ROSE
by 6.8%** (1.206 → 1.289). post-project: 17 residual overlaps that
`project_overlaps` could not resolve; code fell back to init.

**Root cause**: Smooth `_rudy_congestion` uses bbox-uniform-spread; canonical
`plc.get_congestion_cost()` uses per-net trace-route. Their gradient
directions are weakly correlated on hard benches — descending smooth cong
makes canonical cong worse. Same E88/E95 failure pattern, now reproduced for
a third time.

This is the *first* test where the wrong-direction issue produces actual
overlaps (because the placement is being moved aggressively toward smooth's
optimum, which isn't even feasible under canonical density).

### What would unblock this
Build differentiable RUDY based on per-net trace-route, not bbox-spread. That
requires reimplementing the canonical congestion in torch (multi-day
engineering). Filed as future work; not pursued in this experiment.

## Pointers
- Calibration probe: `code/calibrate_cong.py` (killed at 17 min — autograd
  through `_rudy_congestion` × 16 perturbations was too slow for spike)
- Failed polish: `code/cong_polish.py`
- Log: `/tmp/cong_polish_ibm14.log` (full 173s descent + project trace)
- Related: E88 (FALSIFIED), E95 (in_progress)
