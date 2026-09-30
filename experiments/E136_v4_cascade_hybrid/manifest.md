---
id: E136
name: v4_cascade_hybrid
status: in_progress
parent: E127
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E136: v4_cascade_hybrid — V4-Gaussian basin + cascade saddle escape

## Hypothesis

thinkorplace-v2 (V4 + Gaussian density + extended CD polish) finds an Adam
descent basin with proxy ~0.984 combined. Cascade saddle escape (E84) was
proven on Option C's SDF/DPO basin and on E110 smooth-bbox basin, but has
NEVER been run on the V4-Gaussian basin. The V4 basin is C∞ smooth and
likely has different soft-mode structure than the SDF or DPO basins — the
Hessian's smallest eigenvector may unlock a different escape direction.

Even a single saddle iteration (one ε perturbation along the soft mode +
short CD polish) would add ~60-90s of wall and tells us whether the basin
has unexploited curvature. If `λ_min` is already ≥ 0 we learn V4+CD
already reaches a true local min — a useful negative result.

## Method

Reuse the thinkorplace-v2 Placer to produce the basin placement (do NOT
modify it). Then in a wrapper:

1. Run V4 + CD polish via the production Placer with a reduced
   `cd_polish_s` (so we leave wall budget for saddle).
2. Apply `cascading_saddle_escape` (E84) with conservative settings:
   - `max_iters=2`
   - `eps_values=(0.5, 1.5)` (small probe + large jump; no Lévy yet)
   - `polish_budget=120s`
   - `total_budget_s = remaining wall - 30s safety`

If saddle finds improvement, the wrapper returns the saddle output; else
it returns the V4+CD basin (no regression possible).

## Kill gate

ibm03 smoke: V4+cascade proxy >= V4-baseline proxy + 0.005 (i.e., no
improvement within noise) → kill the hypothesis. The hybrid must add at
least 0.5% lift over a single bench to justify the wall cost.

## Generalization check

`--fast` (4 benches) must average ≤ V4-baseline minus 0.2% AND no
single-bench regression > 0.5%. If passes, run `--all` and compare to
thinkorplace-v2 0.984.

## Outcome (filled when decided)

**Smoke ibm03 PASSED** (2026-05-21):
- V4 basin: proxy=0.90430 ovl=0 wall=473s
- λ_min = -0.219 (NEGATIVE — basin has soft Hessian modes; saddle escape well-motivated)
- Cascade iter 1: ε=0.5 → 0.90311 (-0.13%), ε=1.5 → 0.89842 (-0.65%)
- Cascade iter 2: ε=0.5 → 0.89665, ε=1.5 → 0.89642, ε=-1.5 → **0.89530**
- **Final: 0.89530 (LIFT -1.00% on ibm03)**, wall=954s
- vs E163 same bench: 0.90407 (Lévy saturated immediately at iter 1 — smallest Lévy ε=0.824 missed the eps=0.5 sweet spot)

**Mechanism confirmed**: V4+CD basin is NOT a true local min on smooth proxy;
cascade saddle escape along softest eigvec descends to a lower basin.
Generalization probe queued on ibm10. If lift holds on ≥2/4 hard benches,
candidate for partial integration.

Kill gate cleared (lift >= 0.5%) on ibm03; lift is bench-dependent.

**--fast (ibm01, ibm04, ibm09, ibm13) 2026-05-21**:
| Bench | V4+CD basin | Cascade final | Lift |
|---|---:|---:|---:|
| ibm01 | 0.83105 | 0.82989 | −0.14% |
| ibm04 | 0.92536 | 0.92407 | −0.14% |
| ibm09 | 0.76569 | 0.76569 | 0.00% (saddle saturated) |
| ibm13 | 0.84265 | 0.84006 | −0.31% |
| **AVG** | **0.84119** | **0.83993** | **−0.15%** |

Cascade contributes only −0.15% avg on --fast, much smaller than ibm03's
−1.00% singleton. ibm03 was an outlier bench. The bigger lever is
multi-seed basin diversity (see E164) — cascade is the smaller-mag polish.
Total --fast wall = 3431s = 57 min, well under 60 min/bench cap.

E136 alone is not a strong shipping candidate; absorbed into E164 stack.



## Pointers

- Code: `code/v4_cascade_placer.py`
- Cascade primitive: `experiments/E84_cascading_saddle/code/cascading_saddle.py`
- V4 production placer: `submissions/thinkorplace-v2/placer.py` (read-only)
- Discussion: TBD
