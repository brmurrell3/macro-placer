---
id: E53
name: dpo_basin_eval
status: falsified
parent: E41
created: 2026-05-01
decided: 2026-05-02
champion_at_time: 1.0990 (E12 verified champion); 1.08151 (E48 hybrid strongest verified candidate); 1.0848 (E41 prior-strongest verified)
fast_outcome: 0.9254 (--fast); +0.36 % vs E41 fast 0.92178; +0.56 % vs E48 fast 0.9202. Per-bench ibm01=0.9143, ibm04=0.9912, ibm09=0.8451, ibm13=0.9511. Across 4 benches, **DPO-basin phase contributed 0 accepts in 350 total restarts** (ibm01 0/121, ibm04 0/104, ibm09 0/72, ibm13 0/53). 600 s/bench × 4 of GPU MPS work (76 % of phase 7 wall) returned exactly zero proxy improvement. The −2.16 % lift seen on shortened-budget smoke (40 s CD/LNS/SA budgets, ibm01 1.0361 → 1.0137) does not reproduce at production budgets.
outcome: falsified — the basin-hopping mechanism (LNS-destroy + GPU-DPO-settler) does NOT find lift on top of fully-converged CD-LNS-SA. Smooth-proxy gradient cannot escape the local optimum that breakpoint enumeration + grid-bin LNS + breakpoint Metropolis already reach.
champion_delta: +0.36 % regression vs E41 fast (no --all run)
graduated_to: null
superseded_by: null
---

# E53: dpo_basin_eval — GPU DPO basin evaluator replacing K-joint

## Hypothesis
The K-joint LNS step (E41 step 7) enumerates `top_N^K = 5^3 = 125`
single-position candidates per K-tuple under the *exact* incremental
proxy. That is CPU-bound and ~600 s/bench. Independently, GPU DPO
(E18 init) settles a basin to a smooth-proxy local minimum in ~15-45 s
on M3 Max MPS by exploiting parallelism across `(num_macros × num_nets ×
grid_cells)` tensor ops.

**Claim:** the basin-hopping framework `(LNS-destroy → settle → exact-eval
→ accept)` works with DPO as the settler, not just enumerative reinsertion.
Replace step 7 with: destroy K macros (cost-aware), scatter them to random
feasible positions, run a short GPU DPO refinement on the destroyed
macros' positions only (frozen background), pull back to CPU, validate
under exact `compute_proxy_cost`, and accept on improvement.

If wall per "basin sample" is 10-50 ms instead of 600 s / num_kjoints,
we get **10-100× more basin samples** in the same budget. Even if each
sample is noisier than K-joint's exact enumeration, the sample-count
multiplier should dominate. And — more strategically — a faster pipeline
opens headroom for **multi-seed verification**, the missing defense
against vmallela-style 1st→12th verification drops.

## Method
Pipeline per benchmark — steps 1-6 identical to E41
(`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`):

  1. DPO best_of_v2 init.
  2. project_overlaps.
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive (≤ 2400 s).
  5. Grid-bin LNS (≤ 600 s).
  6. SA-v2 (≤ 600 s).
  7. **NEW: DPO-basin LNS (≤ 600 s).** Replaces K-joint:
     - Cost-aware destroy K=3 macros.
     - Scatter to random feasible positions on canvas.
     - Move position tensor + net_data + grid constants to MPS device.
     - Run `n_steps=80` Adam steps (lr=0.04, gamma=0.001·canvas_w,
       overlap_lambda=50.0) with **only destroyed macros' positions
       trainable** (frozen background).
     - Pull destroyed positions to CPU; clamp to feasibility.
     - Sequentially apply per-macro moves through the evaluator. If
       any single move violates `_is_legal_2d`, revert all and skip.
     - Else: compute exact proxy via incremental evaluator. Accept if
       improvement > 1e-7; revert otherwise.
     - Defensive: post-commit `compute_overlap_metrics` check (mirrors
       K-joint's defensive revert pattern).
  8. Validate (zero overlaps), preserve fixed macros, return.

GPU device: `torch.device("mps")` if available, else CPU. M3 Max has 30
MPS cores + 36 GB unified memory; net_data tensors stay resident on
device across iterations to avoid PCIe-equivalent transfer cost (which
on unified memory is ~free, but kernel-launch overhead matters).

## Kill gate
- **--fast regression:** if avg --fast > **0.9264** (E41 fast 0.92178 +
  0.5 %), kill — DPO-basin underperforms K-joint enumeration on the
  same backbone. The hypothesis (basin-hopping with GPU settler) is
  dead in this configuration; would need to re-design the settler
  (e.g., full-pose refresh instead of destroyed-only).
- **--fast match (within 0.3 %):** continue to --all even if no lift,
  *because* the wall savings buy multi-seed verification headroom.
- **--fast lift (≥ 0.3 %):** strong signal; proceed to --all and --ng45.

## Generalization check
Run --ng45 (top-7 commercial designs). E41 is 0.69022 there with 4/4
wins vs E12. If E53 matches E41 NG45 within noise (≤+0.3 %), the
basin-evaluator transfers; if E53 lifts ≥ -0.3 %, it dominates.

## Outcome — FALSIFIED 2026-05-02 00:48

### --fast result (4 jobs parallel, 8762 s wall)
| Bench | E53 | E41 fast | E48 fast | DPO-basin accepts |
|---|---|---|---|---|
| ibm01 | 0.9143 | ~0.917 | 0.8909 | 0 / 121 restarts |
| ibm04 | 0.9912 | 0.9845 | 0.9988 | 0 / 104 |
| ibm09 | 0.8451 | 0.8415 | 0.8415 | 0 / 72 |
| ibm13 | 0.9511 | 0.9497 | 0.9497 | 0 / 53 |
| **avg** | **0.9254** | 0.92178 | 0.9202 | **0 / 350** |

All VALID, zero overlaps. **DPO-basin phase contributed exactly zero
accepts on every benchmark.** 600 s/bench × 4 benches × ~80 ms/restart
= 6 hours of GPU work, zero proxy improvement.

### Why it failed
The smoke test on shortened budgets (CD 20 s, LNS 10 s, SA 10 s)
showed −2.16 % lift on ibm01 (1.0361 → 1.0137). At production budgets
(CD 2400 s, LNS 600 s, SA 600 s), the prior phases converge tightly
enough that **smooth-proxy GPU gradient cannot find improvements past
CD-LNS-SA's local optimum**. Per-iteration internal state confirms:
ibm13 SA-v2 found best=0.94224 at t=0 s (LNS exit was already SA's
best), chain wandered up to 0.95079, restoration restored 0.94224 →
DPO-basin then took it as init and could not improve.

### Lesson for the project (and the writeup)
**GPU DPO basin polish is not additive on top of fully-converged
CD-LNS-SA.** The mechanism would need to *replace* CPU CD itself
(architectural restructure), not act as a polish at the end. The
verification-defense angle (multi-seed averaging) remains valid but is
mechanism-independent — it's being tested via E53_multiseed_hybrid
separately.

### Falsification record (paper-relevant)
This is the first directly negative result on **GPU acceleration
inside our pipeline**. Adds to the Falsification Record alongside:
E14 (SA without best-tracking), E32 (SAM-CD), E42 (K=4 K-joint),
E43 (longer K-joint), E44 (spatial K-joint), E45 (multigrid).

## Pointers
- Code: `code/cd_lns_sa_dpo_basin.py` (defines `CDLNSSADPOBasinPlacer`).
- Parents: E41 (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`),
  E18 (DPO init / smooth-proxy fns), E39 (LNS-gridbin, SA-v2,
  `_cost_aware_destroy`, `_is_legal_2d`).
- Discussion: extends the basin-hopping framework E41 established;
  motivated 2026-05-01 by verification-risk concern (vmallela 1st→12th)
  + observation that DPO settle is 40-100× faster than CD-to-plateau
  per basin.
