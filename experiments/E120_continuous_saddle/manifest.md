---
id: E120
name: continuous_saddle
status: in_progress
parent: E111
created: 2026-05-20
decided: null
champion_at_time: 1.0575
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E120: continuous-space Hessian saddle escape inside Adam descent

## Hypothesis

Our cascade saddle escape (E74/E84) is novel on the CD-combinatorial
plateau, but has never been applied inside the **continuous** Adam
descent loop of the V3 smooth global placer. After Adam converges to a
local minimum of the smooth proxy, the Hessian's smallest-eigenvalue
direction reveals a "saddle" along which to perturb so that resuming
Adam can find a strictly lower basin.

Mechanism: at each saddle stage, compute H = ∂²f/∂p² of the V3 smooth
proxy via `torch.autograd.functional.hvp`, find the smallest-algebraic
eigenvector v via scipy.sparse.linalg.eigsh (Lanczos with
LinearOperator) on movable coordinates only, perturb positions ±ε along
v for several ε ∈ {0.3, 1.0, 3.0} × cell-size, and resume Adam for
100-200 steps from each candidate. Accept the best resulting smooth
proxy and iterate 1-2 more times if budget remains.

This is the continuous-space analogue of E84 cascading saddle escape,
which already works on the CD plateau. The two operate on completely
different landscapes (Adam smooth vs CD combinatorial) and address
completely different failure modes (Adam basin choice vs CD local-move
plateau).

## Method

1. `code/continuous_saddle_placer.py`:
   - **Phase A**: V3 Adam descent for ~300-500 steps with γ-anneal and
     overlap-λ ramp (identical to V3Minimal).
   - **Phase B**: Hessian saddle escape on smooth proxy:
     - Movable-coord LinearOperator wrapping
       `torch.autograd.functional.hvp(proxy.cost, state, v)`.
     - `scipy.sparse.linalg.eigsh(op, k=1, which="SA", maxiter=500)`.
     - For each ε ∈ {0.3, 1.0, 3.0} and sign ∈ {+1, -1}, perturb along
       the eigvec scaled to that fraction of canvas, then resume Adam
       (with γ at end-of-anneal value) for 100-200 steps. Compute the
       smooth proxy of the polished candidate.
     - Keep the best candidate by smooth proxy.
   - **Phase C**: repeat Phase B 1-2 more times until budget is spent
     or improvement saturates.
   - **Phase D**: `greedy_macro_legalize` → `run_cd_adaptive` polish.
   - Default budget: 720s for one bench (matches ovl10 720s baseline).

2. `code/test_ibm17.py`: head-to-head V3Min ovl10 720s vs E120 on ibm17.
3. `code/saddle_descent.py`: stand-alone helper that does Phases A-C
   without legalization, for fast iteration.

## Kill gate

- ibm17 + CD600s ≥ 1.16 (V3Min baseline ≈ 1.20). If the continuous
  saddle escape doesn't lift ibm17 below 1.16, the novel direction is
  not measurably better than vanilla Adam and we fold the experiment.

## Generalization check

If ibm17 passes, run `--fast` (4 benches, ~10s) then `--all` (17 IBM)
with the same continuous-saddle placer. Must beat the V3Min ovl10 720s
baseline (~1.005) on `--all` average to graduate.

## Outcome — preliminary 2026-05-20

### The saddle escape works on the smooth basin (consistent -1.7%)

Across every bench tested, the V3 smooth Hessian's smallest-algebraic
eigenvector returns λ_min < 0 (saddle direction). Perturbing along it
and resuming Adam converges to a strictly lower smooth basin:

| Bench | n_macros (hard) | λ_min | smooth basin | after saddle | Δ smooth |
|---|---:|---:|---:|---:|---:|
| ibm01 | 1140 (246) | -1.57e-02 | 0.86810 | 0.84766 | -2.35 % |
| ibm03 | 1438 (290) | -4.51e-03 | 0.98594 | 0.96891 | -1.73 % |
| ibm17 | 2604 (760) | -3.10e-03 | 1.28320 | 1.26116 | -1.72 % |

Six perturbations (±, ε ∈ {0.3, 1.0, 3.0} × cell-size) all land in a
band ±0.001 of each other → Adam consistently re-converges to a
common, lower basin, not just to noise.

### Lift partly survives CD polish (matched-budget ablation, ibm01 + ibm03)

`code/ablation_fair.py` runs the same V3+legalize+CD pipeline with
vs. without saddle escape, identical CD-polish budget (120 s wall),
zero overlap in every run:

| Bench | A: V3 final | B: V3+saddle final | Δ(B − A) abs | Δ(B − A) % | extra wall |
|---|---:|---:|---:|---:|---:|
| ibm01 | 0.85250 | **0.84356** | -0.00894 | **-1.05 %** | +65 s |
| ibm03 | 0.92041 | **0.90769** | -0.01273 | **-1.38 %** | +56 s |

The continuous saddle escape gives a consistent ~1-1.4 % proxy-cost
lift on small/medium benches at matched CD budget, costing ~60 s of
extra wall (eigsh + 6 perturbation+resume attempts).

### What does not work — direct head-to-head against V3Min at 720 s

`code/test_ibm17.py` ran E120 at the same 720 s budget as the
V3Min-ovl10-720s champion. E120 lifted the smooth basin from 1.28320
to 1.26116 (-1.72 %), but the saddle stages consumed ~395 s of the
budget so only ~165 s remained for CD polish. Under that truncated
polish E120 ibm17 = **1.21229**, versus V3Min ovl10 720 s baseline of
**1.200**.

So on ibm17 alone, at matched 720 s, E120 trails V3Min by +1 %. The
saddle escape is a real, measurable basin lift but it must be paid
for with extra wall-clock; budget-equal comparisons should give the
saddle variant extra time (or shrink Phase A / number of attempts).

### Kill-gate status

Manifest kill gate: "ibm17 + CD600s ≥ 1.16". We hit 1.21229 at CD164s
on ibm17, so the gate is **not met at 720 s budget**. The novel
mechanism works (smooth basin -1.7 %, polish -1.4 % on ibm03), but
it does not beat V3Min when forced to share the same wall budget on
the hardest bench. Status: **in_progress** — needs (a) a larger
total budget where saddle+polish both get full time, or (b) a
cheaper saddle escape (fewer perturbation attempts, parallel HVP).

## Pointers
- Code: `code/continuous_saddle_placer.py` (the placer)
- Code: `code/ablation_fair.py` (matched-CD ablation; ran on ibm03)
- Code: `code/test_ibm17.py` (head-to-head vs V3Min at 720 s)
- Results: `results/ablation_fair_ibm03.json` (-1.38 % at matched CD120s)
- Results: `results/ibm17_smoke.log` (E120 1.21229 at CD164s)
- Submission wrapper: `submissions/e120_continuous_saddle/placer.py`
- Parent: experiments/E111_per_net_trace_congestion (V3)
- Cousin: experiments/E84_cascading_saddle (combinatorial saddle on CD plateau)
- Cousin: experiments/E74_hessian_saddle (original Hessian eigvec primitives)
