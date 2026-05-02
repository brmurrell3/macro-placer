---
id: E60
name: replica_sa
status: in_progress
parent: E25
created: 2026-05-02
decided: null
champion_at_time: 1.0990 (E12; E48 hybrid 1.08151 strongest verified candidate; ADR-011 *Proposed*)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E60: replica_sa (replica exchange SA-v2)

## Hypothesis
E25/E41/E48's SA-v2 phase uses a single chain at T₀ = 5e-4 (cold) with
geometric anneal to Tf = 1e-6. E26 falsified longer SA budget — extra
time at the same T schedule produces only float-drift fluctuation.
This is the textbook signature of a chain stuck at a basin floor that
COLD MOVES alone cannot escape.

**Replica exchange (parallel tempering)** is the standard physics
technique for this exact problem class:
- K parallel chains at temperatures T₁ < T₂ < ... < T_K.
- Hot chains (high T_K) explore broadly; cold chains (low T₁) refine.
- Every N moves, attempt swap between adjacent chains via Metropolis
  criterion: accept(swap_k_kp1) = min(1, exp((1/T_k - 1/T_kp1) ·
  (E_kp1 - E_k))).
- Best-tracked across all chains.

The hot-chain exploration finds NEW basins that the cold chain can then
refine, breaking out of the SA-v2 plateau. **This is how spin-glass /
protein-folding / simulated annealing breakthroughs happened in the
1990s after vanilla SA saturated.**

E26 falsified longer-budget SA. Replica exchange is structurally
different — it adds new chains at higher T, not more time at the same T.

## Method
Replace E25's SA-v2 phase with a replica-exchange SA-v2 phase. K=4
chains at:
  - T₀ = 5e-4 (cold; same as E25/E41 baseline)
  - T₀ = 5e-3 (warm; ~10 % accept of typical worsening moves)
  - T₀ = 5e-2 (hot; ~50 % accept of typical worsening)
  - T₀ = 5e-1 (very hot; nearly random walk)

All chains anneal geometrically to Tf = 1e-6.

Round-robin dispatch:
- Each chain has its own IncrementalProxyEvaluator (independent state).
- Each chain starts from the post-LNS state.
- Per round, each chain runs N=200 moves at its current T.
- After all K chains complete a round, attempt one swap between a random
  pair of adjacent chains via Metropolis.
- If accepted, swap chain TEMPERATURES (mathematically equivalent to
  swap placements, but cheaper since evaluator state stays put).
- Total budget: 600 s split across K chains = 150 s per chain (matches
  E25/E41 SA budget; no wall penalty).

Best-tracked: across all chains, the lowest-proxy placement seen at any
time is preserved and returned at the end.

Pipeline: SDF init → CD plateau (≤2400 s) → grid-bin LNS (≤600 s) →
replica-exchange-SA-v2 (≤600 s, K=4) → validate.

## Kill gate
- **Regression on --fast:** if avg --fast > E25 fast 0.9336 + 0.5 % →
  kill (something broke).
- **No lift over E25 SA-v2:** if avg --fast within ±0.1 % of E25 fast
  0.9336, mark marginal — replica exchange doesn't help on top of
  CD+LNS on these benchmarks.

## Generalization check
If --fast lifts ≥ 0.3 % over E25 fast 0.9336, run --ng45.
If --ng45 lifts, queue --all.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/replica_sa.py` (replica-exchange SA-v2 implementation).
- Code: `code/cd_lns_replica_sa.py` (placer using SDF + CD + LNS +
  replica-SA).
- Parents: E25 (CDLNSSA pipeline; SA-v2 baseline). E26 (longer SA
  budget falsification — motivates this structurally-different
  approach).
- Discussion: research-survey commit message mentions parallel tempering
  as the well-validated cross-field technique.
