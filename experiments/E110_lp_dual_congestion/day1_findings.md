# v2 Day 1 Findings (May 18, evening)

## Headline

Both planned breakthrough hypotheses (H1 LP-dual, H2 PA / multi-start)
are empirically marginal-to-negative at the LNS / polish stage. **SA-v2
with best-so-far tracking is genuinely hard to beat at this regime.**

Pivot: H2 → **extended single-chain SA-v2 post-cascade polish**.
Strict-improvement bet on budget reallocation rather than method
replacement. Test on ibm04 in flight.

## Quantitative results

### H1 (LP-dual congestion gradient)

| Test | Bench | Mode | Δ vs cost-aware |
|---|---|---|---|
| Calibration probe (Spearman π vs canonical Δproxy) | ibm01 | — | +0.32 (borderline) |
| Isolated LNS 120s | ibm01 | pure_lp | +0.087% |
| Isolated LNS 120s | ibm01 | mix | **-0.238%** |
| Isolated LNS 240s | ibm13 | mix | -0.047% |

**Verdict**: LP-dual signal is below the variance floor of LNS itself.
Mean lift across three configurations: roughly 0%. Spearman 0.32 says
there IS signal, but cost-aware probe captures most of the same
information. Falsified for this LNS regime.

**Why theory diverged from practice**: smooth-RUDY gradient mismatch
(documented failure of E88/E95/E98) only matters for *gradient-descent*
methods. LNS destroy ranking is a *categorical selection* — and the
cost-aware move-to-center probe already does a one-step canonical
forward eval. LP-dual's improvement in macro ranking is too small to
change the basket of selected macros meaningfully.

### H2 (population annealing → multi-start → extended SA)

Standalone smoke on ibm01, identical 180s polish budget from CD plateau:

| Method | Plateau | Final | Δ from plateau |
|---|---|---|---|
| SA-v2 single-chain | 1.10453 | **1.01986** | -8.5% |
| PA (12 replicas) | 1.10487 | 1.05824 | -4.2% |
| Multi-start SA-v2 (N=4) | 0.99939 | 0.97471 | -2.5% |

PA loses to SA-v2 by **3.7%** at same budget. Resample/clone overhead
eats moves (28,800 vs 32,560); each replica only gets ~2400 moves total.
At polish-stage temperatures the landscape isn't rugged enough for
replica diversification to recover the depth loss.

Multi-start: 4 chains converged to within 0.17% of each other —
**essentially zero diversity benefit**. SA-v2's best-so-far tracking
already explores broadly enough that multiple seeds find the same basin.

**Verdict**: PA and multi-start both falsified at the polish stage.
Per-chain depth wins over replica diversity here.

## Current plan: H2 → extended SA-v2 post-cascade polish

The 8.5% lift SA-v2 produces from CD plateau in 180s suggests the
cascade pipeline's internal SA budget (~90s out of 3300s) leaves
substantial improvement on the table. Reserving 600s for a dedicated
**post-cascade** SA-v2 polish, starting from the cascade output (not
CD plateau), is a strict-improvement bet:

- If the cascade output already captures all SA can find → polish is a
  no-op, ships identical to Option C.
- If the cascade output has any unexplored basin → polish lifts by
  however much remains.

In flight: ibm04 base vs v2-extended at 1200s budget. ETA ~25 min.

## v2 ship decision tree

| Scenario | Ship |
|---|---|
| ibm04 extended-SA shows ≥ 0.5% lift over base | `MPC_V2_H2_VARIANT=extended`, full --fast on May 19 |
| ibm04 extended-SA ≥ 0% but < 0.5% | run --fast anyway; ship whichever wins |
| ibm04 extended-SA regresses or ties | ship Option C unchanged |
| Either case: H1 NOT enabled | per the falsification above |

Floor floor: Option C at 1.0575 vs current submitted 1.0771 = **-1.8%**
guaranteed even with no algorithmic change.

## Cross-disciplinary lessons

The user asked "what cross-disciplinary methods apply to congestion?"
We pursued the two most theoretically-grounded answers:
1. **LP duality** (transportation/OR) — Magnanti-Wong, multi-commodity
   flow capacity duals.
2. **Population annealing** (statistical physics) — Hukushima-Iba 2003,
   provably escapes spin-glass landscapes.

Both have solid theoretical backing for problems with the structural
properties of macro placement. **Both empirically fail at our scale.**

Why theory and practice diverged:
- **LP-dual**: cost-aware probe already does forward canonical eval per
  candidate, so the LP-dual extra info is redundant in LNS context.
- **PA**: budget allocated to SA polish (~90s inside cascade) is too
  small for PA's startup cost (24 replicas × ~1s build = 24s baseline).
- **Multi-start**: SA-v2's best-so-far tracking already provides
  pseudo-diversity within a single chain.

The principled answer to "where's the breakthrough?" appears to be:
the breakthrough is **budget reallocation** (more SA, less other), not
method replacement. Boring but empirically correct.
