# Submission entry — Option A (cascade, no external deps)

**File:** `placer_adaptive.py`
**Class:** `CDLNSSACascadeAdaptivePlacer`

> Option A is the safer Tier-1 candidate. **Option B**
> (`submissions/cd_lns_sa_cascade_dp_lane/placer.py`) is verified strictly
> better on both gates (IBM 1.06650 / NG45 0.68086) but requires
> DREAMPlace; ship Option A if the eval environment lacks DREAMPlace.
> See [`../README.md`](../README.md) for the A-vs-B comparison.

## Run

```bash
git submodule update --init external/MacroPlacement
uv sync
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --ng45 --json
```

`OPENBLAS_NUM_THREADS=8` is load-bearing on EPYC — without it `numpy`
defaults to 1 thread (3-9× slowdown on the routing inner loop).

## Algorithm

Per-bench best-of pipeline combining SDF and DPO initialization basins
with iterated Hessian saddle escape on the smooth proxy:

1. **E25 lane** — SDF init + CD coordinate descent + grid-bin LNS +
   SA-v2 polish.
2. **E41 lane** — DPO init + CD + LNS + SA-v2 + K-macro joint LNS
   (K=3, top-N=5).
3. **Plateau pick** — argmin canonical proxy over {E25, E41} outputs.
4. **Cascading Hessian saddle escape** on the picked plateau:
   smooth-proxy Hessian-vector product (`torch.autograd.functional.hvp`)
   → Lanczos smallest-algebraic eigenvector (scipy `eigsh` with
   `LinearOperator`) → ±ε perturbation along the soft mode → CD-adaptive
   polish → iterate until smallest eigenvalue ≥ 0 (true local min),
   no proxy improvement, or budget exhausts.

CD plateau detection is property-tuned by canvas area (IBM-class vs
NG45-class commercial). **Single global algorithm; no per-benchmark
hyperparameters; no benchmark-identity dispatch** (only canvas-area
property dispatch); CPU-only (PyTorch for autograd HVP only, no GPU
required); no external solvers.

Default `budget_seconds=3000` (50 min/bench, 10-min safety margin under
the 60-min cap on EPYC). The 2026-05-16 budget-management fix uses
rolling `avg_iter_wall × 1.2` to predict next-iter cost (vs the prior
fixed 60s safety floor), keeping the cascade from stopping early when
one more iter could fit.

Output is overlap-validated with a defensive fallback chain: if cascade
fails validation → fall through to E41 → E25, both always overlap-free
by construction.

## Files

| File | Role |
|------|------|
| `placer_adaptive.py` | Entry — `CDLNSSACascadeAdaptivePlacer`, canvas-area-tuned wrapper |
| `placer.py` | Base — `CDLNSSACascadePlacer`, the cascade pipeline (wall-safe E84) |
| `README.md` | This file |

## Verified results

| Mode | Avg proxy | Overlaps | Max wall |
|------|----------:|---------:|---------:|
| `--all` (17 IBM ICCAD04) | **1.07820** | 0 | ≤ 57 min |
| `--ng45` (4 NG45 commercial) | **0.68102** | 0 | ≤ 52 min |

Verified 2026-05-13 / 2026-05-16 on AMD EPYC 9655P (cloud cross-validation
on AWS c6a.4xlarge spot, the partcl-equivalent hardware), under
`OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`,
60-min/bench cap, `--jobs 4`.

**vs baselines:**
- vs RePlAce 1.4578 → **−26.0 % aggregate** on `--all`.
- vs public leaderboard reference 1.1172 → **−3.5 %**.
- vs every verified leaderboard entry → **≥ −13 % aggregate**.
- vs Option B (`cd_lns_sa_cascade_dp_lane`) 1.06650 → **+1.1 %** on `--all`
  (Option B is verified strictly better but adds a DREAMPlace dependency).

## Composite (Tier-1 + Tier-2 reference)

Composite avg across 21 benches (17 IBM + 4 NG45): **0.998**.
For comparison, Option B is **0.993** (−0.5 % composite).

## Algorithm provenance

The cascade base is `submissions/cd_lns_sa_cascade/placer.py`
(`CDLNSSACascadePlacer`), the wall-safe descendant of E84 cascading
saddle escape that achieved the **1.0612 uncapped IBM** result on M3 Max
(2026-05-10) — the algorithm's theoretical ceiling. PATH A post-A1
implementation speedup (commits `59a7a8b`, `53b4a26`, `af520c3`,
`9df5ac2`, `f2269b3`) reduced CD wall by **5.36×** on ibm10, closing
most of the cap-vs-ceiling gap to land at the verified `--all` 1.07820
under the 60-min/bench cap.

Champion lineage that built up the cascade: E12 (1.0990) → E48 hybrid
(1.08151, ADR-011) → E74 Hessian saddle (1.0666, ADR-012) →
E84 cascade (1.0612 uncapped) → A4-v2 cascade-adaptive (1.07820
capped). See [`../../docs/approach.md`](../../docs/approach.md) for the
full mechanism description.
