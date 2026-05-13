# Submission entry

**File:** `placer_adaptive.py`
**Class:** `CDLNSSACascadeAdaptivePlacer`

## Run

```bash
git submodule update --init external/MacroPlacement
uv sync
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --ng45 --json
```

## Algorithm

Best-of pipeline combining SDF and DPO initialization basins with iterated
Hessian saddle escape on the smooth proxy. Per benchmark:

1. **E25** — SDF init + CD coordinate descent + grid-bin LNS + SA-v2 polish.
2. **E41** — DPO init + CD + LNS + SA-v2 + K-macro joint LNS (K=3, top-N=5).
3. **Cascade saddle escape** — on the better of E25/E41: Lanczos
   smallest-algebraic eigenvector of the smooth-proxy Hessian → ±ε
   perturbation along the soft mode → CD polish → iterate until the
   smallest eigenvalue ≥ 0 (true local min) or budget exhausts.

CD plateau detection is property-tuned by canvas area (IBM-class vs
NG45-class commercial). Single global algorithm; no per-benchmark
hyperparameters; no benchmark-identity dispatch; CPU-only (PyTorch for
autograd HVP only, no GPU required); no external solvers.

Default `budget_seconds=3000` (50 min/bench, safe under the 60-min cap
on EPYC). Output is overlap-validated with a defensive fallback chain
(if cascade fails validation → fall through to E41 → E25, both always
overlap-free by construction).

## Files

| File | Role |
|------|------|
| `placer_adaptive.py` | Entry — `CDLNSSACascadeAdaptivePlacer`, canvas-area-tuned wrapper |
| `placer.py` | Base — `CDLNSSACascadePlacer`, the cascade pipeline |

## Verified results

| Mode | Avg proxy | Overlaps | Max wall |
|------|----------:|---------:|---------:|
| `--all` (17 IBM ICCAD04) | **1.0771** | 0 | ~52 min |
| `--ng45` (4 NG45 commercial) | **0.6870** | 0 | ~50 min |

Verified 2026-05-13 on AMD EPYC 9655P / 30 cores / Ubuntu, under
`OPENBLAS_NUM_THREADS=8`, 60-min/bench cap, jobs=4.
