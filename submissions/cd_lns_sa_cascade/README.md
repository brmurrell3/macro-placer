# cd_lns_sa_cascade — wall-safe E84 cascading saddle escape

E48 plateau pick (best of {E25, E41}) followed by **cascading** Hessian
saddle escape on the smooth proxy. Iterates: Lanczos smallest-algebraic
eigenvector → ±ε perturbation → CD-adaptive polish → repeat from new
state until (a) Hessian's smallest eigenvalue ≥ 0 (true local min reached),
(b) cascade iteration produced no improvement, (c) max_iters or
(d) wall budget exhausted.

## Verified result (no wall cap)

Canonical `--all` via `experiments/E84_cascading_saddle/code/loader_placer.py`:

| Metric | Value |
|--------|------:|
| Avg proxy (17 IBM) | **1.0612** |
| vs E48 (1.08151) | **−1.88 %** |
| vs E74 (1.0666) | **−0.51 %** |
| vs RePlAce (1.4578) | gap +27.2 % |
| Overlaps | 0 / 0 / 0 ... (all clean) |

## Wall budget

`budget_seconds=3300.0` default (55 min/bench, fits the 1-hr/bench
partcl cap with 5-min margin).

Internal allocation when `budget_seconds=B`:
- E25 (Phase 1): cd=B·0.20, lns=B·0.06, sa=B·0.06 ≈ 32 % budget
- E41 (Phase 2): cd=B·0.20, lns=B·0.05, sa=B·0.05, kjoint=B·0.05 ≈ 35 % budget
- Cascading saddle (Phase 3): remainder ≈ 33 % budget
- Skip rules: E41 skipped if <300s left; cascade skipped if <60s left;
  cascade stops cleanly at deadline — last good state returned.

Pass `budget_seconds=None` for offline mode (matches the unbudgeted
1.0612 result).

## Pipeline

```
benchmark
  → Phase 1: E25 (SDF + CD + LNS + SA-v2)                ← budget-capped
  → Phase 2: E41 (DPO + CD + LNS + SA-v2 + K-joint)      ← budget-capped, skip if budget tight
  → Phase 3: cascading saddle on min(E25, E41) plateau   ← deadline-bound
  → final: best-of {E25, E41 if run, cascade}            ← overlap-validated
```

## Compared to siblings

- `cd_lns_sa_hessian/placer.py` (E74) — single saddle pass, slightly
  weaker proxy (1.0666 unbudgeted) but matches cascade on tight-budget
  benches where only 1 cascade iter fits.
- `cd_lns_sa_hessian_dp/placer.py` — E74 + optional DREAMPlace lane
  (subprocess), gracefully no-op if DREAMPLACE_ROOT not set.

## Smoke

```bash
# Local, fast — proves deadline + multi-iter cascade work
uv run python -c "
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer
bench, plc = load_benchmark_from_dir(str(find_benchmark_dir('ibm03')))
p = CDLNSSACascadePlacer(budget_seconds=600.0).place(bench)
"
```

## Going wide

```bash
# All 17 IBM via the canonical evaluator, --jobs 4 (~5-6 hr wall)
uv run evaluate submissions/cd_lns_sa_cascade/placer.py --all --jobs 4 --json --hypothesis cascade_walltight

# NG45 (4 designs, ~2.5 hr wall under --jobs 4)
uv run evaluate submissions/cd_lns_sa_cascade/placer.py --ng45 --jobs 4 --json
```

## Promote-to-champion gate

- `--all` avg ≤ 1.080 with zero overlaps everywhere
- Every bench wall ≤ 55 min (under 60-min cap with margin)
- `--ng45` avg ≤ 0.69 with no NG45 regression vs E48 (0.6922)
- E74 (1.0666) is the proxy bar; cascade variant should match or beat it
  under budget enforcement
