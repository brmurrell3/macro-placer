# Submissions

Competition-ready placers, organized by lineage.

## The competition entry

**`cd_lns_sa_cascade/placer_adaptive.py`** — `CDLNSSACascadeAdaptivePlacer`.

### Pre-A1 baseline (2026-05-11)

| Metric | Value | Reference |
|--------|------:|-----------|
| IBM avg `--all` | 1.137 | cloud EPYC, 60-min/bench cap |
| NG45 avg `--ng45` | 0.6925 | cloud EPYC |
| Max wall | 57 min | safe under 60-min partcl cap |
| Overlaps | 0 / 17 IBM | canonical eval, zero on every bench |
| vs RePlAce 1.4578 | −22 % | published baseline |

### Post-A1 (2026-05-12, verification in flight — A4)

The cascade CD inner loop received a 5.36× speedup via four commits to
`macro_place/incremental_evaluator.py` (`delta_cost`,
`delta_cost_axis_batch`, `_net_cong_contrib_flat`, `commit`) plus six
LNS-helper sites in E25 / E18 / E39 (see `MECHANISM.md` for the full
account). The cascade saddle now reaches 2–3 iterations under the cap
instead of 1.

**Partial A4 results (first 4 IBM benches landing):** 0.88864, 0.94975,
1.07339, 0.99148 → partial avg ~0.97. Full 17-bench number expected
1.05–1.08 IBM.

**Partial NG45 (3/4 landed):** 0.67644, 0.65146, 0.73743 → partial avg
~0.69. Full 4-bench number expected ≤ 0.69.

Numbers in this table will be updated to verified A4-v2 aggregates once
the wall-safe LNS-delta variant completes (chained behind A4-v1 on
`ubuntu@129.213.18.245`).

See `cd_lns_sa_cascade/MECHANISM.md` for the form-submission algorithm
description.

Pipeline: E25 (SDF + CD + LNS + SA) → E41 (DPO + CD + LNS + SA + K-joint) →
cascading Hessian saddle escape on the best-of-{E25, E41} plateau. The
adaptive variant tunes CD-polish parameters by canvas area (IBM-class vs
NG45-class) — property-based dispatch is rule-compliant, identity-based
is not.

## Run

```bash
# Full evaluation (17 IBM, partcl-equivalent)
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --all --json

# NG45 commercial designs (Tier 1 top-7 evaluation)
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py --ng45 --json

# Single benchmark (debugging)
uv run evaluate submissions/cd_lns_sa_cascade/placer_adaptive.py -b ibm03
```

## Active layout

| Folder | Role |
|--------|------|
| `cd_lns_sa_cascade/` | **Submission entry** — `placer_adaptive.py` + base `placer.py` |
| `cd_lns_sa/` | E25 component, imported by cascade (do not edit lightly) |
| `examples/` | Reference placers (greedy, random) for API documentation |

The E41 component lives at
`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` because it
depends on `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py` for the
DPO init helper. Don't move it without rewiring the import chain.

## Archived

`_archive/` holds superseded champions and falsified variants. Kept for
historical reference (lineage in `docs/results.md`); no live import path
depends on them.

| Folder | Avg | Note |
|--------|----:|------|
| `_archive/cd_lns_sa_hessian/` | 1.0666 | E74 — Hessian saddle escape (ADR-012). Subsumed by cascade; saddle primitives live in `experiments/E74_hessian_saddle/code/` |
| `_archive/cd_lns_sa_hybrid/` | 1.08151 | E48 — per-bench best-of-{E25, E41}. ADR-011 |
| `_archive/cd_lns_gridbin/` | 1.0990 | E12 — CD plateau + grid-bin LNS. ADR-007 |
| `_archive/cd_adaptive/` | 1.1055 | E9 — plateau-detection CD on full proxy |
| `_archive/cd_only/` | 1.1193 | Fixed-budget CD baseline |
| `_archive/will_seed/` | 1.5338 | Pre-fork SA seed |
| `_archive/cd_lns_sa_cascade_variants/` | (experimental) | 12 dead-end cascade tunings (placer_finegrain*, placer_widesaddle, etc.) from the wall-safe tuning wave |
| `_archive/falsified/cd_lns_sa_hessian_dp/` | DREAMPlace lane | DP path falsified 2026-05-11 — 5D HP sweep showed DP basin is structurally 5–16 % worse than cascade. See `TODO.md` PATH B autopsy |

## Lineage

`E12 (1.0990, ADR-007) → E48 (1.08151, ADR-011) → E74 (1.0666, ADR-012)
→ E84 cascade (1.0612 uncapped on M3, 1.137 wall-safe on cloud)`.

See `docs/decisions/` for the ADRs and `docs/results.md` for per-bench
numbers.
