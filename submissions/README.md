# Submissions

Competition-ready placers, organized by lineage. Last updated 2026-05-05.

## Active

| Folder | Status | Avg `--all` | NG45 | Notes |
|--------|--------|------------:|-----:|-------|
| `cd_lns_sa_hessian/` | **CHAMPION** | **1.0666** | **0.6813** | E74 — Hessian saddle escape on E48 plateau. ariane133 0.6641 (−3.21 % vs E48). Entry: `placer.py`. **Promoted 2026-05-05 (ADR-012)**; supersedes ADR-011. |
| `cd_lns_sa_hybrid/` | fallback / prior champion | 1.08151 | 0.6922 | E48 — per-bench best-of-{E25, E41} hybrid. ADR-011 (superseded 2026-05-05 by ADR-012). Kept as fallback at `placer.py`. |
| `cd_lns_sa/` | component (called by E74 + E48) | 1.0954 standalone | — | E25 — SDF init + CD + LNS + SA-v2 polish. Imported as `CDLNSSAPlacer` by the champion and fallback. |
| `examples/` | demo | — | — | Greedy / random reference placers. |

`cd_lns_sa_hessian/placer.py` is the **competition entry**.
`cd_lns_sa_hessian/loader_placer.py` is a validator-only script that loads
saved best-per-bench placements; not for submission.

## E41 component (lives outside submissions/)

The E41 lane (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` —
DPO best_of_v2 init + CD + LNS + SA-v2 + K-joint K=3) is imported as
`CDLNSSADPOKJointPlacer` by both the champion (E74) and the fallback (E48).
It stays under `experiments/` because it depends on
`experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py` for the DPO init
helper.

## Archived

`_archive/` holds superseded prior champions. Kept for historical
reference (lineage in `docs/results.md`); no live import path depends on
them.

| Folder | Avg | Note |
|--------|----:|------|
| `_archive/cd_lns_gridbin/` | 1.0990 | E12 — CD plateau + grid-bin LNS. ADR-007. |
| `_archive/cd_adaptive/` | 1.1055 | E9 plateau-detection CD on full proxy. |
| `_archive/cd_only/` | 1.1193 | Fixed-budget CD. |
| `_archive/will_seed/` | 1.5338 | Pre-fork SA seed (Will). |

## Run

```
uv run evaluate submissions/cd_lns_sa_hessian/placer.py --all --json
uv run evaluate submissions/cd_lns_sa_hessian/placer.py --ng45
```

Validate aggregate via cached best-per-bench placements:
```
uv run evaluate submissions/cd_lns_sa_hessian/loader_placer.py --all --json
```

## Promotion path

`experiments/E<NN>_*/` (with manifest) → here once the kill gate passes
and a champion delta is verified. The manifest's `graduated_to` field
records the destination.

See `experiments/README.md` for the experiment lifecycle, `docs/decisions/`
for the structural decisions (ADR-007 → ADR-012), and `docs/results.md`
for current per-benchmark numbers.
