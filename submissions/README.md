# Submissions

Competition-ready placers, organized by lineage.

| Folder | Status | Avg `--all` | Notes |
|--------|--------|------------:|-------|
| `cd_lns_sa_hybrid/` | **CHAMPION** | **1.08151** | E48 — per-bench best-of-{E25, E41} hybrid. NG45 0.6922. Entry: `placer.py`. **Promoted 2026-05-02 (ADR-011)**; supersedes ADR-007/008/009/010. |
| `cd_lns_gridbin/` | prior champion | 1.0990 | E12 — CD plateau + grid-bin LNS overlay. Promoted 2026-04-28 (ADR-007); superseded 2026-05-02. |
| `cd_lns_sa/` | E48 component (lane 1, SDF basin) | 1.0954 standalone | E25 — CD + LNS + SA-v2 polish. Called by `cd_lns_sa_hybrid/placer.py`. Was champion candidate (ADR-008 *Superseded by ADR-011*). |
| `cd_adaptive/` | prior champion | 1.1055 | E9 plateau-detection CD on full proxy. Kept as a baseline reference and as the CD-phase runner imported by both the champion and component placers. |
| `cd_only/` | superseded | 1.1193 | Prior-prior champion (fixed-budget CD). Kept as a baseline reference. |
| `examples/` | demo | — | Greedy / random reference placers. |
| `will_seed/` | baseline | 1.5338 | Pre-fork SA seed. |

The E41 component (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` —
DPO-init pipeline + K-joint LNS) is also called by `cd_lns_sa_hybrid/placer.py`
as lane 2 but stays under `experiments/` because it depends on
`experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py` for the DPO init helper.
Both are imported by the champion at evaluate-time.

Run any placer:

```
uv run evaluate submissions/<name>/placer.py --all --json --hypothesis <name>
```

Promotion path: `experiments/E<NN>_*/` (with manifest) → here once the kill gate is passed and a champion delta is verified. The manifest's `graduated_to` field records the path.

See `experiments/README.md` for the experiment lifecycle, `docs/decisions/` for the structural decisions behind these submissions, and `docs/results.md` for current per-benchmark numbers.
