# Submissions

Competition-ready placers, organized by lineage.

| Folder | Status | Avg `--all` | Notes |
|--------|--------|------------:|-------|
| `cd_lns_gridbin/` | **CHAMPION** | 1.0990 | E12 — CD plateau + grid-bin LNS overlay. Entry: `placer.py`. Promoted 2026-04-28 (ADR-007). |
| `cd_adaptive/` | prior champion | 1.1055 | E9 plateau-detection CD on full proxy. Superseded 2026-04-28 by `cd_lns_gridbin/`. Kept as a baseline reference and as the CD-phase runner imported by the champion. |
| `cd_only/` | superseded | 1.1193 | Prior-prior champion (fixed-budget CD). Kept as a baseline reference. |
| `examples/` | demo | — | Greedy / random reference placers. |
| `will_seed/` | baseline | 1.5338 | Pre-fork SA seed. |

Run any placer:

```
uv run evaluate submissions/<name>/placer.py --all --json --hypothesis <name>
```

Promotion path: `experiments/E<NN>_*/` (with manifest) → here once the kill gate is passed and a champion delta is verified. The manifest's `graduated_to` field records the path.

See `experiments/README.md` for the experiment lifecycle, `docs/decisions/` for the structural decisions behind these submissions, and `docs/results.md` for current per-benchmark numbers.
