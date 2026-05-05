# Submissions archive

Superseded prior champions. Lineage is in `docs/results.md` and
`CLAUDE.md`. The active champion is `submissions/cd_lns_sa_hessian/`
(E74, ADR-012, 2026-05-05).

| Folder | Era | Avg `--all` | ADR | Superseded by |
|--------|-----|------------:|-----|---------------|
| `cd_only/` | CD-only era (2026-04-27) | 1.1193 | — | E9 |
| `cd_adaptive/` | E9 plateau detection (2026-04-27) | 1.1055 | — | E12 |
| `cd_lns_gridbin/` | E12 CD + grid-bin LNS (2026-04-28 → 2026-05-02) | 1.0990 | ADR-007 | E48 |
| `will_seed/` | Pre-fork SA seed (Will, 2026-04-23) | 1.5338 | — | (baseline only) |

These are kept for:
1. **Reproducibility of historical results** (numbers cited in
   `docs/results.md`, `writeup/historical_results.md`, ADRs 007-011).
2. **Lineage continuity** — manifests under `experiments/` reference these
   paths via their `graduated_to` field; moving them out of the tree would
   break those references retrospectively.

Old experiment manifests reference these paths assuming `submissions/<name>/`.
Code that needs to import from them should use `submissions/_archive/<name>/`.
No active code path imports from this archive — only the active
submissions in `submissions/cd_lns_sa*/` are live.
