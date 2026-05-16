# Architecture Decision Records

This directory captures the load-bearing structural decisions made during the
Macro Placement Challenge 2026 work. Each ADR explains *why* a decision was
made — the context, what was tried, what was observed, and what alternatives
were ruled out — so a future reader can understand the rationale without
reconstructing it from chat history or commit logs.

## Format

Each ADR is a short markdown file with five sections:

1. **Context** — the problem state at the time. What we tried; what we
   observed. Numbers cited from `writeup/evidence.md` or
   `writeup/contributions.md`.
2. **Decision** — what was decided, in active voice.
3. **Consequences** — what this unlocked, what it cost, and which
   alternatives were ruled out.
4. **Evidence** — pointers to the supporting sections in
   `writeup/evidence.md` and `writeup/contributions.md`.

ADRs use the template at the top of `001_optimize_full_proxy.md`.

## Immutability

ADRs are immutable once accepted. If a decision is later reversed or
refined, do **not** edit the original — write a new ADR with the next
sequence number and mark its status as `Accepted`, with a `Supersedes:
ADR-NNN` line. The superseded ADR keeps `Status: Accepted` but a
`Superseded by: ADR-MMM` line is added at the top so the chain stays
traceable. The point is to preserve why we believed what we believed at
the time, not to keep the docs perpetually correct.

## Index

| ADR | Title | One-line summary |
|-----|-------|------------------|
| [001](001_optimize_full_proxy.md) | Optimize the full composite proxy directly, not LP-HPWL | LP-HPWL has rho = -0.001 with the real proxy; congestion is 74.9 % of cost. Rank moves on f(p) = WL + 0.5 D + 0.5 C, not on a single component. |
| [002](002_incremental_evaluator.md) | Build an incremental evaluator with bit-for-bit parity | 4657x speedup over `compute_proxy_cost` on ibm10 with parity to 1.1e-15. Gates the CD generation. |
| [003](003_plateau_detection.md) | Use per-benchmark plateau detection, not a fixed budget | Defaults `(min_time_s=300, hard_cap_s=3600, patience=3, plateau_threshold=0.005)`. 1.1193 to 1.1055 on `--all`. |
| [004](004_threshold_0_005.md) | Ship plateau_threshold=0.005, not the tighter 0.001 from E16 | E16 at 0.001 scored 1.1025 (-0.27 %) at +46 % wall — formally marginal under the >= 0.005 graduate gate. Conservative defaults retained. |
| [005](005_sdf_init_canonical.md) | Retain SDF init across every algorithmic generation | Random init gives 4.94 avg proxy. Every alternative init (spectral, hMETIS, greedy, boundary) is 1.75-1.91. SDF analytical spreading is the canonical init. |
| [006](006_bypass_dont_fix.md) | Bypass approximations rather than improving them | When an approximation is structurally wrong (LP-HPWL rho = -0.001; RUDY top-5 % overlap 10.9 %), prefer exact evaluation backed by faster data structures over a more accurate approximation. |
| [007](007_cd_lns_gridbin_promotion.md) | Promote CD + grid-bin LNS as champion (E12 -> 1.0990) | Three escape mechanisms (E3 LNS, SDF jitter, subset-CD destroy) failed because they reused CD's move type. Grid-bin LNS uses a different move type (all (col, row) cell centers) and lands at 1.0990, beating leaderboard 1.1172 by -1.63 %. |
| [008](008_cd_lns_sa_promotion.md) | Promote CD + LNS + SA-v2 (E25) | 1.0954 (-0.33 % vs E12). *Proposed*; not accepted (superseded by chain E18 → E41 → E48). |
| [009](009_dpo_init_promotion.md) | Promote DPO-init + E25 polish (E18) | 1.08979 (-0.84 % vs E12); 4/4 NG45 wins (DPO basin transfers to commercial). *Proposed*; superseded by E41. |
| [010](010_dpo_kjoint_promotion.md) | Promote DPO + CD + LNS + SA-v2 + K-joint K=3 (E41) | 1.0848 (-1.29 % vs E12), --ng45 0.69022. 14/17 IBM wins. *Proposed*; superseded by E48 hybrid. |
| [011](011_hybrid_e25_e41_promotion.md) | Promote per-bench best-of-{E25, E41} hybrid (E48) | 1.08151 (-1.66 % vs E12); --ng45 0.6922. *Accepted* 2026-05-02; superseded by ADR-012 (E74) 2026-05-05. |
| [012](012_e74_hessian_saddle_promotion.md) | Promote E48 hybrid + Hessian saddle escape (E74) | 1.0666 (-1.38 % vs E48, -4.53 % vs leaderboard 1.1172); --ng45 0.6813, ariane133 0.6641 = -3.21 % (breaks the failure point that killed E42/E43/E44/E54/E62). *Accepted* 2026-05-05. Supersedes ADR-011. |
| [012a](012a_e61v2_spatial_block_crossover_proposed_superseded.md) | E61_v2 spatial-block GA crossover (alternative ADR-012 path) | 1.08083 (--all marginal -0.07 %); --ng45 0.6908 first NG45-positive mechanism since E18. *Proposed → superseded before acceptance* when ADR-012 (E74) was accepted two days later. Renumbered 012 → 012a on 2026-05-16. |
