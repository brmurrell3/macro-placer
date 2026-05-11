---
id: E76
name: dreamplace_integration
status: in_progress
parent: leaderboard top-9 dominated by DREAMPlace-based approaches
created: 2026-05-04
decided: null
champion_at_time: 1.08151 (E48 hybrid)
outcome: **Local prep landed 2026-05-10**: TILOS-protobuf → Bookshelf converter (`code/tilos_to_bookshelf.py`, validated on ibm01: 1140 nodes / 5993 nets / canvas 22.95×23.04 emitted correctly to all 6 Bookshelf files); DREAMPlace `.gp.pl` → our `.pt` reverse converter (`code/bookshelf_to_pt.py`); cloud-side DREAMPlace runner (`cloud/run_dreamplace.sh`) using JSON config + Docker image `limbo018/dreamplace-cuda`; wave converter (`code/convert_all_benches.sh`) for all 17 IBM + 4 NG45; `code/multi_init_ensemble.py` updated with `DREAMPlace_cached` init label so E85 picks up cached `.pt` outputs automatically. Full step-by-step workflow documented in `README.md`. **Awaits cloud GPU provision + DREAMPlace install + run; .pt files expected to drop into `experiments/E76_dreamplace_integration/results/dreamplace_*.pt`.**
champion_delta: null
graduated_to: null
superseded_by: null
---

# E76: dreamplace_integration — use DREAMPlace as a 3rd basin source

## Hypothesis

Six of nine top leaderboard entries use DREAMPlace (Cezar / Hoop Dreams /
Shoom / KLA MACH may be Numba-CD / MTK / UTAustin AS / Mike Gao [DQ]).
DREAMPlace is the open-source GPU-accelerated analytical placer at
github.com/limbo018/DREAMPlace.

E62 (WillSeed-init lane) showed standalone init quality doesn't predict
polished basin quality — but DREAMPlace is **netlist-structure-aware**
(its analytical formulation depends on netlist topology). Polished
DREAMPlace output may land in a basin distinct from SDF (E25) or DPO
(E41), giving a third lane to the E48 hybrid.

## Status: SCOPING (not started)

DREAMPlace requires:
- CUDA (we're on macOS Apple Silicon — no CUDA)
- Specific PyTorch + NLOpt + LEMON dependencies
- C++ build chain

Likely WON'T install on this machine. Three options:

1. **Cloud workaround**: rent a GPU box (Lambda Labs / Modal), build
   DREAMPlace there, run on all 17 IBM, ship outputs back. ~1 day setup.
2. **CPU-only port**: DREAMPlace has a CPU mode in some forks; quality
   degraded vs GPU. ~1-2 day port, uncertain quality.
3. **Approximate via DPO v2 with Adam optimizer extensions**: my DPO
   already does smooth-proxy + Adam. Differences from DREAMPlace:
   electrostatic-style density formulation (Poisson solve), more careful
   step scheduling. ~3-5 day reimplementation if I build my own DREAMPlace-
   style. Uncertain whether it would match DREAMPlace's lift.

## Method (when implemented)

Pipeline per benchmark:
  1. DREAMPlace.run(benchmark) → analytical placement.
  2. project_overlaps → legalize.
  3. Polish via E25 pipeline (CD + LNS + SA-v2).
  4. Polish via E41 pipeline (DPO + CD + LNS + SA-v2 + K-joint).
  5. Best-of {E48 hybrid, DREAMPlace + E25 polish, DREAMPlace + E41 polish}.

If this lands in a new basin similar to E18 (DPO init + E25 polish, -0.51%
--all over E12), expect ~0.3-0.5% additional lift over E48.

## Kill gate (deferred)

- Install fails on this hardware → defer to user setting up cloud.
- DREAMPlace output proxies don't beat SDF on any --fast bench → kill.

## Generalization check

NG45 ariane133 — DREAMPlace authors report good results on similar designs,
should transfer.

## Outcome

[Empty — scoping only.]

## Pointers

- DREAMPlace: https://github.com/limbo018/DREAMPlace
- Lai et al. 2024 "Chip Placement with Diffusion Models" cites DREAMPlace
  as baseline architecture: arXiv:2407.12282.
- Leaderboard top entries using DREAMPlace: see README.md leaderboard
  table fetched 2026-05-04.
