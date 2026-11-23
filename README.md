# Thinkorplace

Smooth proposals and exact local search for chip macro placement.

Thinkorplace placed ninth on the final leaderboard of the Partcl/HRT Macro
Placement Challenge 2026 (verified mean proxy 0.9974; the organizers report 132
verified eligible submissions). Its mean is 31.6% below the published RePlAce
baseline, with a listed runtime of 12 of the allowed 60 minutes per design.

![Mean proxy cost of all 129 ranked entries on the final leaderboard, plotted by final rank, where lower is better. Thinkorplace is highlighted at rank 9 with 0.9974. Dashed lines mark the RePlAce baseline at 1.4578 and the simulated-annealing baseline at 2.1251. Ranks 128 and 129 are off scale at 3.60183 and 7.215.](writeup/interactive/leaderboard-all.svg)

The [project page](https://brendanmurrell.com/projects/macro-placer/) is a short
overview. The [technical report](writeup/thinkorplace-report.pdf) covers the
method, proofs, per-design records and a post-competition reproduction.

## How it works

Gradient descent on a smooth approximation of the proxy score moves every macro
at once, and a legalizer removes the remaining overlaps. Exact coordinate descent
then moves one macro at a time and keeps a move only when the true score falls.
An incremental evaluator rescores only what a move changes, so each exact check
takes milliseconds instead of seconds.

![Animation of Thinkorplace placing ibm01. On the left, the large blocks move on the chip canvas, blue during gradient descent on the smooth score and orange during coordinate descent on the exact score, with blocks moved since the previous frame shown darker. On the right, the exact proxy cost falls from 1.1953 to 0.8824 during gradient descent, rises to 0.9126 when legalization removes the overlaps, and falls to 0.8300 by the end of the 315-second run.](writeup/interactive/thinkorplace-ibm01.gif)

The animation is a post-competition laptop run on ibm01 of the configuration
inferred to have been ranked. [`scripts/render_placement_animation.py`](scripts/render_placement_animation.py)
draws it from the run record in
[`writeup/runs/animation-ibm01-20261117/`](writeup/runs/animation-ibm01-20261117/).

## Versions

Each version has an entry point in [`submissions/`](submissions/) and its solver
code in the [`thinkorplace/`](thinkorplace/) package.

| Version | Entry | Code | Method |
|---|---|---|---|
| v1 | [`submissions/thinkorplace-v1/`](submissions/thinkorplace-v1/) | [`thinkorplace/v1/`](thinkorplace/v1/) | Two starting lanes, three local searches and a curvature cascade. About 53 minutes per design. |
| v2 | [`submissions/thinkorplace-v2/`](submissions/thinkorplace-v2/) | [`thinkorplace/v2/`](thinkorplace/v2/) | The ranked entry. Smooth descent, legalisation, then exact coordinate descent, in 12 minutes per design. |
| v3 | [`submissions/thinkorplace-v3/`](submissions/thinkorplace-v3/) | [`thinkorplace/v3/`](thinkorplace/v3/) | After the competition. Draws many seeds by recursive bisection, screens them with a cheap span score, and runs three chains for 12 minutes on three cores, one warm from the supplied placement. |

Results, by evidence class. Rows of different classes come from different
runs and machines, so compare within a class.

| Class | Result |
|---|---|
| Development (May 2026) | v2 scored 7.1% lower than v1 on average over the 17 IBM designs, lower on 16 of them, in 524 s instead of 3,192 s per design. |
| Official | v2 ranked ninth with a verified mean proxy of 0.9974, 31.6% below the RePlAce baseline. |
| Reproduction | v2 rerun on a laptop CPU, each design alone: mean 0.98121, all 17 placements valid. |
| Measurement | At 720 s and three cores per design on one M3 Max, v3's research implementation scored 0.9443 against 0.9879 for v2 run beside it, 4.4% lower and lower on 16 of 17 designs. The package in `thinkorplace/v3/` run alone scored 0.9376. |

All scores are the challenge's proxy cost. They say nothing about routed
timing, power or area. The [v3 README](submissions/thinkorplace-v3/README.md)
lists the caveats of the v3 comparison.

## Engineering

- **Incremental evaluator** ([`thinkorplace/common/incremental_evaluator.py`](thinkorplace/common/incremental_evaluator.py)).
  The reference scorer takes 1.4 to 27 s per call, too slow for exact search.
  The evaluator updates only the nets and grid cells a move touches: 1.3 to
  3.7 ms per move on sampled moves, with a worst difference from the reference
  of 1.1e-15 over 130 random moves. That turns a few hundred exact checks per
  design into about 100,000 ([decision record](docs/decisions/002_incremental_evaluator.md)).
- **Deadline-safe process supervision** ([`thinkorplace/common/runtime_safety.py`](thinkorplace/common/runtime_safety.py),
  [`thinkorplace/v2/worker.py`](thinkorplace/v2/worker.py), [`thinkorplace/v3/scheduler.py`](thinkorplace/v3/scheduler.py)).
  Lanes and chains run in spawned processes against one monotonic deadline,
  publish only validated placements through atomic writes, and are terminated
  at the deadline, so a run returns its best legal placement in time.
- **Bit-identical rewrite.** v3 was rewritten from its research implementation
  and checked stage by stage: for fixed inputs every stage returns identical
  arrays.
- **GPU kernels** ([`kernels/`](kernels/)). v3's gradient-descent stage as
  custom kernels with hand-derived gradients, in Metal and in CUDA, with two
  kernels also in Triton, each gated against a float64 reference. A 500-step
  descent of a batch of placements runs 18 to 41 times faster than stock
  PyTorch on an Apple M3 Max and 19 to 47 times faster on an NVIDIA L4. The
  kernels are not part of the ranked entry.
- **Tests** in [`test/`](test/) cover the evaluator against the reference, the
  safety layer, package isolation and v3's seeds, legalisation, search and
  deadline.

## Repository map

| Path | Contents |
|---|---|
| [`thinkorplace/`](thinkorplace/) | Solver package: `common/`, `v1/`, `v2/`, `v3/` |
| [`submissions/`](submissions/) | One entry point per version. The root [`placer.py`](placer.py) loads v2, the ranked entry |
| [`kernels/`](kernels/) | Metal, CUDA and Triton kernels for v3's gradient-descent stage |
| [`macro_place/`](macro_place/) | Benchmark loader, proxy cost, the `evaluate` command and the incremental evaluator |
| [`writeup/`](writeup/) | Report source, frozen evidence, study configurations and run records |
| [`experiments/`](experiments/) | About 140 development experiments with their outcomes, including falsified ones |
| [`docs/`](docs/), [`analysis/`](analysis/) | May 2026 development records and diagnostics |
| [`scripts/`](scripts/), [`test/`](test/) | Study runner, publication renderers and checks, and tests |

Most other top-level files and folders come from the upstream
[challenge repository](https://github.com/partcleda/macro-place-challenge-2026).

## Reproduce

Set up the environment. [`SETUP.md`](SETUP.md) has the details.

```bash
git clone https://github.com/brmurrell3/macro-placer.git
cd macro-placer
git submodule update --init external/MacroPlacement
uv sync
```

Evaluate v3 on one design. It needs the `kahypar` package, which `uv sync`
does not install.

```bash
uv pip install kahypar==1.3.6
uv run evaluate submissions/thinkorplace-v3/placer.py -b ibm01
```

Evaluate the v2 entry. Its current defaults run three lanes with a 2,700-second
budget, not the 720-second single pipeline inferred to have been ranked.

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py -b ibm01
```

Rerun the configuration inferred to have been ranked on all 17 designs. Scores
depend on host speed, because coordinate descent stops at a wall-clock limit.
[`writeup/study/README.md`](writeup/study/README.md) documents the other study
configurations.

```bash
.venv/bin/python scripts/run_publication_study.py \
  --config writeup/study/ranked_reproduction.json \
  --output writeup/runs/ranked-reproduction-$(date +%Y%m%d)
```

Build the report and run its release check. This needs a TeX distribution with
`latexmk`.

```bash
make -C writeup/latex
.venv/bin/python scripts/check_publication_paper.py
```

## License

Apache-2.0, inherited from the upstream challenge repository. See
[`LICENSE.md`](LICENSE.md).
