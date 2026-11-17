# Thinkorplace

Smooth proposals and exact local search for chip macro placement.

Thinkorplace placed ninth on the final leaderboard of the Partcl/HRT Macro
Placement Challenge 2026 (verified mean proxy 0.9974; the organizers report 132
verified eligible submissions). Its mean is 31.6% below the published RePlAce
baseline, with a listed runtime of 12 of the allowed 60 minutes per design.

![Mean proxy cost of all 129 ranked entries on the final leaderboard, plotted by final rank, where lower is better. Thinkorplace is highlighted at rank 9 with 0.9974. Dashed lines mark the RePlAce baseline at 1.4578 and the simulated-annealing baseline at 2.1251. Filled dots are entries verified by the organizers and hollow dots are not verified. Ranks 128 and 129 are off scale at 3.60183 and 7.215.](writeup/interactive/leaderboard-all.svg)

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

## Repository map

| Path | Contents |
|---|---|
| [`thinkorplace/`](thinkorplace/) | Solver package, the corrected post-competition runtime |
| [`submissions/`](submissions/) | Competition entry points. The root [`placer.py`](placer.py) launches the v2 entry in `submissions/thinkorplace-v2/` |
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

Evaluate the v2 entry on one design. Its current defaults run three lanes with a
2,700-second budget, not the 720-second single pipeline inferred to have been
ranked.

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
