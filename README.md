# Thinkorplace

Smooth proposals and exact local search for chip macro placement.

Thinkorplace placed ninth on the final leaderboard of the Partcl/HRT Macro
Placement Challenge 2026 (verified mean proxy 0.9974; the organizers report 132
verified eligible submissions).

## Result

Official result: mean proxy cost over the 17 public IBM designs (lower is better).

| Entry | Mean proxy |
|---|---:|
| Archgen (first) | 0.9507 |
| Carrotato (second) | 0.9522 |
| **thinkorplace (ninth)** | **0.9974** |
| RePlAce (published baseline) | 1.4578 |
| Simulated annealing (published baseline) | 2.1251 |

The thinkorplace entry had zero hard-macro overlaps, a best design at 0.7829, a
worst at 1.2602, and a listed runtime of 12 minutes per design. Its mean is 31.6%
below the RePlAce mean and 4.9% above the winner's. Source: the organizers' final
leaderboard at upstream revision
[`996193c`](https://github.com/partcleda/macro-place-challenge-2026/blob/996193c83eaad151e5ae3fb166cedbd83372d060/README.md).
The baselines are the published means from *An Updated Assessment of
Reinforcement Learning for Macro Placement*, as cited by the challenge. Runtimes
are self-listed, and hardware differs between entries.

## Read more

- [Read the technical report](writeup/thinkorplace-report.pdf) (technical report 1.0, 30 September 2026)
- [Website summary](writeup/website-summary.md): the article text for the
  author's website; its relative links resolve in the packaged site built by
  `scripts/package_publication.py`, not in this repository.
- [Placement explorer](writeup/interactive/placement-comparison.html): a
  standalone HTML page; download it and open it in a browser.

## How it works

A chip floorplan holds a few hundred macros, large fixed-size blocks such as
memories, which must sit inside the canvas without overlapping. The challenge
scores a placement with a proxy cost: wirelength, plus half the placement
density, plus half the routing congestion, as computed by the TILOS
MacroPlacement evaluator. Thinkorplace first moves all macros at once by
gradient descent on a smooth stand-in for that score, and a legalizer then
removes the remaining overlaps. Exact coordinate descent improves the legal
placement one macro at a time and accepts a move only when the true proxy score
falls; an incremental evaluator makes each scored move take milliseconds instead
of seconds. The judged commit was not recorded; from the listed runtime and the
20 May resubmission date we infer that the ranked entry was most likely this
single pipeline with a 720-second budget, about 12 minutes per design. Two
plateau-escape ideas, a step along a negative-curvature direction of the smooth
surrogate and a Hungarian reassignment of macros to slots, were committed before
the deadline as extra lanes and, by the same inference, were never scored. In the
historical development records, about 85% of the gain over the earlier serial
pipeline came from lower routing congestion.

## Repository map

| Path | Contents |
|---|---|
| [`thinkorplace/`](thinkorplace/) | The solver package: the corrected post-competition runtime of v1 and v2. |
| [`submissions/`](submissions/) | Competition entry points (v2: `submissions/thinkorplace-v2/placer.py`), example placers, and archived earlier placers. |
| [`macro_place/`](macro_place/) | Challenge infrastructure: benchmark loader, proxy cost, the `evaluate` command, and the incremental evaluator. |
| [`writeup/`](writeup/) | Report source, frozen evidence, study configurations and run records, the explorer, and the website summary. |
| [`experiments/`](experiments/) | About 150 development experiments, each with a manifest of hypothesis, method, kill gate and outcome, including falsified ones. |
| [`docs/`](docs/) | May 2026 development records: experiment index, decision records, and the competition-era agent instructions in [`docs/competition/`](docs/competition/). The original challenge README is [upstream at revision `996193c`](https://github.com/partcleda/macro-place-challenge-2026/blob/996193c83eaad151e5ae3fb166cedbd83372d060/README.md). |
| [`scripts/`](scripts/) | Study runner, publication renderers and checks, and development utilities. |
| [`test/`](test/) | Focused tests for the package, the study runner, and the post-competition repairs. |

The root [`placer.py`](placer.py) is the competition launcher; it dispatches to
the v2 entry.

## Reproduce

Set up the environment as described in [`SETUP.md`](SETUP.md):

```bash
git clone https://github.com/brmurrell3/macro-placer.git
cd macro-placer
git submodule update --init external/MacroPlacement
uv sync
```

Evaluate the v2 entry on one design:

```bash
uv run evaluate submissions/thinkorplace-v2/placer.py -b ibm01
```

This uses the entry's current defaults (three lanes, 2,700-second budget), not
the 720-second configuration inferred to have been ranked.

Rerun the configuration inferred to be the ranked one (single pipeline,
720-second budget per design, all 17 IBM designs, one design at a time) with the
study runner:

```bash
.venv/bin/python scripts/run_publication_study.py \
  --config writeup/study/ranked_reproduction.json \
  --output writeup/runs/ranked-reproduction-$(date +%Y%m%d)
```

Scores depend on host speed, because coordinate descent stops at a wall-clock
limit. A first run of this configuration (30 September 2026) was stopped after
it exposed a checkpoint defect in the corrected runtime, which can discard the
coordinate-descent result when a coordinate rounds past the canvas edge; a fix is
pending, and the report's Appendix F describes the attempt. The other study
configurations and the runner's options are documented in
[`writeup/study/README.md`](writeup/study/README.md).

Build the report and run its release check (requires a TeX distribution with
`latexmk`, pdfLaTeX and BibTeX; the PDF is written to
`writeup/latex/build/main.pdf`):

```bash
make -C writeup/latex
.venv/bin/python scripts/check_publication_paper.py
```

## About the challenge

The Partcl/HRT Macro Placement Challenge 2026 asked entrants to place the macros
of 17 public IBM benchmark designs, within one hour per design and with zero
hard-macro overlaps, scored by a published proxy cost; see the
[upstream repository](https://github.com/partcleda/macro-place-challenge-2026).

## Authors

Brendan Murrell (lead and corresponding author,
[brendanmurrell.com](https://brendanmurrell.com)), Xiaofei Zhang, and Andrew
Murrell. The competition entry was submitted as team "thinkorplace".

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this
repository" button reads it.

## License

Apache-2.0, inherited from the upstream challenge repository; see
[`LICENSE.md`](LICENSE.md).
