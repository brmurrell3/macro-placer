# Thinkorplace technical report

Source of technical report 1.2, *Thinkorplace: Smooth Proposals and Exact Local
Search for Chip Macro Placement*. Build from this directory with:

```sh
make
```

Use a TeX distribution providing `latexmk`, pdfLaTeX, and BibTeX. If it is not on
your path, pass its location with `make LATEXMK=/path/to/latexmk`. The PDF is written
to `build/main.pdf`; the build directory is disposable.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Preamble, title block, abstract, Figure 1, acknowledgments, bibliography, and the `\input` list |
| `sections/{intro,problem,approach,results,lessons,limitations,related,conclusion}.tex` | Main text (about 11 pages) |
| `sections/app_*.tex` | Appendices: glossary, objective, algorithms, proofs, curvature, verification, records, reproducibility |
| `generated/*.tex` | Tables rendered from frozen data; do not edit by hand |
| `figures/*.pdf` | Figures rendered from frozen data; the architecture, feasible-set and assignment drawings are TikZ inside the section files |
| `references.bib` | Bibliography |

Theorem statements (`prop:geometry`, `lem:tail`, `prop:acceptance`,
`prop:curvature`) live in the main text; their proofs are in `app_proofs.tex`.
Evidence labels use the macros `\official`, `\historical`, `\measurement` and
`\reproduction`.

## Regenerating figures and tables

Run these from the repository root. None of them runs the solver; each reads the
frozen records under `writeup/data/publication/`.

```sh
.venv/bin/python scripts/render_publication_v1.py          # v1.0 figures, competition, benchmark, baseline and curvature-stage tables
.venv/bin/python scripts/render_publication_devrecords.py  # development records and chronology tables
.venv/bin/python scripts/render_publication_curvature.py   # curvature diagnostic table and follow-up text
.venv/bin/python scripts/render_publication_reproduction.py  # reproduction macros and per-design table
```

The other table generators (`render_publication_measurements.py`,
`render_publication_pilot.py`, `render_publication_integration.py`) and their inputs
are listed in `writeup/data/publication/README.md`.

## Checks

Run `make check` after editing. It builds the paper and runs
`scripts/check_publication_paper.py`, which checks required evidence, citations,
labels, numerical presentation, caption placement, and the final log. The
[editorial standards](EDITORIAL_STANDARDS.md) also require visual inspection of every
final page and set the audience rules for the main text. Passing mechanical checks
alone does not complete review.

## Releases

The report is dated and versioned; see [CHANGELOG.md](CHANGELOG.md). Preserve
released PDFs rather than silently replacing their evidence. After reviewing the
rendered PDF, copy it to `output/pdf/thinkorplace-technical-report.pdf` at the
repository root, then run:

```sh
.venv/bin/python scripts/package_publication.py
```

That command collects the reviewed PDF, website summary, and standalone interactive
comparison into `output/publication/thinkorplace.zip`, with working relative links
and file hashes. It does not deploy a website. Add `--update-tracked-pdf` to also
copy the PDF to `writeup/thinkorplace-report.pdf`, the copy the repository README
links.
