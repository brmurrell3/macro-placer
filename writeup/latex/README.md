# Thinkorplace technical report

Build from this directory with:

```sh
make
```

Use a TeX distribution providing `latexmk`, pdfLaTeX, and BibTeX. If it is not on
your path, pass its location with `make LATEXMK=/path/to/latexmk`.

The PDF is written to `build/main.pdf`. Historical tables under `generated/` and
figures under `figures/` are derived from the frozen publication data. The release
source requires every referenced artifact; missing evidence fails the build.
Primary pilot, repair, and final integration results have separate inputs.

Run `make check` after editing. It builds the paper and checks required evidence,
citations, labels, numerical presentation, caption placement, and the final log.
The [editorial standards](EDITORIAL_STANDARDS.md) also require visual inspection
of every final page. Passing mechanical checks alone does not complete review.

Regenerate the curvature table and precision/branch follow-up from their existing
immutable records without running either diagnostic again:

```sh
python3 scripts/render_publication_curvature.py
```

Run this command from the repository root. The other table-generation commands
are listed in `writeup/data/publication/README.md`.

The report is dated and versioned because later corrected suite runs may change
the result section. Preserve released PDFs rather than silently replacing their
evidence.

The build directory is disposable. After reviewing the rendered PDF, copy it to
`output/pdf/thinkorplace-technical-report.pdf` at the repository root. Then run:

```sh
.venv/bin/python scripts/package_publication.py
```

That command collects the reviewed PDF, website summary, and standalone
interactive comparison into `output/publication/thinkorplace.zip`, with working
relative links and file hashes. It does not deploy a website.
