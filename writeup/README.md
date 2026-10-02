# `writeup/`

Source, evidence and companion material for *Thinkorplace: Smooth Proposals and
Exact Local Search for Chip Macro Placement* (technical report 1.0,
2 October 2026).

| Path | Contents |
|---|---|
| [`thinkorplace-report.pdf`](thinkorplace-report.pdf) | The released report. |
| [`latex/`](latex/) | Report source. From the repository root, build with `make -C writeup/latex` (the PDF is written to `latex/build/main.pdf`) and run the release check with `.venv/bin/python scripts/check_publication_paper.py`. See [`latex/README.md`](latex/README.md) and [`latex/EDITORIAL_STANDARDS.md`](latex/EDITORIAL_STANDARDS.md). |
| [`data/publication/`](data/publication/) | Frozen evidence: historical records, measurements, diagnostics, source snapshots and their hashes. See its [README](data/publication/README.md). |
| [`study/`](study/) | Study-runner configurations. [`study/README.md`](study/README.md) documents the runner. |
| [`runs/`](runs/) | Run records written by the study runner: manifests, per-run requests, results and logs. See [`runs/README.md`](runs/README.md). |
| [`interactive/`](interactive/) | The standalone placement explorer, [`placement-comparison.html`](interactive/placement-comparison.html), and its data. Download the page and open it in a browser. |
| [`website-summary.md`](website-summary.md) | Short summary of the report for the author's website. Its relative links resolve in the packaged site built by `scripts/package_publication.py`, not in this directory. |
| [`archive/`](archive/) | The May 2026 development data log, [`archive/2026-05-drafts/evidence.md`](archive/2026-05-drafts/evidence.md), which manifests and ADRs cite, and archived placer code that the tests and frozen evidence reference by path. |
