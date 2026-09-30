# Historical placement explorer

Open `placement-comparison.html` in a browser. Select an IBM benchmark, move the
comparison divider, and inspect a macro by pointer or keyboard. It displays 6,869
hard macros across the 17 saved E74→E84 pairs. The two endpoints are real saved
placements; the connecting line on a selected macro indicates displacement, not a
legal path or an optimization replay. Soft clusters are omitted from the drawing.

Scores and overlap counts come from historical JSON records, and source tensor
checksums are included in `placement-data.json`. Display coordinates are rounded
to five decimals; do not use them as replacement solver output or infer exact
legality from touching pixels. The full precision tensors remain the evidence.

Regenerate the fragment and data from the repository root:

```sh
.venv/bin/python scripts/build_publication_explorer.py
```

The standalone document wraps that fragment with theme and control styles. It
uses pinned D3 7.9.0 from jsDelivr and makes no data requests. To embed it on a
website, serve `placement-comparison.html` as an iframe, or adapt the fragment to
the site's CSS and D3 dependency. No website has been deployed.

Browser verification is in `scripts/check_publication_explorer.cjs`; it needs
Playwright and a Chromium browser (`PLAYWRIGHT_CHANNEL=chrome` selects an installed
Chrome). QA covers responsive light/dark layouts, all benchmark options, actual
macro marks, keyboard selection and divider updates. A static paper comparison
is in `../latex/figures/cascade_placement_ibm01.pdf`.
