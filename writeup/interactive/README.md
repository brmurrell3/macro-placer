# Curvature-cascade case studies (historical explorer)

Open `placement-comparison.html` in a browser. It shows the 6,869 hard macros of
the 17 public IBM designs at their saved positions before and after a
curvature-cascade refinement. These are the May 2026 curvature-cascade
experiments of the v1 predecessor (internal stages E74 → E84), not the ranked
20 May pipeline.

- **Displacement** (default): macros at their after-cascade positions, colored
  by distance moved on a banded log scale, with arrows from the earlier
  positions. Moves shorter than 6 px on screen are shown by color only.
- **Before/after divider**: the two saved placements side by side, split by a
  draggable divider.

Arrows and segments show displacement, not a solver path or an optimization
replay. Soft clusters are not drawn. Each design also shows, for both stages,
the hard-overlap count and the strict-bounds status from the September 2026
rescore (`../data/publication/e74_e84_canonical_rescore_20260907.json`,
cross-checked against `e74_e84_geometry_crosscheck_20260907.json`): 22 of the
34 placements pass; the other 12 exceed a canvas edge by at most 7.63e-6 µm,
one float32 rounding step.

Proxy costs come from the historical run records. Display coordinates are
rounded to five decimals, so do not use them as solver output or judge legality
from touching pixels; the full-precision tensors remain the evidence, and their
checksums are in `placement-data.json`.

Regenerate the fragment, the data and the standalone page from the repository
root:

```sh
.venv/bin/python scripts/build_publication_explorer.py
```

`placement-comparison.template.html` holds the markup, styles and script;
`placement-comparison.fragment.html` is the template with data, for embedding in
another page; `placement-comparison.html` wraps the fragment in a minimal page.
The page has no external dependencies and makes no network requests. Its report
link expects `thinkorplace-technical-report.pdf` beside it, as in the package
built by `scripts/package_publication.py`.

Browser QA: from the repository root, run

```sh
.venv/bin/python scripts/check_publication_explorer.py
```

It needs only the Python standard library and an installed Chrome or Chromium
(`--chrome` or `$CHROME` selects one), which it drives headless over the
DevTools protocol. It covers 360 and 736 px, light and dark, all 17 designs,
keyboard view switching and divider control, pointer selection, reduced motion,
console errors and horizontal scroll, and exits non-zero on any failure. Results
are in `qa/checks.json` and screenshots in `qa/`. Last run (30 September 2026,
headless Chrome 154): 68 design/viewport/theme checks, no failures.
