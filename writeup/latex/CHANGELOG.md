# Report revisions

## 0.4 — 8 September 2026

Problem background and integration of the completed referee follow-up checks;
no new solver runs or runtime changes in this revision.

- Added an introduction to physical design, coupled placement decisions,
  routing-space tradeoffs, hard geometry, and the cost of evaluating proposals.
- Added a schematic showing why greater separation can improve routing room
  while lengthening the connection between two blocks. The caption states the
  routing-layer assumption and distinguishes the drawing from measured data.
- Incorporated the completed canonical rescoring of 34 E74/E84 endpoints.
  Recorded scores agree closely, but 12 placements fail strict canvas bounds.
  The corresponding means are explicitly described as including invalid states.
- Incorporated the completed float64 and guarded-direction curvature follow-up,
  separating canvas-boundary effects from persistent piecewise-component effects.
- Added the final leaderboard rank and current primary documentation for the
  physical-design background; updated the website summary.
- Kept the curvature proposition and proof together and adjusted bibliography
  URL typography after reviewing the rendered pages.

The controlled 51-run comparison remains unexecuted. The v0.3 PDF is preserved
locally under
`output/pdf/releases/thinkorplace-technical-report-v0.3-2026-09-07.pdf`.
Frozen numerical records retain their original precision and source identities.

## 0.3 — 7 September 2026

Mathematical analysis; no new placement experiments or runtime changes.

- Replaced the mechanism discussion with a section containing explicit
  assumptions, three propositions, a lemma, and their proofs.
- Derived the finite polyhedral decomposition of the feasible set and the
  exact union of intervals for a single-coordinate move. Distinguished that
  set from the implementation's conservative current-side interval.
- Stated the incremental cache invariant and its conditional preservation;
  derived the acceptance margin required under bounded endpoint score error.
- Derived variational forms and norm bounds for top-k aggregation. Explained
  why the current scored move still traverses the full grid.
- Proved a signed negative-curvature decrease bound and the symmetric-operator
  eigenvalue residual bound. Distinguished those conditions from the measured
  automatic-differentiation operator and the legalized canonical endpoint.
- Defined assignment interaction residuals and constructed an overlap-free
  HPWL counterexample with a predicted decrease of 18 and actual increase of
  14 before fixed positive normalization.
- Added mathematical review requirements to the editorial standards and the
  new section directory to the build dependencies.

The v0.2 PDF is preserved locally under
`output/pdf/releases/thinkorplace-technical-report-v0.2-2026-09-07.pdf`.
Frozen numerical records retain their original precision and source identities.

## 0.2 — 7 September 2026

Editorial revision; no new placement experiments or runtime changes.

- Replaced the decorative cover and contents pages with a compact article opening.
  Set 11-point body text, at least 9-point tables, consistent caption positions,
  numbered citations, explicit units, and controlled display precision.
- Rewrote the abstract, introduction, and conclusion; removed repeated caveats,
  unsupported comparative language, and prose about how claims should be written.
- Added the missing account of v1's serial two-initializer, cascade, portfolio,
  and periphery search.
- Corrected Lane C's acceptance description against the frozen competition
  source: batches use the incremental replica; phase endpoints are fully
  rescored; a state within 0.5% of CD1 can seed CD2; final selection compares
  full scores.
- Defined the Gaussian density normalization explicitly. The implemented
  component is half the top-decile mean and receives another half-weight in
  the loss, giving the raw mean an overall coefficient of one quarter.
- Separated curvature measurements and final NG45 integration into their own
  subsections. Defined finite differences, units, and the distinction between
  the random probe directions and the Lanczos vector.
- Added the missing ICCAD04 reference DOI and pinned the evaluator repository
  link to the recorded revision.
- Added `make check` and `scripts/check_publication_paper.py`. The release check
  verifies required evidence, citations, references, table presentation, a clean
  LaTeX log, and current build outputs. Visual inspection remains a separate step.

The v0.1 PDF is preserved locally under
`output/pdf/releases/thinkorplace-technical-report-v0.1-2026-09-06.pdf`.
Frozen numerical records retain their original precision and source identities.

## 0.1 — 6 September 2026

Initial technical report, historical reconstruction, component measurements,
engineering-run results, and website companion.
