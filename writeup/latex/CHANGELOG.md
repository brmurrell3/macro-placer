# Report revisions

## 1.0 — 2 October 2026

Audience-led rewrite for readers outside chip design, with the first reproduction
of the ranked configuration. The authors are Brendan Murrell (lead and
corresponding author), Xiaofei Zhang and Andrew Murrell; the author's note is
removed. Dated 2 October 2026 in the title block, running header, `CITATION.cff`
and the website package metadata.

- New Section 4.2, "Reproduction on a laptop CPU" (Reproduction): the configuration
  inferred to be ranked, rerun at its 720 s budget on all 17 IBM designs, one at a
  time on the lead author's Apple M3 Max CPU under the one-hour cap. Its numbers are
  macros rendered from the run records by
  `scripts/render_publication_reproduction.py`, which writes
  `writeup/data/publication/reproduction_ranked.json`, the summary macros and the
  per-design table (Appendix G). The renderer withholds the suite mean, and the
  text says so, unless every design has a valid, usable, study-eligible result.
  The abstract, the "At a glance" Result row, the introduction's evidence labels,
  Limitations (one run, one host, CPU rather than GPU, wall-time-capped search),
  the glossary and the evidence ledger cite it; it is removed from future work.
- Two runtime defects, each exposed by a stopped reproduction attempt, were fixed
  before the reported run; both attempts are kept as engineering records
  (Appendix F). Attempt 1: the per-phase checkpoint checked canvas bounds strictly,
  so a float32 overshoot discarded IBM03's coordinate-descent result; the
  checkpoint now projects one step inward when no coordinate moves more than
  1e-4 µm and records the shift (commit 64b50a7, runtime cbfbff0c81c3). Attempt 2:
  coordinate descent was budgeted all time to the 720 s deadline, leaving none for
  the final checkpoint on IBM10 and IBM16; each phase now reserves 10 s plus twice
  the slowest measured checkpoint (commit d9f2577, runtime cdd01ecc7c98). The
  reported reproduction is attempt 3.
- New title, result subtitle, code link and version line. The abstract (at most
  190 words) leads with the official result: ninth on the final leaderboard,
  verified mean proxy 0.9974, with the organizers' count of 132 verified eligible
  submissions, the published RePlAce and SA baselines and the winner. An "At a
  glance" box (Problem, Why hard, Approach, Result, Lessons) and an "In brief"
  opener on every main section after the introduction summarize the report for
  skimmers.
- The method section describes the ranked 20 May single pipeline (smooth descent,
  legalization, exact incremental coordinate descent; 720 s budget). The judged
  commit was not recorded, so the ranked program, and by the same inference the
  never-scored status of Lanes B and C (committed on 21–22 May), are stated as
  inferences throughout; the basis is in Appendix G. v1 is the predecessor. The
  smooth descent's device dependence is stated.
- Domain terms are defined at first use. Mathematics follows intuition, equation,
  consequence: log-sum-exp smoothing, Gaussian density, the top-k tail and its
  variational form, a Taylor view of negative curvature, Lanczos with
  Hessian-vector products, and the Hungarian assignment with its counterexample.
  Proposition and lemma content is unchanged from 0.4; Lemma 2.2 is now stated for
  any z in R^N with weights named beta, Proposition 3.2 drops a redundant
  smoothness hypothesis, and nets are indexed by n. All proofs are in an appendix
  and match the main-text statements.
- Corrections from review: the score is discontinuous as well as kinked (pins
  snap to grid cells); coordinate descent scores at most ten candidates per axis
  in one batched incremental call; the tail example compares the density
  component D = A_k/2 (0.0025) with its bound ||delta||_1/(2k) = 0.0153; the
  acceptance margin covers the internal float64 state, the batched path was not
  sampled, and saved endpoints rescore only to about 1e-7; the speedup range is
  10^3 to 2x10^4; ranks four to nine lie within 1.4%.
- The acceptance margin uses the sampled absolute proxy error (4.44e-16 over 36
  endpoints, so the 1e-9 threshold is about 1.1e6 times twice the error).
- Curvature kicks: the heading, abstract and summary no longer credit or
  discredit the curvature itself; step sizes are total displacement norms; the
  missing random-direction and zero-step (polish-only) controls are named in the
  lessons, limitations and conclusion.
- The post-competition audit is counted as ten defects. Most sit in parts that
  were never scored or in hierarchical-design parsing; the two that could touch
  the single pipeline (float32 export at the upper canvas edge, input soft
  clusters outside the canvas) are stated as not excludable. A new frozen record,
  `ibm_repair_equivalence.json`, shows the repaired parsing leaves all 17 IBM
  inputs unchanged.
- Evidence labels: Official, Historical, Measurement and Reproduction. The
  official score is no longer compared with a historical record, and the
  Reproduction result is shown beside it, not as a matched comparison. Caveats are
  collected in one Limitations section. Experiment codenames, hashes and repository
  paths are confined to
  the appendices and their glossary, which maps the main text's record names to
  them.
- New main-text figures: result in context, an IBM01 placement over the
  evaluator's congestion map, per-design component deltas and curvature attempts
  (rendered from frozen data), plus TikZ drawings of the architecture, one pair's
  non-overlap branches and the assignment counterexample. The
  wirelength-versus-channel schematic is kept as a panel.
- New tables: competition context; representative development records with
  verdicts. Appendix tables add a glossary and a single chronology table,
  benchmark statistics, records against the published baselines and the
  curvature stages.
- Moved to appendices: the exact proxy definition, algorithm settings and seeds,
  proofs, the curvature diagnostic, the post-competition audit with an
  impact-on-judged-entry column and the condensed engineering runs, historical
  per-design records and the evidence ledger, and the reproducibility index.
- Corrected record attributions: the 1.05750 record is a 17 May run of a later v1
  variant, not the 13 May entry; the 1.0771 listing is an earlier, unverified
  submission; curvature stage times reach 4,007 s (first stage) and 3,454 s
  (cascade).
- New sections: what worked and what did not (141 recorded experiments, 39
  falsified), related work grouped by topic with an expanded bibliography, and
  acknowledgments.
- Layout: main text of about 14.3 pages (the limit was raised from 14 to 15 pages for the reproduction section); floats kept inside their appendix; the chronology
  and glossary are page-breaking tables; bibliography in small type with linked
  DOIs; identifiers break only at punctuation; the title page has no running
  header.
- `sections/problem_background.tex` is merged into Sections 1–2 and removed;
  `sections/mathematical_analysis.tex` is no longer part of the build.

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
