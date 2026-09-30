# Technical report editorial standards

This report is a standalone, one-column project technical report with an
11-point body size. It does not claim ACM, IEEE, journal, conference, or
archival-proceedings format compliance. A release must identify its version and
date, preserve its evidence boundaries, and remain readable as a self-contained
PDF.

## Executable checks

From the repository root, check source structure and referenced artifacts while
editing:

```sh
.venv/bin/python scripts/check_publication_paper.py --source-only
```

After the final LaTeX build, run the release gate:

```sh
make -C writeup/latex
.venv/bin/python scripts/check_publication_paper.py
```

The release command exits nonzero when any of these invariants fails:

- every literal `\input`, `\includegraphics`, `\reportfigure`/`\maybefigure`, and
  `\IfFileExists` release artifact resolves; a conditional cannot silently omit
  missing generated evidence;
- the TeX and BibTeX sources contain no TODO/FIXME/TBD placeholders, raw tool
  citation markers, or tool-call tokens;
- every cited BibTeX key is defined, active labels are unique, and source-level
  references resolve, apart from the `lastpage` package's `LastPage` label;
- the document uses the explicit 11-point, one-column `article` format;
- active tables put captions above their tabulars, have labels, use at least
  9-point type (`\footnotesize` in the 11-point article is allowed), are not
  mechanically scaled, and report no more than six decimal places;
- the final log records a successful build with no LaTeX, package, or class
  warnings, overfull or underfull boxes, undefined or multiply defined
  references, rerun requests, or fatal errors;
- the final PDF exists, has a PDF header, and the final log and PDF are at least
  as new as the checked TeX, bibliography, generated-table, and figure inputs.

Macro arguments such as `#1` in a command definition are dynamic values, not
literal filenames. The checker deliberately excludes them from file existence
tests. `--tex-root`, `--main`, `--log`, and `--pdf` allow the same checks on a
staged build without weakening the rules.

## Evidence integrity

Tables and claims must retain their evidence class: historical records,
component microbenchmarks, limited-budget primary runs, post-repair validation,
and bounded integration checks remain separate. Failed, timed-out, partial, and
checkpoint-recovery outcomes stay visible. Never substitute a later repair for a
primary observation or calculate a matched mean from incomplete coverage.

Report source identities, input hashes, resource limits, seeds, validity, and
pipeline outcome at the precision supported by the immutable record. A checksum
establishes byte identity within its stated scope; it does not make two study
protocols comparable. Proxy scores do not imply routed timing, power, area, or
design-rule closure.

## Typography and numerical presentation

Body text uses the report's 11-point base size in one column. Tables use
`\footnotesize` (9 point) or larger. Redesign columns, abbreviate repeated
labels, or split a table instead of using `\scriptsize`, `\tiny`, `\resizebox`,
or `\scalebox` on a tabular. Put each table caption above its tabular. Captions
state the evidence scope and define abbreviations close to the data.

Use no more than six decimal places in tables. Match precision to measurement
resolution in prose: proxy costs normally use five or six decimal places and
wall times normally use one decimal place. Higher precision belongs only in an
immutable machine-readable record or a specifically justified numerical
diagnostic. Align comparable numeric columns and use an em dash or a defined
symbol for unavailable values; do not display missing data as zero.

## Mathematical review

Define symbols before use and state the domain, fixed quantities, norm, and
regularity assumptions needed by each result. Provide proofs for the report's
propositions and lemmas, or cite a precise external result. Explain the
consequence for the actual solver after each derivation. Elementary identities
and standard inequalities are analysis tools, not claims of research novelty.

Keep exact real-arithmetic identities separate from floating-point evidence.
State whether error bounds hold only on sampled traces or uniformly on the
states required by a theorem. Identify which objective a derivative, curvature
bound, or acceptance test concerns. Legalization changes the endpoint and
requires its own argument if a pre-legalization bound is to survive.

Check counterexamples arithmetically and verify their stated feasibility.
Review constants, inequality directions, normalization factors, boundary cases,
and theorem-to-implementation correspondence independently. A clean LaTeX build
checks presentation and references; it does not verify a proof.

## Prose and visual review

Use neutral, direct prose. State the observation, its measurement boundary, and
its implication. Avoid promotional or hand-wavy words such as “obviously,”
“clearly,” “groundbreaking,” “revolutionary,” and “dramatically.” The checker
reports these phrases as advisories because a mechanical phrase list cannot
judge meaning and must not act as an AI-authorship detector.

Manual visual review must inspect every page of the final PDF at normal reading
size after the release gate passes. Check title and version, page order,
headings, line and page breaks, equations, figure detail, captions, table type
size and column alignment, footnotes, hyperlinks, bibliography, widows and
orphans, and the absence of clipped or overlapping content. Record or retain the
reviewed PDF; the script cannot attest that this visual review happened.
