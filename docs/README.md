# `docs/`

Development records from the May 2026 competition. They are historical: scores,
champion labels and plans reflect what was believed at the time, on hardware,
budgets and concurrency that differ from run to run. For current results, read
the [technical report](../writeup/thinkorplace-report.pdf) and the
[project README](../README.md).

| Path | Contents |
|---|---|
| [`experiment_index.md`](experiment_index.md) | Catalog of experiments, live and falsified; the manifests are in [`../experiments/`](../experiments/). |
| [`decisions/`](decisions/) | Architecture decision records, immutable once accepted. |
| [`gotchas.md`](gotchas.md) | Codebase pitfalls found during development; the solver source cites it. |
| [`competition/`](competition/) | Competition-era files: the challenge README with local notes, and the agent instructions that set each experiment's kill gate. |

Status reports, handoff notes, plans, the May results and roadmap pages, research
notes and the publication plan were removed in October 2026; the technical report
supersedes them. Manifests, ADRs and the index still cite some of them by path;
read those at commit `b7de373` with `git show b7de373:<path>`.

Some documents link to `docs/reviews/`. That directory held local review notes
and is not published with the repository.
