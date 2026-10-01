#!/usr/bin/env python3
"""Render the development-records table and the codename chronology.

Input: writeup/data/publication/development_records.json. Its curated fields
(row selection, plain-language text, verdicts, glossary entries) are written by
hand. Its computed fields (manifest blob ids, quoted lines, status counts and
frozen-data values) are filled by ``--freeze`` from git objects at the recorded
commit and from the frozen files under writeup/data/publication/.

Default run: verify every computed field against those sources, then write
writeup/latex/generated/development_records.tex and chronology.tex (a longtable;
the paper must load the longtable package).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'writeup/data/publication'
RECORDS = DATA / 'development_records.json'
GENERATED = ROOT / 'writeup/latex/generated'
MANIFEST_RE = re.compile(r'^experiments/(?:_archive/)?[^/]+/manifest\.md$')
TEMPLATE = 'experiments/_template/manifest.md'
NAMED_BUCKETS = ('graduated', 'falsified', 'marginal', 'in_progress')


def git(*args: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def manifest_status(text: str) -> str | None:
    """First ``status:`` value inside the leading YAML block, without comments."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != '---':
        return None
    for line in lines[1:]:
        if line.strip() == '---':
            return None
        if line.startswith('status:'):
            return line[len('status:'):].split('#', 1)[0].strip()
    return None


def survey(commit: str) -> dict:
    paths = [p for p in git('ls-tree', '-r', '--name-only', commit, '--', 'experiments').splitlines()
             if MANIFEST_RE.match(p) and p != TEMPLATE]
    statuses = Counter(manifest_status(git('show', f'{commit}:{p}')) for p in paths)
    other = {k: v for k, v in sorted(statuses.items()) if k not in NAMED_BUCKETS}
    counts = {k: statuses.get(k, 0) for k in NAMED_BUCKETS}
    counts.update(other=sum(other.values()), total=len(paths), other_breakdown=other)
    directories = [d for base in ('experiments/', 'experiments/_archive/')
                   for d in git('ls-tree', '-d', '--name-only', commit, base).splitlines()
                   if d not in ('experiments/_archive', 'experiments/_template')]
    with_manifest = {p.rsplit('/', 1)[0] for p in paths}
    return {'status_counts': counts,
            'directory_counts': {'experiment_directories': len(directories),
                                 'with_manifest': len(with_manifest),
                                 'without_manifest': sorted(set(directories) - with_manifest)}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proxies(path: str) -> dict:
    return {b['name']: b['proxy_cost'] for b in json.loads((DATA / path).read_text())['benchmarks']}


def frozen_values(check: dict) -> dict:
    """Recompute the values a frozen-data check asserts."""
    kind, files = check['kind'], check['files']
    if kind == 'record_mean':
        data = json.loads((DATA / files[0]).read_text())['benchmarks']
        return {'designs': len(data), 'mean_proxy': statistics.mean(b['proxy_cost'] for b in data),
                'total_overlaps': sum(b['overlap_count'] for b in data)}
    if kind == 'designs_lower':
        before, after = proxies(files[0]), proxies(files[1])
        assert before.keys() == after.keys()
        return {'designs': len(before), 'designs_lower': sum(after[k] < before[k] for k in before),
                'mean_change_percent': 100 * (statistics.mean(after.values()) / statistics.mean(before.values()) - 1)}
    if kind == 'cascade_stages':
        stage = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(DATA / files[0])))]
        cascade = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(DATA / files[1])))]
        return {'designs': len(stage), 'start_mean': statistics.mean(r['start_proxy'] for r in stage),
                'first_stage_mean': statistics.mean(r['final_proxy'] for r in stage),
                'cascade_mean': statistics.mean(r['best_proxy'] for r in cascade)}
    if kind == 'evaluator_ranges':
        summaries, errors = [], []
        for name in files:
            data = json.loads((DATA / name).read_text())
            for bench in data['benchmarks']:
                summaries.append(bench['summary'])
                errors.extend(r['worst_relative_component_error'] for r in bench['repetitions'])
        return {'incremental_ms_min': min(s['mean_incremental_ms_per_move'] for s in summaries),
                'incremental_ms_max': max(s['mean_incremental_ms_per_move'] for s in summaries),
                'full_ms_min': min(s['mean_canonical_ms_per_move'] for s in summaries),
                'full_ms_max': max(s['mean_canonical_ms_per_move'] for s in summaries),
                'median_ratio_min': min(s['median_speedup'] for s in summaries),
                'median_ratio_max': max(s['median_speedup'] for s in summaries),
                'max_relative_component_error': max(errors)}
    raise ValueError(f'unknown frozen check kind {kind}')


def file_hashes(check: dict) -> dict:
    return {str(Path(p).relative_to(DATA)): sha256(Path(p))
            for pattern in check['files'] for p in sorted(glob.glob(str(DATA / pattern)))}


def all_records(doc: dict):
    for row in doc['rows']:
        yield from row['records']
    for entry in doc['chronology'] + doc['related_records']:
        if entry.get('manifest_path'):
            yield entry


def label(record: dict) -> str:
    return record.get('id') or record['name']


def manifest_fields(commit: str, record: dict) -> dict:
    path = record['manifest_path']
    lines = git('show', f'{commit}:{path}').splitlines()
    fields = {'manifest_blob': git('rev-parse', f'{commit}:{path}').strip(),
              'status': manifest_status('\n'.join(lines) + '\n')}
    if 'quote_lines' in record:
        quoted = [{'line': n, 'text': lines[n - 1]} for n in record['quote_lines']]
        if record.get('outcome_verbatim') and not any(record['outcome_verbatim'] in q['text'] for q in quoted):
            raise SystemExit(f'{label(record)}: outcome_verbatim not found in quoted lines of {path}')
        fields['quoted_outcome_lines'] = quoted
    return fields


def note_working_tree(doc: dict) -> None:
    """Later edits to working-tree manifests do not change the frozen blobs; say which differ."""
    changed = sorted({label(r) for r in all_records(doc)
                      if (ROOT / r['manifest_path']).is_file()
                      and git('hash-object', str(ROOT / r['manifest_path'])).strip() != r['manifest_blob']})
    if changed:
        print(f"note: working-tree manifests differ from the frozen blobs for {', '.join(changed)}; "
              f"the record cites the blobs at {doc['source']['commit'][:12]}", file=sys.stderr)


def freeze(doc: dict, commit: str) -> None:
    commit = git('rev-parse', '--verify', f'{commit}^{{commit}}').strip()
    doc['source']['commit'] = commit
    doc['source']['commit_date'] = git('show', '-s', '--format=%cI', commit).strip()
    doc.update(survey(commit))
    for record in all_records(doc):
        record.update(manifest_fields(commit, record))
    for check in doc['frozen_checks']:
        check['sha256'] = file_hashes(check)
        check['values'] = frozen_values(check)
    note_working_tree(doc)


def verify(doc: dict) -> list[str]:
    problems, commit = [], doc['source']['commit']
    recomputed = survey(commit)
    for key in ('status_counts', 'directory_counts'):
        if recomputed[key] != doc[key]:
            problems.append(f'{key} differ: {recomputed[key]} != {doc[key]}')
    for record in all_records(doc):
        expected = manifest_fields(commit, record)
        for key, value in expected.items():
            if record.get(key) != value:
                problems.append(f'{label(record)}: {key} does not reproduce')
    note_working_tree(doc)
    for check in doc['frozen_checks']:
        if file_hashes(check) != check['sha256']:
            problems.append(f"{check['id']}: frozen file hashes differ")
        for key, value in frozen_values(check).items():
            if abs(value - check['values'][key]) > 1e-12:
                problems.append(f"{check['id']}: {key} = {value} != {check['values'][key]}")
    return problems


TEX_CHARS = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
             '−': r'\ensuremath{-}', '×': r'\ensuremath{\times}', '≈': r'\ensuremath{\approx}',
             '–': '--', '—': '---'}


def tex(text: str) -> str:
    """Escape plain text; segments between $ signs pass through as math."""
    parts = text.split('$')
    if len(parts) % 2 == 0:
        raise ValueError(f'unbalanced $ in {text!r}')
    return ''.join(f'${part}$' if i % 2 else ''.join(TEX_CHARS.get(c, c) for c in part)
                   for i, part in enumerate(parts))


# Evidence-label macros defined in writeup/latex/main.tex; used in captions only.
LABEL_MACROS = {'(Historical)': r'(\historical)', '(Measurement)': r'(\measurement)'}


def caption_tex(text: str) -> str:
    out = tex(text)
    for plain, macro in LABEL_MACROS.items():
        out = out.replace(plain, macro)
    return out


VERDICT_MEANING = {'kept': 'adopted into a later pipeline',
                   'dropped': 'abandoned after a negative result',
                   'inconclusive': 'effect too small to separate from run-to-run variation'}


def render_development(doc: dict) -> str:
    rows = doc['rows']
    used = [v for v in VERDICT_MEANING if any(r['verdict'] == v for r in rows)]
    meanings = '; '.join(r'\emph{%s}, %s' % (v, VERDICT_MEANING[v]) for v in used)
    lines = [r'\begin{table}[htbp]', r'\centering', r'\footnotesize',
             r'\caption{%s Verdict records the development decision at the time: %s. '
             r'Appendix~\ref{app:glossary} maps each numbered row to its record.}' % (caption_tex(doc['table_caption']), meanings),
             r'\label{tab:development}',
             r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash\hsize=0.9\hsize}X'
             r'>{\raggedright\arraybackslash\hsize=1.1\hsize}Xl@{}}',
             r'\toprule', r'Idea (plain words) & What happened & Verdict \\', r'\midrule']
    for i, row in enumerate(rows, 1):
        assert row['row'] == i and row['verdict'] in VERDICT_MEANING
        lines.append(f"{i}.~{tex(row['plain_idea'])} & {tex(row['what_happened'])} & {row['verdict']} \\\\")
        if i < len(rows):
            lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabularx}', r'\end{table}', '']
    return '\n'.join(lines)


def record_cell(entry: dict) -> str:
    parts = []
    if entry.get('dev_row'):
        parts.append(r'Row~%d' % entry['dev_row'])
    if entry.get('manifest_blob'):
        parts.append(r'manifest \texttt{%s}' % entry['manifest_blob'][:7])
    if entry.get('record'):
        parts.append(tex(entry['record']))
    return '; '.join(parts)


def render_chronology(doc: dict) -> str:
    """One inline longtable, so the chronology stays in its appendix instead of floating into the next.

    Needs \\usepackage{longtable}. Name (2.1 cm) fits a 12-character hash in footnotesize typewriter, Date
    (1.6 cm) fits "21--22 May" on one line, and Record (4.0 cm) keeps the E1 record to two lines.
    """
    spec = doc['chronology_table']
    groups = [g['group'] for g in spec['groups']]
    assert sorted(groups) == sorted({e['group'] for e in doc['chronology']}), 'every entry needs a known group'
    caption = caption_tex(spec['caption'])
    if any(e.get('dev_row') for e in doc['chronology']):
        caption += r' Row~$n$ refers to the numbered rows of Table~\ref{tab:development}.'
    header = r'Name & Date & What it is & Record \\'
    lines = [r'% Generated by scripts/render_publication_devrecords.py; do not edit.',
             r'\begingroup', r'\footnotesize', r'\setlength{\LTcapwidth}{\linewidth}',
             r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.1cm}>{\raggedright\arraybackslash}p{1.6cm}'
             r'>{\raggedright\arraybackslash}p{\dimexpr\linewidth-7.7cm-6\tabcolsep\relax}'
             r'>{\raggedright\arraybackslash}p{4.0cm}@{}}',
             r'\caption{%s}\label{%s}\\' % (caption, spec['label']),
             r'\toprule', header, r'\midrule', r'\endfirsthead',
             r'\caption*{\textbf{\tablename~\thetable}, continued}\\',
             r'\toprule', header, r'\midrule', r'\endhead',
             r'\bottomrule', r'\endfoot']
    for index, group in enumerate(spec['groups']):
        if index:
            lines.append(r'\addlinespace[4pt]')
        # \\* keeps a group heading on the same page as its first row.
        lines.append(r'\multicolumn{4}{@{}l}{\textit{%s}}\\*' % tex(group['heading']))
        for entry in (e for e in doc['chronology'] if e['group'] == group['group']):
            name = (r'\texttt{%s}' % tex(entry['name'])) if entry.get('monospace') else tex(entry['name'])
            lines.append(f"{name} & {tex(entry['date'])} & {tex(entry['description'])} & {record_cell(entry)} \\\\")
    lines += [r'\end{longtable}', r'\endgroup', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--freeze', metavar='COMMIT', nargs='?', const='HEAD',
                        help='fill computed fields from COMMIT (default HEAD) and the frozen data files, then render')
    parser.add_argument('--no-verify', action='store_true', help='render without checking sources')
    args = parser.parse_args()
    doc = json.loads(RECORDS.read_text())
    if args.freeze:
        freeze(doc, args.freeze)
        RECORDS.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    elif not args.no_verify:
        problems = verify(doc)
        if problems:
            print('\n'.join(problems), file=sys.stderr)
            return 1
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / 'development_records.tex').write_text(render_development(doc))
    (GENERATED / 'chronology.tex').write_text(render_chronology(doc))
    counts = doc['status_counts']
    print(f"Rendered {len(doc['rows'])} development rows and {len(doc['chronology'])} glossary entries "
          f"from {counts['total']} manifests at {doc['source']['commit'][:12]}.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
