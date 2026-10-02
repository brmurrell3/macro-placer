#!/usr/bin/env python3
"""Render the ranked-configuration reproduction from its run records.

Input: a study run directory written by scripts/run_publication_study.py
(manifest.json, summary.json and runs/<run_id>/result.json), the frozen
leaderboard parse (published SA and RePlAce values per design) and the
corrected-runtime snapshots, which identify the runtime that produced the run.

Outputs (default paths; --out-dir writes all three into one directory):
  writeup/data/publication/reproduction_ranked.json
  writeup/latex/generated/reproduction_summary.tex     (macro definitions only)
  writeup/latex/generated/reproduction_per_design.tex  (Table tab:reproduction)

Coverage rule: the suite mean and other suite means are reported only when each of
the 17 IBM designs has a valid, usable, study-eligible result and the runner's
summary lists every planned run as finished. Otherwise they are withheld, the
macros say so, and the script exits 1 unless --allow-incomplete is given. Integrity
failures (unidentified runtime, inconsistent sources, summary disagreeing with a
result) exit 2 and write nothing. No solver or evaluator runs here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'writeup/data/publication'
GENERATED = ROOT / 'writeup/latex/generated'
LEADERBOARD = DATA / 'leaderboard_996193c.json'
DEFAULT_RUN = 'writeup/runs/ranked-reproduction-20261001-r3'
CONFIG = 'writeup/study/ranked_reproduction.json'
JSON_NAME = 'reproduction_ranked.json'
SUMMARY_NAME = 'reproduction_summary.tex'
TABLE_NAME = 'reproduction_per_design.tex'
CONFIG_KEYS = ('benchmarks', 'budget_seconds', 'cd_seconds', 'device', 'jobs',
               'num_steps', 'seeds', 'threads', 'timeout_seconds', 'variants')
# Checkpoint phases of the single pipeline (Lane A), in plain words.
PHASE_WORDS = {
    'initializer': 'after initialization',
    'descent_legalized': 'after smooth descent and legalization, before coordinate descent',
    'cd1': 'after coordinate descent, before the final check',
    'final': 'at the final check',
}
OUTCOME_NOTES = {
    'CR': 'checkpoint recovery, which returned the last valid checkpoint',
    'ER': 'ended without a valid placement',
    'IV': 'placement failed validation',
    'NE': 'valid but not study-eligible (source or environment mismatch)',
    'NR': 'no run record',
}


class IntegrityError(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def identify_runtime(source: dict) -> dict:
    """Find the frozen runtime whose file digests reproduce the run's source hash.

    The runner hashes each source file's path and digest in path order
    (run_publication_study.sha256_paths); recomputing that from each snapshot's
    per-file digests identifies the runtime without reading the live tree.
    """
    matches = []
    for snap_path in sorted(DATA.glob('corrected_runtime_snapshot_*.json')):
        snap = load(snap_path)
        files = snap['files']
        if any(name not in files for name in source['source_files']):
            continue
        digest = hashlib.sha256()
        for name in sorted(source['source_files']):
            digest.update(name.encode('utf-8'))
            digest.update(b'\0')
            digest.update(bytes.fromhex(files[name]))
        if digest.hexdigest() == source['variant_source_sha256']:
            matches.append((snap_path, snap))
    if len(matches) != 1:
        raise IntegrityError('run source matches %d frozen runtimes, expected exactly one'
                             % len(matches))
    snap_path, snap = matches[0]
    return {'runtime_id': snap['source_identity_sha256'][:12],
            'source_identity_sha256': snap['source_identity_sha256'],
            'archive': snap['archive'], 'archive_sha256': snap['archive_sha256'],
            'snapshot': rel(snap_path),
            'variant_source_sha256': source['variant_source_sha256'],
            'evaluator_source_sha256': source['evaluator_source_sha256'],
            'worker_sha256': source['worker_sha256']}


def strict_bounds(legality: dict) -> bool:
    violations = (legality.get('full_violations', [])
                  + legality.get('structural_violations_without_overlap', []))
    return not any('canvas bounds' in item for item in violations)


def outcome_code(row: dict) -> str:
    if not row['has_record']:
        return 'NR'
    if row['status'] != 'completed' or row['canonical'] is None:
        return 'ER'
    if not row['valid']:
        return 'IV'
    if not (row['usable_operating_result'] and row['study_eligible']):
        return 'NE'
    return 'OK' if row['pipeline_completed'] else 'CR'


def design_row(name: str, run_id: str | None, run_dir: Path, baseline: dict,
               summary_rows: dict) -> dict:
    row = {'design': name, 'label': name.upper(), 'run_id': run_id,
           'replace': baseline['replace'], 'sa': baseline['sa'], 'has_record': False,
           'covered': False, 'canonical': None}
    path = run_dir / 'runs' / str(run_id) / 'result.json'
    if run_id is None or not path.is_file():
        row['outcome_code'] = 'NR'
        return row
    result = load(path)
    if result.get('run_id') != run_id:
        raise IntegrityError('%s names run %s' % (rel(path), result.get('run_id')))
    supervisor = result.get('supervisor') or {}
    legality = result.get('legality') or {}
    overlap = legality.get('hard_macro_overlap') or {}
    phases = (result.get('phase_completion') or {}).get('A') or {}
    metadata = result.get('solver_metadata') or {}
    lanes = metadata.get('lane_results') or []
    picked = [lane for lane in lanes if lane.get('lane_idx') == 0]
    timing = result.get('timing') or {}
    row.update({
        'has_record': True,
        'result_json': rel(path),
        'result_sha256': sha256_file(path),
        'status': result.get('status'),
        'error': result.get('error'),
        'valid': bool(result.get('valid')),
        'usable_operating_result': bool(result.get('usable_operating_result')),
        'study_eligible': bool(result.get('study_eligible')),
        'source_consistent': bool(result.get('source_consistent')),
        'environment_matches_manifest': bool(result.get('environment_matches_manifest')),
        'solver_outcome': result.get('solver_outcome'),
        'pipeline_completed': bool(result.get('pipeline_completed')),
        'used_fallback': result.get('used_fallback'),
        'checkpoint_phase': phases.get('checkpoint_phase'),
        'incomplete_phases': result.get('incomplete_phases', []),
        'wall_seconds': supervisor.get('wall_seconds'),
        'timed_out': supervisor.get('timed_out'),
        'return_code': supervisor.get('return_code'),
        'peak_rss_bytes': supervisor.get('peak_rss_bytes'),
        'solver_place_seconds': timing.get('solver_place_seconds'),
        'bounds_clamp_max_shift_um': picked[0].get('bounds_clamp_max_shift') if picked else None,
        'final_placement_file_sha256': result.get('final_placement_file_sha256'),
        'head_commit': (result.get('repository') or {}).get('head_commit'),
        'source': result.get('source'),
    })
    if legality:
        row.update({'full_legal': bool(legality.get('full_legal')),
                    'strict_bounds': strict_bounds(legality),
                    'overlap_count': overlap.get('overlap_count'),
                    'total_overlap_area': overlap.get('total_overlap_area')})
    canonical = result.get('canonical_cost')
    if canonical is not None:
        proxy = canonical['proxy_cost']
        row['canonical'] = {
            'proxy': proxy,
            'wirelength': canonical['wirelength_cost'],
            'density': canonical['density_cost'],
            'congestion': canonical['congestion_cost'],
            'half_density': 0.5 * canonical['density_cost'],
            'half_congestion': 0.5 * canonical['congestion_cost'],
        }
        row['ratio_to_replace'] = proxy / baseline['replace']
        row['below_replace'] = proxy < baseline['replace']
        row['below_sa'] = proxy < baseline['sa']
    summary = summary_rows.get(run_id)
    row['in_summary'] = summary is not None
    if summary is not None:
        for key in ('status', 'valid', 'study_eligible', 'usable_operating_result',
                    'solver_outcome', 'pipeline_completed'):
            if summary.get(key) != result.get(key):
                raise IntegrityError('summary.json disagrees with %s on %s' % (rel(path), key))
    row['outcome_code'] = outcome_code(row)
    row['covered'] = row['outcome_code'] in ('OK', 'CR') and row['in_summary']
    if row['covered'] and (row.get('overlap_count') != 0 or not row.get('strict_bounds')):
        raise IntegrityError('%s is valid but reports overlaps or a bounds excess' % rel(path))
    return row


def local_dates(start: str, end: str, zone: str) -> dict:
    tz = ZoneInfo(zone)
    first = datetime.datetime.fromisoformat(start).astimezone(tz).date()
    last = datetime.datetime.fromisoformat(end).astimezone(tz).date()
    if first == last:
        text = '%d %s %d' % (first.day, first.strftime('%B'), first.year)
    elif (first.year, first.month) == (last.year, last.month):
        text = '%d--%d %s %d' % (first.day, last.day, first.strftime('%B'), first.year)
    else:
        text = '%d %s %d--%d %s %d' % (first.day, first.strftime('%B'), first.year,
                                       last.day, last.strftime('%B'), last.year)
    return {'timezone': zone, 'first_local_date': first.isoformat(),
            'last_local_date': last.isoformat(), 'text': text}


def build_record(run_dir: Path, zone: str) -> dict:
    manifest = load(run_dir / 'manifest.json')
    summary_path = run_dir / 'summary.json'
    summary = load(summary_path) if summary_path.is_file() else None
    summary_rows = {r['run_id']: r for r in (summary or {}).get('runs', [])}
    leaderboard = load(LEADERBOARD)
    baselines = {b['name']: b for b in leaderboard['baseline_per_design']}
    invocation = manifest['invocation']

    config_path = ROOT / CONFIG
    config = load(config_path)
    for key in CONFIG_KEYS:
        if config.get(key) != invocation.get(key):
            raise IntegrityError('manifest %s differs from %s' % (key, CONFIG))
    if len(invocation['seeds']) != 1 or len(invocation['variants']) != 1:
        raise IntegrityError('expected one seed and one variant per design')

    planned = list(invocation['benchmarks'])
    suite = list(baselines)
    run_ids = {}
    for run in manifest['runs']:
        if run['benchmark'] in run_ids:
            raise IntegrityError('more than one run planned for %s' % run['benchmark'])
        run_ids[run['benchmark']] = run['run_id']
    rows = [design_row(name, run_ids.get(name), run_dir, baselines[name], summary_rows)
            for name in suite]

    recorded = [r for r in rows if r['has_record'] and r.get('source')]
    if not recorded:
        raise IntegrityError('no run records under %s' % rel(run_dir))
    sources = {json.dumps(r['source'], sort_keys=True) for r in recorded}
    # A worker that never finished has not rechecked its source, so only
    # completed runs must report a consistent source.
    if len(sources) != 1 or not all(r['source_consistent'] for r in recorded
                                    if r['status'] == 'completed'):
        raise IntegrityError('run records disagree on their source')
    commit = manifest['repository']['head_commit']
    if any(r['head_commit'] != commit for r in recorded):
        raise IntegrityError('run records disagree with the manifest commit')
    runtime = identify_runtime(recorded[0]['source'])
    for r in rows:
        r.pop('source', None)
        r.pop('head_commit', None)

    covered = [r for r in rows if r['covered']]
    finished = (summary is not None
                and summary.get('planned_run_count') == len(manifest['runs'])
                and summary.get('finished_run_count') == summary.get('planned_run_count'))
    complete = planned == suite and len(covered) == len(suite) and finished
    missing = [r['label'] for r in rows if not r['covered']]

    def mean_or_none(values):
        return statistics.mean(values) if complete else None

    proxies = [r['canonical']['proxy'] for r in covered]
    walls = [r['wall_seconds'] for r in covered]
    best = min(covered, key=lambda r: r['canonical']['proxy']) if covered else None
    worst = max(covered, key=lambda r: r['canonical']['proxy']) if covered else None
    ratios = [r['ratio_to_replace'] for r in covered]
    mean_proxy = mean_or_none(proxies)
    component_means = {
        key: mean_or_none([r['canonical'][key] for r in covered])
        for key in ('wirelength', 'half_density', 'half_congestion')}
    replace_mean = leaderboard['baselines']['replace']['mean_proxy']
    aggregates = {
        'complete': complete,
        'coverage_note': ('all %d designs have a valid, usable, study-eligible result'
                          % len(suite)) if complete else
                         ('suite means withheld: valid, usable, study-eligible results '
                          'cover %d of %d designs%s; missing %s'
                          % (len(covered), len(suite),
                             '' if finished else ', and the run summary is not final',
                             ', '.join(missing) or 'none')),
        'designs_planned': len(suite),
        'designs_with_record': sum(r['has_record'] for r in rows),
        'designs_covered': len(covered),
        'valid_count': sum(bool(r.get('valid')) for r in rows),
        'pipeline_completed_count': sum(bool(r.get('pipeline_completed')) for r in rows),
        'checkpoint_recovery_count': sum(r['outcome_code'] == 'CR' for r in rows),
        'mean_proxy': mean_proxy,
        'component_means': component_means,
        'congestion_share_percent': (100 * component_means['half_congestion'] / mean_proxy
                                     if complete else None),
        'mean_ratio_to_replace': mean_or_none(ratios),
        'mean_proxy_over_replace_mean': (mean_proxy / replace_mean if complete else None),
        'mean_wall_seconds': mean_or_none(walls),
        'over_covered_designs': {
            'min_proxy': best['canonical']['proxy'] if best else None,
            'min_design': best['label'] if best else None,
            'max_proxy': worst['canonical']['proxy'] if worst else None,
            'max_design': worst['label'] if worst else None,
            'median_wall_seconds': statistics.median(walls) if walls else None,
            'max_wall_seconds': max(walls) if walls else None,
            'below_replace_count': sum(r['below_replace'] for r in covered),
            'below_sa_count': sum(r['below_sa'] for r in covered),
            'ratio_to_replace_min': min(ratios) if ratios else None,
            'ratio_to_replace_max': max(ratios) if ratios else None,
        },
    }

    starts = [a['started_at'] for r in rows if r['has_record']
              for a in load(ROOT / r['result_json']).get('attempts', [])]
    ends = [a['finished_at'] for r in rows if r['has_record']
            for a in load(ROOT / r['result_json']).get('attempts', [])]
    host = manifest['host']
    environment = load(ROOT / recorded[0]['result_json']).get('environment') or {}
    return {
        'schema_version': 1,
        'kind': 'post-competition-reproduction-ranked-config',
        'evidence_label': 'Reproduction',
        'generator': 'scripts/render_publication_reproduction.py',
        'run_directory': rel(run_dir),
        'study_id': manifest['study_id'],
        'manifest_sha256': sha256_file(run_dir / 'manifest.json'),
        'summary_sha256': sha256_file(summary_path) if summary else None,
        'summary_counts': ({k: v for k, v in summary.items() if k.endswith('_count')}
                           if summary else None),
        'config': {'path': CONFIG, 'sha256': sha256_file(config_path),
                   **{key: invocation[key] for key in CONFIG_KEYS},
                   'variant_parameters': config.get('variant_parameters')},
        'repository': manifest['repository'],
        'commit': commit,
        'runtime': runtime,
        'host': {'cpu_model': host.get('cpu_model'), 'os': host.get('os'),
                 'machine': host.get('machine'),
                 'logical_cpu_count': host.get('logical_cpu_count'),
                 'total_memory_bytes': host.get('total_memory_bytes'),
                 'python': '%s %s' % (host.get('python_implementation'),
                                      host.get('python_version')),
                 'torch': environment.get('torch_version'),
                 'device': invocation['device'],
                 'threads': invocation['threads'],
                 'torch_num_threads': environment.get('torch_num_threads'),
                 'installed_distributions_sha256': host.get('installed_distributions_sha256')},
        'dates': {'manifest_generated_at_utc': manifest['generated_at'],
                  'first_attempt_started_at_utc': min(starts) if starts else None,
                  'last_attempt_finished_at_utc': max(ends) if ends else None,
                  **local_dates(min(starts), max(ends), zone)} if starts else None,
        'baselines': {'source': rel(LEADERBOARD),
                      'replace_mean': replace_mean,
                      'sa_mean': leaderboard['baselines']['sa']['mean_proxy']},
        'designs': rows,
        'aggregates': aggregates,
    }


def fmt_proxy(value: float) -> str:
    return '%.5f' % value


def fmt_ratio(value: float) -> str:
    return '%.3f' % value


def fmt_seconds(value: float) -> str:
    return '{:,.0f}'.format(value)


def render_summary(record: dict) -> str:
    agg = record['aggregates']
    cov = agg['over_covered_designs']
    complete = agg['complete']
    withheld = 'withheld'

    def suite(value, fmt):
        return fmt(value) if complete else withheld

    def partial(value, fmt):
        return fmt(value) if value is not None else '---'

    macros = [
        ('ReproMean', suite(agg['mean_proxy'], fmt_proxy)),
        ('ReproMin', partial(cov['min_proxy'], fmt_proxy)),
        ('ReproMinDesign', cov['min_design'] or '---'),
        ('ReproMax', partial(cov['max_proxy'], fmt_proxy)),
        ('ReproMaxDesign', cov['max_design'] or '---'),
        ('ReproCompleted', str(agg['pipeline_completed_count'])),
        ('ReproValid', str(agg['designs_covered'])),
        ('ReproDesigns', str(agg['designs_planned'])),
        ('ReproWallMedian', partial(cov['median_wall_seconds'], fmt_seconds)),
        ('ReproWallMax', partial(cov['max_wall_seconds'], fmt_seconds)),
        ('ReproBelowReplace', str(cov['below_replace_count'])),
        ('ReproRatioMin', partial(cov['ratio_to_replace_min'], fmt_ratio)),
        ('ReproRatioMax', partial(cov['ratio_to_replace_max'], fmt_ratio)),
        ('ReproWLMean', suite(agg['component_means']['wirelength'], fmt_proxy)),
        ('ReproDenMean', suite(agg['component_means']['half_density'], fmt_proxy)),
        ('ReproCongMean', suite(agg['component_means']['half_congestion'], fmt_proxy)),
        ('ReproCongShare', suite(agg['congestion_share_percent'], lambda v: '%.1f' % v)),
        ('ReproRuntimeId', record['runtime']['runtime_id']),
        ('ReproCommit', record['commit'][:7]),
        ('ReproDate', record['dates']['text'] if record['dates'] else '---'),
    ]
    lines = ['% Generated by scripts/render_publication_reproduction.py from '
             + record['run_directory'] + '; do not edit.',
             '% Proxy values have 5 decimals, ratios 3, times are whole seconds, and',
             '% ReproCongShare is a percentage without the sign. ' + agg['coverage_note'] + '.',
             r'\newif\ifReproComplete',
             r'\ReproComplete%s' % ('true' if complete else 'false')]
    lines += [r'\newcommand{\%s}{%s}' % (name, value) for name, value in macros]
    return '\n'.join(lines) + '\n'


def render_table(record: dict) -> str:
    agg = record['aggregates']
    rows = record['designs']
    body = []
    for r in rows:
        code = r['outcome_code']
        outcome = r'$\checkmark$' if code == 'OK' else code
        c = r['canonical']
        if r['covered']:
            cells = [fmt_proxy(c['proxy']), fmt_proxy(c['wirelength']),
                     fmt_proxy(c['half_density']), fmt_proxy(c['half_congestion']),
                     fmt_ratio(r['ratio_to_replace']), fmt_seconds(r['wall_seconds'])]
        else:
            cells = ['---'] * 6
        body.append(' & '.join([r['label']] + cells + [outcome]) + r' \\')
    if agg['complete']:
        means = agg['component_means']
        mean_row = ' & '.join([
            'Mean', fmt_proxy(agg['mean_proxy']), fmt_proxy(means['wirelength']),
            fmt_proxy(means['half_density']), fmt_proxy(means['half_congestion']),
            fmt_ratio(agg['mean_proxy_over_replace_mean']),
            fmt_seconds(agg['mean_wall_seconds']), '']) + r' \\'
    else:
        mean_row = (r'Mean & \multicolumn{7}{l}{withheld: valid, usable, study-eligible '
                    r'results cover %d of %d designs} \\'
                    % (agg['designs_covered'], agg['designs_planned']))

    notes = [r'$\checkmark$: full pipeline completed']
    present = sorted({r['outcome_code'] for r in rows} - {'OK'})
    for code in present:
        designs = [r for r in rows if r['outcome_code'] == code]
        if code == 'CR':
            by_phase = {}
            for r in designs:
                by_phase.setdefault(r['checkpoint_phase'], []).append(r['label'])
            detail = '; '.join('%s: taken %s' % (', '.join(labels),
                                                PHASE_WORDS.get(phase, phase or 'unknown'))
                               for phase, labels in sorted(by_phase.items()))
            notes.append('CR: %s (%s)' % (OUTCOME_NOTES[code], detail))
        else:
            notes.append('%s: %s (%s)' % (code, OUTCOME_NOTES[code],
                                          ', '.join(r['label'] for r in designs)))
    caption = (
        r'Reproduction (post-competition), ranked configuration rerun on the lead '
        r"author's Apple M3 Max CPU, one design at a time; all values from the canonical "
        r'evaluator. Proxy = wirelength + $\tfrac{1}{2}$\,density + '
        r'$\tfrac{1}{2}$\,congestion. Ratio to RePlAce: each proxy over the published '
        r'RePlAce value for its design; in the mean row, the mean proxy over the RePlAce '
        r'mean (%s). Wall: supervised time per design, including final validation '
        r'(%s\,s solver budget, %s\,s limit). '
        % ('%.4f' % record['baselines']['replace_mean'],
           fmt_seconds(record['config']['budget_seconds']),
           fmt_seconds(record['config']['timeout_seconds']))
        + '; '.join(notes) + '.')
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\footnotesize',
        r'\caption{%s}' % caption,
        r'\label{tab:reproduction}',
        r'\begin{tabular}{lrrrrrrc}',
        r'\toprule',
        r'Design & Proxy & Wirelength & $\tfrac{1}{2}$\,Density & $\tfrac{1}{2}$\,Congestion'
        r' & Ratio to RePlAce & Wall (s) & Outcome \\',
        r'\midrule',
        *body,
        r'\midrule',
        mean_row,
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', default=DEFAULT_RUN,
                        help='study run directory (default: %(default)s)')
    parser.add_argument('--out-dir', type=Path,
                        help='write all three outputs into this directory instead of the '
                             'repository paths')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='exit 0 even when coverage is incomplete (means are still withheld)')
    parser.add_argument('--timezone', default='America/New_York',
                        help="IANA zone of the run host's local dates; the records store UTC "
                             '(default: %(default)s)')
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    try:
        record = build_record(run_dir, args.timezone)
    except (IntegrityError, OSError, KeyError, ValueError) as exc:
        print('reproduction render failed: %s' % exc, file=sys.stderr)
        return 2

    if args.out_dir:
        out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
        targets = (out / JSON_NAME, out / SUMMARY_NAME, out / TABLE_NAME)
    else:
        targets = (DATA / JSON_NAME, GENERATED / SUMMARY_NAME, GENERATED / TABLE_NAME)
    texts = (json.dumps(record, indent=2, sort_keys=True) + '\n',
             render_summary(record), render_table(record))
    for path, text in zip(targets, texts):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        print('wrote %s' % rel(path))

    agg = record['aggregates']
    print(agg['coverage_note'])
    if agg['complete']:
        print('mean proxy %.5f over %d designs; %d pipelines completed'
              % (agg['mean_proxy'], agg['designs_planned'], agg['pipeline_completed_count']))
    if record['repository'].get('dirty'):
        print('warning: the run manifest records a dirty working tree', file=sys.stderr)
    if not agg['complete'] and not args.allow_incomplete:
        print('coverage incomplete: suite means withheld (pass --allow-incomplete to accept)',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
