#!/usr/bin/env python3
"""Derive baseline context and sampled numerical margins from frozen records."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'writeup/data/publication'
UPSTREAM = '996193c83eaad151e5ae3fb166cedbd83372d060'
HISTORICAL = [
    ('v1', 'CDLNSSACascadeStackedPeripheryPlacer_20260517_062436.json'),
    ('earlier_v2', 'EPYC_thinkorplace_v2_validation_20260520.json'),
    ('gaussian_ancestor', 'E111MinimalGaussianPlacer_20260520_135149.json'),
]
MEASUREMENTS = ['evaluator_ibm01_mac.json', 'evaluator_ibm10_mac.json', 'evaluator_ibm10_hard_mac.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA / 'referee_context_20260907.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve evidence.')
    readme = subprocess.check_output(['git', 'show', UPSTREAM + ':README.md'], cwd=ROOT)
    published = {}
    leaderboard = []
    for line in readme.decode().splitlines():
        cells = [part.strip().replace('**', '') for part in line.split('|')[1:-1]]
        if len(cells) == 7 and cells[0].startswith('ibm'):
            published[cells[0]] = {'sa': float(cells[5]), 'replace': float(cells[6].split()[0])}
        if len(cells) == 9 and cells[0].isdigit() and int(cells[0]) in (1, 2, 9):
            leaderboard.append({'rank': int(cells[0]), 'team': cells[1].strip('"'),
                                'mean_proxy': float(cells[2]), 'notes': cells[8],
                                'verification_field': cells[7]})
    assert len(published) == 17 and len(leaderboard) == 3
    record = {
        'classification': 'referee_response_arithmetic_from_frozen_records',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(Path(__file__).resolve()),
        'official_source': {
            'revision': UPSTREAM, 'readme_sha256': hashlib.sha256(readme).hexdigest(),
            'url': 'https://github.com/partcleda/macro-place-challenge-2026/blob/' + UPSTREAM + '/README.md',
            'leaderboard': leaderboard, 'reported_verified_eligible_submissions': 132,
            'baselines_per_design': published,
            'baseline_arithmetic_means': {k: statistics.mean(row[k] for row in published.values()) for k in ('sa', 'replace')},
        },
        'historical_baseline_comparisons': [], 'sampled_error_margins': [],
        'scope': 'Published reference scores and historical comparisons are not matched reruns. Sampled endpoint errors do not bound arbitrary search states.',
    }
    for name, filename in HISTORICAL:
        path = DATA / 'raw' / filename
        raw = json.loads(path.read_text())
        rows = raw['benchmarks']
        assert len(rows) == 17
        per_design = []
        for row in rows:
            proxy = row['wirelength'] + .5 * row['density'] + .5 * row['congestion']
            assert row['sa_baseline'] == published[row['name']]['sa']
            assert row['replace_baseline'] == published[row['name']]['replace']
            per_design.append({'name': row['name'], 'recomputed_proxy': proxy,
                               'sa_baseline': row['sa_baseline'], 'replace_baseline': row['replace_baseline']})
        record['historical_baseline_comparisons'].append({
            'name': name, 'source': str(path.relative_to(ROOT)), 'source_sha256': digest(path),
            'mean_proxy': statistics.mean(row['recomputed_proxy'] for row in per_design),
            'lower_than_sa_count': sum(row['recomputed_proxy'] < row['sa_baseline'] for row in per_design),
            'lower_than_replace_count': sum(row['recomputed_proxy'] < row['replace_baseline'] for row in per_design),
            'per_design': per_design,
        })
    for filename in MEASUREMENTS:
        path = DATA / filename
        raw = json.loads(path.read_text())
        for benchmark in raw['benchmarks']:
            errors = []
            endpoints = 0
            adjacent_delta_errors = []
            for repetition in benchmark['repetitions']:
                signed_errors = []
                for inc, can in zip(repetition['costs']['incremental'], repetition['costs']['canonical']):
                    signed_errors.append(inc['proxy'] - can['proxy'])
                    errors.append(abs(signed_errors[-1]))
                    endpoints += 1
                adjacent_delta_errors.extend(abs(b-a) for a,b in zip(signed_errors, signed_errors[1:]))
            epsilon = max(errors)
            record['sampled_error_margins'].append({
                'source': str(path.relative_to(ROOT)), 'source_sha256': digest(path),
                'benchmark': benchmark['name'], 'sampled_endpoints_with_repeats': endpoints,
                'distinct_prescribed_moves': len(benchmark['moves']),
                'maximum_absolute_proxy_error': epsilon,
                'maximum_adjacent_delta_error': max(adjacent_delta_errors),
                'twice_observed_endpoint_error': 2*epsilon,
                'cd_threshold_divided_by_twice_observed_error': 1e-9/(2*epsilon) if epsilon else None,
                'assignment_threshold_divided_by_twice_observed_error': 1e-7/(2*epsilon) if epsilon else None,
                'caveat': 'Initial baseline error was not stored. Adjacent saved endpoints are prescribed kernel moves, not an optimizer acceptance trace. The maximum is empirical, not uniform.'
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'official': record['official_source']['leaderboard'],
        'baseline_means': record['official_source']['baseline_arithmetic_means'],
        'historical_counts': [{k:r[k] for k in ('name','lower_than_sa_count','lower_than_replace_count')} for r in record['historical_baseline_comparisons']],
        'margins': record['sampled_error_margins']}, indent=2))

if __name__ == '__main__':
    main()
