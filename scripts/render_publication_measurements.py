#!/usr/bin/env python3
"""Regenerate measured report fragments from complete microbenchmark records."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'writeup/data/publication'
GENERATED = ROOT / 'writeup/latex/generated'
FILES = ['evaluator_ibm01_mac.json', 'evaluator_ibm10_mac.json', 'evaluator_ibm10_hard_mac.json']


def main():
    records = []
    for filename in FILES:
        path = DATA / filename
        data = json.loads(path.read_text())
        if data.get('measurement_status') != 'completed':
            raise SystemExit(f'Measurement is not complete: {filename}')
        for bench in data['benchmarks']:
            assert len(bench['repetitions']) == data['config']['repetitions']
            assert all(len(r['per_move_seconds']['canonical']) == data['config']['moves'] for r in bench['repetitions'])
            hard = sum(move[0] < bench['num_hard_macros'] for move in bench['moves'])
            kind = 'hard' if hard == len(bench['moves']) else 'soft' if hard == 0 else 'mixed'
            records.append({'file': filename, 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'name': bench['name'], 'macro_kind_observed': kind,
                            'moves_per_repetition': len(bench['moves']), 'hard_moves_per_repetition': hard,
                            'repetitions': len(bench['repetitions']), 'summary': bench['summary'],
                            'maximum_relative_component_error': max(r['worst_relative_component_error'] for r in bench['repetitions'])})
    lines = [r'\subsection{Matched incremental-evaluator measurements}',
             r'\label{sec:evaluator}',
             'We measured identical prescribed move sequences through two evaluation paths: '
             'incremental updates returning wirelength, density, congestion, and proxy; and '
             'the canonical placement setter followed by the three full component computations. '
             'Neither timed path includes hard-overlap validation. Both use float64 positions '
             'on the Mac CPU with one PyTorch thread. The order of the two engines alternates '
             'between repetitions; initialization and a warm-up component evaluation precede each batch.',
             '', r'\begin{table}[htbp]\centering\footnotesize',
             r'\caption{CPU component-kernel microbenchmarks. Times are means per move; each ratio is computed for a paired batch, then summarized by its median and range. These are not end-to-end placer speedups.}',
             r'\label{tab:incremental-measurements}',
             r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Design / sample & Moves $\times$ repeats & Inc. time (ms) & Full time (ms) & Median ratio & Range \\', r'\midrule']
    for record in records:
        s = record['summary']
        lines.append(f"{record['name'].upper()} / {record['macro_kind_observed']} & "
                     f"${record['moves_per_repetition']} \\times {record['repetitions']}$ & "
                     f"{s['mean_incremental_ms_per_move']:.2f} & {s['mean_canonical_ms_per_move']:.1f} & "
                     f"{s['median_speedup']:.0f}$\\times$ & {s['min_speedup']:.0f}--{s['max_speedup']:.0f}" + r" \\")
    lines += [r'\bottomrule\end{tabular}\end{table}', '']
    worst = max(r['maximum_relative_component_error'] for r in records)
    mantissa, exponent = f'{worst:.1e}'.split('e')
    scientific_error = mantissa + r'\times 10^{' + str(int(exponent)) + '}'
    lines.append('Every measured component agreed within relative tolerance $10^{-5}$, using '
                 r'$|a-b|/\max(|a|,|b|,10^{-30})$; '
                 f'the largest observed relative discrepancy was ${scientific_error}$. '
                 'Evaluator setup is excluded from the ratios and retained in the raw records. '
                 'The selected sequences are small: IBM01 includes three hard-macro moves and five '
                 'soft-cluster moves per repetition; the first IBM10 sequence happened to select only '
                 'soft clusters, so a separate hard-only sequence was measured. Moves may create overlaps. '
                 'These samples show the potential savings from localized cache updates, but do not '
                 'estimate a suite-wide throughput distribution or the gain of replacing another optimized evaluator.')
    setups = ', '.join(f"{r['name'].upper()} {r['macro_kind_observed']}: {r['summary']['mean_incremental_setup_seconds']:.2f} s" for r in records)
    lines.extend(['', f'Mean incremental setup times were {setups}. '
                  'A separate overlap-metrics call on each final state is also timed in the raw data; '
                  'it is excluded from both component paths and is not the full geometry-validation cost. '
                  'Repetitions share the same seed and moves, so their spread measures timing repeatability, '
                  'not independent placement outcomes.', ''])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / 'evaluator_results.tex').write_text('\n'.join(lines))
    (DATA / 'evaluator_measurement_summary.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'Generated measured report table from {len(records)} complete samples.')


if __name__ == '__main__':
    main()
