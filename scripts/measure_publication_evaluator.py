#!/usr/bin/env python3
"""Measure matched component evaluation, with setup and overlap costs separate.

Both engines consume exactly the same accepted-move sequence and return W,D,C
and their weighted sum. This is a kernel microbenchmark, not a solver speedup.
The full engine uses the canonical placement setter and PLC component methods;
the canonical wrapper's separate overlap calculation is timed on the final state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import _set_placement, compute_overlap_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def components(pos, bench, plc):
    _set_placement(plc, pos, bench)
    wl, density, congestion = plc.get_cost(), plc.get_density_cost(), plc.get_congestion_cost()
    return dict(wl=wl, density=density, congestion=congestion,
                proxy=wl + .5 * density + .5 * congestion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', nargs='+', default=['ibm01', 'ibm10'])
    parser.add_argument('--moves', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--macro-kind', choices=['all', 'hard', 'soft'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.moves, args.repetitions, args.threads) <= 0:
        parser.error('moves, repetitions, and threads must be positive')
    if args.output.exists():
        parser.error('output already exists; use a new path to preserve evidence')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = {
        'schema_version': 1, 'kind': 'matched_component_microbenchmark',
        'measurement_status': 'in_progress',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'environment': {'platform': platform.platform(), 'python': sys.version,
                        'torch': torch.__version__, 'cpu_count': os.cpu_count(),
                        'device': 'cpu', 'torch_threads': torch.get_num_threads()},
        'head_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: digest(ROOT / p) for p in [
            'scripts/measure_publication_evaluator.py', 'macro_place/objective.py',
            'macro_place/incremental_evaluator.py', 'macro_place/loader.py', 'macro_place/_plc.py']},
        'protocol': {
            'timed_work': 'Apply accepted macro move and compute W,D,C,proxy; no overlap validation in either engine',
            'order': 'Incremental then canonical in even repetitions; reversed in odd repetitions',
            'warmup': 'One unmeasured component evaluation per engine before each timed sequence',
            'sampling': 'Same seeded sequence across repetitions; uniformly chosen movable tensor row within requested macro-kind, uniform in-bounds center',
            'geometry': 'Moves need not preserve hard-macro nonoverlap; no optimizer or downstream PnR is timed',
            'overlap_timing': 'One canonical overlap-metrics call at final placement per repetition, outside ratio',
            'precision': 'float64 positions; component parity rtol=1e-5, atol=1e-9',
        }, 'benchmarks': []}
    # The post-competition adapter imports these implementations when present.
    manifest['runtime_package_sha256'] = {str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / 'thinkorplace').rglob('*.py'))}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
    for name in args.benchmarks:
        bench_dir = Path(find_benchmark_dir(name))
        load_start = time.perf_counter()
        bench, plc = load_benchmark_from_dir(str(bench_dir))
        load_seconds = time.perf_counter() - load_start
        initial = bench.macro_positions.detach().clone().to(torch.float64)
        rng = random.Random(args.seed)
        movable = (~bench.macro_fixed).nonzero(as_tuple=True)[0].tolist()
        if args.macro_kind == 'hard':
            movable = [i for i in movable if i < bench.num_hard_macros]
        elif args.macro_kind == 'soft':
            movable = [i for i in movable if i >= bench.num_hard_macros]
        if not movable:
            raise RuntimeError(f'No movable {args.macro_kind} macros in {name}')
        moves = []
        for _ in range(args.moves):
            i = rng.choice(movable)
            w, h = (float(v) for v in bench.macro_sizes[i])
            moves.append([i, rng.uniform(w/2 + 1e-3, bench.canvas_width-w/2-1e-3),
                          rng.uniform(h/2 + 1e-3, bench.canvas_height-h/2-1e-3)])
        record = {'name': name, 'num_macros': bench.num_macros,
                  'num_hard_macros': bench.num_hard_macros,
                  'load_seconds': load_seconds, 'moves': moves,
                  'initial_tensor_sha256': hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                  'benchmark_files': {str(p.relative_to(bench_dir)): digest(p)
                                      for p in sorted(bench_dir.rglob('*')) if p.is_file()},
                  'repetitions': []}
        manifest['benchmarks'].append(record)
        save()
        for repetition in range(args.repetitions):
            # Restore canonical state before constructing the incremental cache.
            components(initial, bench, plc)
            setup_start = time.perf_counter()
            evaluator = IncrementalProxyEvaluator(bench, plc, initial.clone())
            evaluator.current_cost()
            setup_seconds = time.perf_counter() - setup_start
            full_pos = initial.clone()
            components(full_pos, bench, plc)
            durations, values = {}, {}
            order = ['incremental', 'canonical'] if repetition % 2 == 0 else ['canonical', 'incremental']
            for engine in order:
                output, seconds = [], []
                for i, x, y in moves:
                    start = time.perf_counter()
                    if engine == 'incremental':
                        value = evaluator.move(i, (x, y))
                    else:
                        full_pos[i, 0], full_pos[i, 1] = x, y
                        value = components(full_pos, bench, plc)
                    seconds.append(time.perf_counter() - start)
                    output.append({k: float(v) for k,v in value.items()})
                durations[engine], values[engine] = seconds, output
            worst_absolute, worst_relative = 0., 0.
            for increment, canonical in zip(values['incremental'], values['canonical']):
                for key in ('wl', 'density', 'congestion', 'proxy'):
                    a, b = increment[key], canonical[key]
                    error = abs(a-b)
                    worst_absolute = max(worst_absolute, error)
                    worst_relative = max(worst_relative, error / max(abs(a), abs(b), 1e-30))
                    if error > max(1e-9, 1e-5 * max(abs(a), abs(b))):
                        raise RuntimeError(f'Parity failure: {name} rep {repetition} {key}: {a} vs {b}')
            start = time.perf_counter()
            overlap = compute_overlap_metrics(full_pos, bench)
            overlap_seconds = time.perf_counter() - start
            run = {'index': repetition, 'order': order,
                   'incremental_setup_seconds': setup_seconds, 'per_move_seconds': durations,
                   'costs': values, 'worst_absolute_component_error': worst_absolute,
                   'worst_relative_component_error': worst_relative,
                   'final_overlap_metrics': overlap, 'final_overlap_check_seconds': overlap_seconds,
                   'component_speedup': sum(durations['canonical']) / sum(durations['incremental'])}
            record['repetitions'].append(run)
            save()
            print(f'{name} repetition {repetition + 1}: matched component ratio {run["component_speedup"]:.2f}x', flush=True)
        record['summary'] = {
            'median_speedup': statistics.median(r['component_speedup'] for r in record['repetitions']),
            'min_speedup': min(r['component_speedup'] for r in record['repetitions']),
            'max_speedup': max(r['component_speedup'] for r in record['repetitions']),
            'mean_incremental_ms_per_move': 1000 * statistics.mean(t for r in record['repetitions'] for t in r['per_move_seconds']['incremental']),
            'mean_canonical_ms_per_move': 1000 * statistics.mean(t for r in record['repetitions'] for t in r['per_move_seconds']['canonical']),
            'mean_incremental_setup_seconds': statistics.mean(r['incremental_setup_seconds'] for r in record['repetitions']),
            'mean_final_overlap_seconds': statistics.mean(r['final_overlap_check_seconds'] for r in record['repetitions']),
        }
        save()
    manifest['measurement_status'] = 'completed'
    save()
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
