#!/usr/bin/env python3
"""Apply the top-k stability bound to the frozen evaluator move sequences."""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
DATA = ROOT / 'writeup/data/publication'
FILES = ['evaluator_ibm01_mac.json', 'evaluator_ibm10_mac.json', 'evaluator_ibm10_hard_mac.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = DATA / 'tail_bound_diagnostic_20260907.json'
    if output.exists():
        raise SystemExit('Output already exists; preserve the completed record.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = {
        'classification': 'topk_stability_illustration_on_frozen_move_sequences',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {name: sha(ROOT/name) for name in [
            'scripts/measure_publication_tail_bounds.py', 'macro_place/loader.py',
            'macro_place/objective.py', 'macro_place/incremental_evaluator.py']},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'torch_threads': 1},
        'definition': 'rho=grid_occupied/grid_area; density_fixed_k=0.5*mean(topk(rho,floor(.1G))); bound=0.5*min(norm_inf(delta),norm_2(delta)/sqrt(k),norm_1(delta)/k)',
        'scope': 'New replica-grid derivation on 14 existing prescribed moves; no solver-quality, timing, or fresh canonical-parity claim.',
        'traces': [],
    }
    for filename in FILES:
        source = DATA/filename
        raw = json.loads(source.read_text())
        for benchmark in raw['benchmarks']:
            directory = Path(find_benchmark_dir(benchmark['name']))
            bench, plc = load_benchmark_from_dir(str(directory))
            inputs = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
            assert inputs == benchmark['benchmark_files']
            initial = bench.macro_positions.detach().clone().to(torch.float64)
            assert hashlib.sha256(initial.numpy().tobytes()).hexdigest() == benchmark['initial_tensor_sha256']
            _set_placement(plc, initial, bench)
            evaluator = IncrementalProxyEvaluator(bench, plc, initial.clone())
            count = evaluator.num_cells
            k = int(count*.1)
            assert k >= 1
            trace = {'benchmark': benchmark['name'], 'source': str(source.relative_to(ROOT)),
                     'source_sha256': sha(source), 'benchmark_files': inputs, 'grid_cells': count, 'k': k, 'moves': []}
            for index, (macro, x, y) in enumerate(benchmark['moves']):
                before = (evaluator.grid_occupied / evaluator.grid_area).clone()
                previous = .5*float(torch.topk(before,k).values.mean())
                actual = evaluator.move(macro,(x,y))
                after = evaluator.grid_occupied / evaluator.grid_area
                current = .5*float(torch.topk(after,k).values.mean())
                delta = after-before
                bounds = {'linf': .5*float(delta.abs().max()),
                          'l2': .5*float(torch.linalg.vector_norm(delta))/(k**.5),
                          'l1': .5*float(delta.abs().sum())/k}
                bound = min(bounds.values())
                change = abs(current-previous)
                assert change <= bound + 1e-12
                trace['moves'].append({'index': index, 'macro_index': macro,
                    'density_before_fixed_k': previous, 'density_after_fixed_k': current,
                    'replica_density_after': actual['density'],
                    'replica_vs_fixed_k_absolute_difference': abs(actual['density']-current),
                    'absolute_density_change': change, 'upper_bounds': bounds,
                    'tightest_bound': bound, 'change_divided_by_bound': change/bound if bound else None,
                    'minimum_grid_load_before': float(before.min()), 'minimum_grid_load_after': float(after.min()),
                    'changed_grid_cells': int((delta!=0).sum())})
            result['traces'].append(trace)
            print(benchmark['name'], filename, 'verified', len(trace['moves']), 'tail bounds', flush=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(output)

if __name__ == '__main__':
    main()
