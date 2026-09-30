#!/usr/bin/env python3
"""Build a compact, data-backed historical placement explorer fragment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'writeup/interactive'


def main():
    records = []
    for i in range(1, 19):
        if i == 5:
            continue
        name = f'ibm{i:02d}'
        paths = [ROOT/f'benchmarks/processed/public/{name}.pt',
                 ROOT/f'experiments/E74_hessian_saddle/results/hessian_{name}.pt',
                 ROOT/f'experiments/E84_cascading_saddle/results/cascade_{name}.pt']
        frozen = ROOT / 'writeup/data/publication/raw'
        actual_paths = [paths[0], frozen/f'placements/hessian_{name}.pt',
                        frozen/f'placements/cascade_{name}.pt']
        geometry, before, after = [torch.load(p, map_location='cpu', weights_only=False) for p in actual_paths]
        n = int(geometry['num_hard_macros'])
        sidecar = ROOT/f'experiments/E84_cascading_saddle/results/cascade_{name}.json'
        frozen_sidecar = frozen/f'cascade/cascade_{name}.json'
        stats = json.loads(frozen_sidecar.read_text())
        paths.append(sidecar)
        actual_paths.append(frozen_sidecar)
        assert abs(float(before['stats']['best_proxy']) - float(stats['init_proxy'])) < 1e-7
        assert before['placement'].shape == after['placement'].shape == geometry['macro_positions'].shape
        assert torch.isfinite(before['placement']).all() and torch.isfinite(after['placement']).all()
        macros = []
        for row in range(n):
            numbers = list(geometry['macro_sizes'][row]) + list(before['placement'][row]) + list(after['placement'][row])
            macros.append([geometry['macro_names'][row]] + [round(float(value), 5) for value in numbers]
                          + [bool(geometry['macro_fixed'][row])])
        records.append({'benchmark': name, 'width': float(geometry['canvas_width']),
                        'height': float(geometry['canvas_height']), 'n': n,
                        'before': float(stats['init_proxy']), 'after': float(stats['best_proxy']),
                        'overlaps': int(stats['final_overlap']), 'macros': macros,
                        'source': [{'path': str(p.relative_to(ROOT)),
                                    'sha256': hashlib.sha256(actual.read_bytes()).hexdigest()}
                                   for p, actual in zip(paths, actual_paths)]})
    packed = json.dumps(records, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
    template = (OUT/'placement-comparison.template.html').read_text()
    assert template.count('__PLACEMENT_DATA__') == 1
    fragment = template.replace('__PLACEMENT_DATA__', packed)
    assert len(fragment.encode()) < 1_000_000
    (OUT/'placement-comparison.fragment.html').write_text(fragment)
    (OUT/'placement-data.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'{len(records)} designs, {sum(r["n"] for r in records)} hard macros, {len(fragment.encode())} fragment bytes')


if __name__ == '__main__':
    main()
