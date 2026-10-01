#!/usr/bin/env python3
"""Build the data-backed historical placement explorer: fragment, data, and standalone page."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'writeup/interactive'
FROZEN = ROOT / 'writeup/data/publication'
RESCORE = FROZEN / 'e74_e84_canonical_rescore_20260907.json'
CROSSCHECK = FROZEN / 'e74_e84_geometry_crosscheck_20260907.json'
# The saved "before cascade" endpoints are the E74 stage; "after cascade" is E84.
STAGES = {'before': 'E74', 'after': 'E84'}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curvature-cascade case studies</title>
<meta name="description" content="Saved hard-macro placements before and after the May 2026 curvature-cascade experiments on 17 public IBM designs (historical, v1 lineage).">
<style>
  :root { color-scheme: light; --page-bg: #f9f9f7; --page-ink: #0b0b0b; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { color-scheme: dark; --page-bg: #0d0d0d; --page-ink: #ffffff; }
  }
  :root[data-theme="dark"] { color-scheme: dark; --page-bg: #0d0d0d; --page-ink: #ffffff; }
  html, body { margin: 0; }
  body { background: var(--page-bg); color: var(--page-ink); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { box-sizing: border-box; max-width: 760px; margin: 0 auto; padding: 24px 16px 32px; }
</style>
</head>
<body>
<main>
__FRAGMENT__
</main>
</body>
</html>
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validity(rescored, crosschecked):
    """Strict-geometry status of one saved placement, from the September rescore."""
    geometry = rescored['geometry_validation']
    bounds_ok = bool(geometry['bounds']['all_within_or_on_canvas'])
    overlaps = int(geometry['hard_macro_overlap']['overlap_count'])
    # The two September audits must agree, and the bounds/overlap label must equal strict validity.
    assert crosschecked['agrees_with_rescore_geometry']
    assert crosschecked['canonical_validate_placement']['full_legal'] == geometry['full_legal']
    assert geometry['full_legal'] == (bounds_ok and overlaps == 0)
    return {'hard_overlaps': overlaps, 'strict_bounds': bounds_ok,
            'boundary_excess': float(geometry['bounds']['maximum_boundary_excess']),
            'out_of_bounds_hard': int(crosschecked['out_of_bounds_hard_count']),
            'out_of_bounds_soft': int(crosschecked['out_of_bounds_soft_count'])}


def main():
    rescore = json.loads(RESCORE.read_text())
    crosscheck = json.loads(CROSSCHECK.read_text())
    assert crosscheck['source_rescore']['sha256'] == sha256(RESCORE)
    rescored = {(b['name'], p['stage']): p for b in rescore['benchmarks'] for p in b['placements']}
    crosschecked = {(p['benchmark'], p['stage']): p for p in crosscheck['placements']}
    audit_sources = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in (RESCORE, CROSSCHECK)]
    records = []
    for i in range(1, 19):
        if i == 5:
            continue
        name = f'ibm{i:02d}'
        paths = [ROOT/f'benchmarks/processed/public/{name}.pt',
                 ROOT/f'experiments/E74_hessian_saddle/results/hessian_{name}.pt',
                 ROOT/f'experiments/E84_cascading_saddle/results/cascade_{name}.pt']
        frozen = FROZEN / 'raw'
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
        status = {}
        for label, tensor, recorded in (('before', actual_paths[1], stats['init_proxy']),
                                        ('after', actual_paths[2], stats['best_proxy'])):
            placement = rescored[(name, STAGES[label])]
            assert placement['source_artifacts']['tensor_file_sha256'] == sha256(tensor)
            assert abs(placement['fresh_canonical']['proxy_cost'] - float(recorded)) < 1e-6
            status[label] = validity(placement, crosschecked[(name, STAGES[label])])
        assert status['after']['hard_overlaps'] == int(stats['final_overlap'])
        macros = []
        for row in range(n):
            numbers = list(geometry['macro_sizes'][row]) + list(before['placement'][row]) + list(after['placement'][row])
            macros.append([geometry['macro_names'][row]] + [round(float(value), 5) for value in numbers]
                          + [bool(geometry['macro_fixed'][row])])
        records.append({'benchmark': name, 'width': float(geometry['canvas_width']),
                        'height': float(geometry['canvas_height']), 'n': n,
                        'before': float(stats['init_proxy']), 'after': float(stats['best_proxy']),
                        'validity': status, 'macros': macros,
                        'source': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(actual)}
                                   for p, actual in zip(paths, actual_paths)] + audit_sources})
    strict = sum(r['validity'][s]['strict_bounds'] for r in records for s in STAGES)
    assert strict == crosscheck['summary']['fully_legal_count']
    packed = json.dumps(records, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
    template = (OUT/'placement-comparison.template.html').read_text()
    assert template.count('__PLACEMENT_DATA__') == 1
    fragment = template.replace('__PLACEMENT_DATA__', packed)
    assert len(fragment.encode()) < 1_000_000
    assert PAGE.count('__FRAGMENT__') == 1
    (OUT/'placement-comparison.fragment.html').write_text(fragment)
    (OUT/'placement-comparison.html').write_text(PAGE.replace('__FRAGMENT__', fragment.rstrip('\n')))
    (OUT/'placement-data.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'{len(records)} designs, {sum(r["n"] for r in records)} hard macros, '
          f'{strict}/{2 * len(records)} placements pass strict bounds, {len(fragment.encode())} fragment bytes')


if __name__ == '__main__':
    main()
