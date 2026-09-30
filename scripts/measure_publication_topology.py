#!/usr/bin/env python3
"""Isolate the hierarchical-owner parsing change; do not rerun optimization."""
from __future__ import annotations
import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from macro_place.loader import load_benchmark_from_dir
from thinkorplace.common.bench_paths import find_benchmark_dir
from thinkorplace.common import smooth_primitives


def main():
    output = ROOT / 'writeup/data/publication/topology_diagnostic.json'
    if output.exists():
        raise SystemExit('Evidence exists; preserve it and choose a new path for a changed study.')
    fixed_source = inspect.getsource(smooth_primitives._extract_net_data)
    assert fixed_source.count('parent_name = pin.get_macro_name()') == 1
    old_source = fixed_source.replace('parent_name = pin.get_macro_name()', 'parent_name = pin_name.split("/")[0]')
    namespace = dict(smooth_primitives.__dict__)
    exec(compile(old_source, '<old-owner-extraction-control>', 'exec'), namespace)
    old_extractor = namespace['_extract_net_data']
    records = []
    for name in ['ibm01', 'ariane133']:
        directory = find_benchmark_dir(name)
        bench, plc = load_benchmark_from_dir(str(directory))
        hard_ids = set(bench.hard_macro_indices)
        def is_hard_pin(node):
            return (node.get_type() in ('MACRO_PIN', 'SOFT_MACRO_PIN') and
                    plc.mod_name_to_indices.get(node.get_macro_name()) in hard_ids)
        raw = [n for n in plc.modules_w_pins if is_hard_pin(n)]
        occurrences = [plc.modules_w_pins[plc.mod_name_to_indices[pin_name]]
                       for driver, sinks in plc.nets.items() for pin_name in [driver] + sinks
                       if pin_name in plc.mod_name_to_indices and
                       is_hard_pin(plc.modules_w_pins[plc.mod_name_to_indices[pin_name]])]
        def describe(extractor):
            data = extractor(bench, plc)
            return {'retained_nets': int(data.mask.shape[0]),
                    'retained_pin_occurrences': int(data.mask.sum()),
                    'hard_pin_occurrences': int(((data.pin_macro_idx < bench.num_hard_macros) & data.mask).sum())}
        records.append({'benchmark': name, 'hard_macros': bench.num_hard_macros,
                        'raw_hard_pin_objects': len(raw),
                        'hard_pin_occurrences_in_net_traversal': len(occurrences),
                        'unique_used_hard_pin_names': len({n.get_name() for n in occurrences}),
                        'plc_nets': len(plc.nets), 'old_ownership': describe(old_extractor),
                        'fixed_ownership': describe(smooth_primitives._extract_net_data),
                        'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [directory/'netlist.pb.txt', directory/'initial.plc']}})
    report = {'schema_version': 1, 'classification': 'controlled_topology_extraction_only',
              'control': 'Identical extractor with the single parent-ownership line restored to the competition expression.',
              'limits': 'No optimization rerun, transfer score, or downstream PnR claim.',
              'fixed_function_sha256': hashlib.sha256(fixed_source.encode()).hexdigest(),
              'old_control_function_sha256': hashlib.sha256(old_source.encode()).hexdigest(),
              'fixed_function_source': fixed_source, 'old_control_function_source': old_source,
              'benchmarks': records}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
