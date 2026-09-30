#!/usr/bin/env python3
"""Separate M4 follow-up: same frozen E74 endpoint, precision and subspace probes.

Run with .venv/bin/python scripts/diagnose_publication_curvature_followup.py.
No optimization, source patching, or rewriting of the original evidence occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import diagnose_publication_curvature as original

EPSILONS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


class BranchTrace(TorchDispatchMode):
    """Observe unmodified forward operations; retain discrete active-set states."""

    def __init__(self):
        super().__init__()
        self.branches = []
        self.float_outputs = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        for item in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                self.float_outputs[str(item.dtype)] += 1
        name = str(func)
        state = None
        if name in ('aten.minimum.default', 'aten.maximum.default'):
            a, b = args[:2]
            state = (a > b).to(torch.int8) - (a < b).to(torch.int8)
        elif name.startswith('aten.clamp.'):
            a = args[0]
            lo = args[1] if len(args) > 1 else kwargs.get('min')
            hi = args[2] if len(args) > 2 else kwargs.get('max')
            state = torch.zeros_like(a, dtype=torch.int8)
            if lo is not None:
                state = state - 2 * (a < lo).to(torch.int8) - (a == lo).to(torch.int8)
            if hi is not None:
                state = state + 2 * (a > hi).to(torch.int8) + (a == hi).to(torch.int8)
        elif name == 'aten.topk.default':
            state = out[1].sort().values
        if state is not None:
            self.branches.append((name, state.detach().clone()))
        return out


def trace(proxy, x):
    tracer = BranchTrace()
    with torch.no_grad(), tracer:
        value = proxy.cost(x)
    return tracer, str(value.dtype)


def branch_delta(left, right):
    assert len(left.branches) == len(right.branches)
    result = Counter()
    for i, ((name, a), (name2, b)) in enumerate(zip(left.branches, right.branches)):
        assert name == name2 and a.shape == b.shape
        group = 'canvas_clamp' if i < 2 else name
        if name == 'aten.topk.default':
            result[group + ':set_symmetric_difference'] += int(torch.isin(a, b, invert=True).sum() + torch.isin(b, a, invert=True).sum())
        else:
            result[group + ':changed_comparisons'] += int((a != b).sum())
    return dict(result)


def bounds(proxy):
    low = proxy.half_sizes
    high = torch.stack((proxy.cw - low[:, 0], proxy.ch - low[:, 1]), 1)
    return low, high


def promote(proxy):
    changed = {}
    for prefix, obj in (('proxy', proxy), ('net_data', proxy.net_data)):
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                changed[prefix + '.' + name] = {'source_dtype': str(value.dtype), 'source_sha256': original.tensor_hash(value)}
                setattr(obj, name, value.to(torch.float64))
                changed[prefix + '.' + name]['promoted_sha256'] = original.tensor_hash(getattr(obj, name))
    return changed


def norm(x):
    return float(torch.linalg.vector_norm(x))


class WeightedComponent:
    """Diagnostic decomposition using unchanged frozen cost primitives."""

    def __init__(self, proxy, module, component):
        self.proxy, self.module, self.component = proxy, module, component

    def __getattr__(self, name):
        return getattr(self.proxy, name)

    def cost(self, positions):
        p, m = self.proxy, self.module
        clamped = torch.stack([
            positions[:, 0].clamp(p.half_sizes[:, 0], p.cw - p.half_sizes[:, 0]),
            positions[:, 1].clamp(p.half_sizes[:, 1], p.ch - p.half_sizes[:, 1]),
        ], dim=1)
        if self.component == 'wirelength':
            return m._lse_hpwl(clamped, p.net_data, p.port_base, p.gamma) / p.wl_norm
        if self.component == 'weighted_density':
            return 0.5 * m._grid_density(clamped, p.sizes, p.cell_x_min, p.cell_x_max, p.cell_y_min, p.cell_y_max, p.cell_area, p.grid_rows, p.grid_cols)
        return 0.5 * m._rudy_congestion(clamped, p.net_data, p.port_base, p.gamma, p.cell_x_min, p.cell_x_max, p.cell_y_min, p.cell_y_max, p.grid_h_routes, p.grid_v_routes, p.grid_rows, p.grid_cols)


def measure(proxy, module, x, v, mask, movable, epsilons):
    base, g, _ = original.value_and_gradient(proxy, x)
    hv = module.hessian_vector_product(proxy, x, v)
    bt, value_dtype = trace(proxy, x)
    low, high = bounds(proxy)
    rows = []
    for eps in epsilons:
        plus, minus = x + eps * v, x - eps * v
        fp, gp, _ = original.value_and_gradient(proxy, plus)
        fm, gm, _ = original.value_and_gradient(proxy, minus)
        pt, _ = trace(proxy, plus)
        mt, _ = trace(proxy, minus)
        jump = gp - gm
        fd = jump / (2 * eps)
        rows.append({
            'epsilon_microns': eps,
            'cost_plus': fp, 'cost_base': base, 'cost_minus': fm,
            'forward_slope': (fp - base) / eps,
            'backward_slope': (base - fm) / eps,
            'central_slope': (fp - fm) / (2 * eps),
            'gradient_plus_dot_v': float((gp * v).sum()),
            'gradient_minus_dot_v': float((gm * v).sum()),
            'gradient_jump_l2_movable': norm(jump[movable]),
            'gradient_jump_l2_effective': norm(jump[mask]),
            'gradient_jump_l2_excluded_movable': norm(jump[movable & ~mask]),
            'gradient_fd_hvp_l2_effective': norm(fd[mask]),
            'gradient_fd_vHv': float((fd * v).sum()),
            'hvp_relative_error_effective': norm((fd - hv)[mask]) / norm(hv[mask]),
            'hvp_relative_error_full_movable': norm((fd - hv)[movable]) / norm(hv[movable]),
            'actual_plus_displacement_l2_microns': norm(plus - x),
            'actual_minus_displacement_l2_microns': norm(x - minus),
            'plus_clamp_counts': original.clamp_counts(proxy, plus),
            'minus_clamp_counts': original.clamp_counts(proxy, minus),
            'plus_minus_branch_delta': branch_delta(pt, mt),
            'base_plus_branch_delta': branch_delta(bt, pt),
            'base_minus_branch_delta': branch_delta(bt, mt),
        })
    return {
        'state_sha256': original.tensor_hash(x), 'direction_sha256': original.tensor_hash(v),
        'dtype': str(x.dtype), 'value_dtype': value_dtype, 'gradient_dtype': str(g.dtype),
        'hvp_dtype': str(hv.dtype), 'forward_float_operation_output_counts': dict(bt.float_outputs),
        'direction_l2': norm(v), 'effective_mask_sha256': original.tensor_hash(mask),
        'effective_coordinate_count': int(mask.sum()),
        'minimum_effective_canvas_clearance_microns': float(torch.minimum(x - low, high - x)[mask].min()),
        'baseline_clamp_counts': original.clamp_counts(proxy, x), 'base_value': base,
        'baseline_gradient_l2_movable': norm(g[movable]),
        'autodiff_slope': float((g * v).sum()), 'autodiff_vHv': float((hv * v).sum()),
        'autodiff_hvp_l2_effective': norm(hv[mask]), 'scales': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=original.REPO / 'writeup/data/publication/curvature_followup_ibm01.json')
    args = parser.parse_args()
    if args.output.resolve() == original.OUTPUT.resolve():
        raise ValueError('Refusing to overwrite original evidence')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    old = json.loads(original.OUTPUT.read_text())
    immutable = {str(p.relative_to(original.REPO)): original.sha256_file(p) for p in (original.OUTPUT, original.ARCHIVE, original.STATE_PT, Path(original.__file__))}
    report = {
        'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'New local M4 follow-up at unchanged saved final E74 IBM01 endpoint; no optimization, legalization, eigensolve, runtime changes or paper edits.',
        'units': {'cost': 'dimensionless surrogate', 'positions_epsilon': 'microns', 'direction': 'dimensionless, original float32 Euclidean unit directions', 'slope_gradient': 'surrogate per micron', 'curvature_hvp': 'surrogate per micron squared'},
        'immutable_input_sha256': immutable,
        'script_sha256': original.sha256_file(Path(__file__)),
        'precision_policy': 'Construct original float32 proxy once per variant. Float64 promotes every floating proxy/net-data tensor without regenerating constants and promotes exact saved float32 coordinates and directions without renormalization; set default dtype=float64 for temporary tensor factories. Python scalar canvas minus half-size is recomputed in float64, so clamp limits can change. No frozen function code is modified. Forward operation output dtypes are audited.',
        'projection_policy': 'Effective HVP means P H P and finite differences of P grad f, where P selects the reported direction mask. Full movable outputs also retained; derivatives in excluded boundary coordinates need not become continuous merely because perturbations are interior.',
        'branch_trace_policy': 'Trace frozen forward aten min/max comparisons (including ties), clamp states (strict exterior, equality, interior), and top-k membership. First two clamp calls are the canvas x/y clamps. Remaining changes establish other piecewise branch crossings but do not alone quantify their individual gradient contribution.',
        'experiment_sequence': 'Initial experiment included all 12 configurations: two original directions, full/strict/guarded masks, float32/float64, nine fixed epsilon values. All 12 are retained. After observing residual guarded-float64 jumps, an additive diagnostic decomposed full and guarded float64 directions into all three weighted frozen components at epsilon 1e-7, 1e-5, 1e-3, 1e-1. No configurations were dropped or directions selected based on favorable agreement.',
        'original_artifact_derived': [], 'runs': [], 'weighted_component_runs': [],
    }
    for d in old['random_directions']:
        report['original_artifact_derived'].append({'seed': d['seed'], 'autodiff_slope': d['autodiff_directional_gradient'], 'scales': [{'epsilon_microns': s['epsilon_microns_l2'], 'forward_slope': (s['cost_plus'] - s['cost_base']) / s['epsilon_microns_l2'], 'backward_slope': (s['cost_base'] - s['cost_minus']) / s['epsilon_microns_l2'], 'gradient_jump_l2_movable': 2 * s['epsilon_microns_l2'] * s['hvp_fd_l2_norm_movable'], 'plus_clamp_counts': s['plus_clamp_counts'], 'minus_clamp_counts': s['minus_clamp_counts']} for s in d['scales']]})
    with tempfile.TemporaryDirectory(prefix='curvature-followup-frozen-') as tmp:
        root = Path(tmp)
        original.extract_frozen_source(root)
        module, benchmark, plc, benchmark_path = original.load_frozen_modules(root)
        state = torch.load(original.STATE_PT, map_location='cpu', weights_only=False)['placement'].detach().cpu().contiguous()
        assert original.tensor_hash(state) == old['provenance']['placement_tensor_sha256']
        proxy32 = module.SmoothProxy(benchmark, plc)
        proxy64 = module.SmoothProxy(benchmark, plc)
        report['promoted_tensors'] = promote(proxy64)
        report['frozen_source_sha256'] = {name: original.sha256_file(root / name) for name in (original.E74_MEMBER, original.DPO_MEMBER)}
        report['benchmark_input_sha256'] = {name: original.sha256_file(benchmark_path / name) for name in ('netlist.pb.txt', 'initial.plc')}
        for key, actual in report['benchmark_input_sha256'].items():
            assert actual == old['provenance']['benchmark_input_sha256'][key]
        movable = (~benchmark.macro_fixed).unsqueeze(1).expand_as(state)
        low32, high32 = bounds(proxy32)
        low64, high64 = bounds(proxy64)
        clearance = torch.minimum(torch.minimum(state - low32, high32 - state).double(), torch.minimum(state.double() - low64, high64 - state.double()))
        masks = {'full': movable, 'strict_interior': movable & (clearance > 0), 'guarded_interior': movable & (clearance > max(EPSILONS))}
        report['masks'] = {name: {'included_coordinate_count': int(mask.sum()), 'excluded_movable_flat_coordinate_indices': (movable & ~mask).flatten().nonzero().flatten().tolist(), 'sha256': original.tensor_hash(mask)} for name, mask in masks.items()}
        report['mask_policy'] = 'Common masks computed as intersection of original float32 and promoted float64 canvas interiors. Strict excludes clearance <= 0; guarded requires clearance > 0.1 microns in both, guaranteeing no moving coordinate can cross a canvas clamp for epsilon <= 0.1 and norm(v) approximately 1. All masks are coordinate-wise, not macro-wise.'
        report['bounds_precision_delta'] = {'maximum_upper_bound_abs_change_microns': float((high32.double() - high64).abs().max()), 'changed_upper_bound_coordinates': int((high32.double() != high64).sum()), 'original_dtype': str(state.dtype), 'state_shape': list(state.shape)}
        for offset in range(2):
            seed = original.SEED + offset
            raw = np.random.default_rng(seed).standard_normal(int(movable.sum()))
            raw /= np.linalg.norm(raw)
            direction = torch.zeros_like(state)
            direction[movable] = torch.from_numpy(raw).to(state.dtype)
            direction /= torch.linalg.vector_norm(direction)
            assert original.tensor_hash(direction) == old['random_directions'][offset]['direction_sha256']
            for mask_name, mask in masks.items():
                v32 = direction.clone()
                v32[~mask] = 0
                if mask_name != 'full':
                    v32 /= torch.linalg.vector_norm(v32)
                for dtype, proxy in ((torch.float32, proxy32), (torch.float64, proxy64)):
                    torch.set_default_dtype(dtype)
                    result = measure(proxy, module, state.to(dtype), v32.to(dtype), mask, movable, EPSILONS)
                    if dtype == torch.float64:
                        assert set(result['forward_float_operation_output_counts']) == {'torch.float64'}
                    result.update({'seed': seed, 'mask_name': mask_name, 'source_direction_sha256': original.tensor_hash(direction)})
                    if mask_name == 'full' and dtype == torch.float32:
                        prior = old['random_directions'][offset]
                        assert result['autodiff_slope'] == prior['autodiff_directional_gradient']
                        assert result['autodiff_vHv'] == prior['autodiff_directional_curvature_vHv']
                        for previous in prior['scales']:
                            current = next(s for s in result['scales'] if s['epsilon_microns'] == previous['epsilon_microns_l2'])
                            assert current['cost_plus'] == previous['cost_plus']
                            assert current['cost_minus'] == previous['cost_minus']
                        result['original_cost_slope_directional_hvp_exact_reproduction'] = True
                    report['runs'].append(result)
                    print(json.dumps({'seed': seed, 'mask': mask_name, 'dtype': str(dtype), 'elapsed_seconds': time.perf_counter() - started}), flush=True)
                if mask_name in ('full', 'guarded_interior'):
                    torch.set_default_dtype(torch.float64)
                    component_results = []
                    for component in ('wirelength', 'weighted_density', 'weighted_congestion'):
                        wrapper = WeightedComponent(proxy64, module, component)
                        result = measure(wrapper, module, state.double(), v32.double(), mask, movable, (1e-7, 1e-5, 1e-3, 1e-1))
                        result.update({'seed': seed, 'mask_name': mask_name, 'component': component})
                        report['weighted_component_runs'].append(result)
                        component_results.append(result)
                    whole = report['runs'][-1]
                    assert abs(sum(r['base_value'] for r in component_results) - whole['base_value']) < 1e-14
                    assert abs(sum(r['autodiff_slope'] for r in component_results) - whole['autodiff_slope']) < 1e-14
                    assert abs(sum(r['autodiff_vHv'] for r in component_results) - whole['autodiff_vHv']) < 1e-14
        report['component_policy'] = 'Float64 full and guarded directions additionally evaluate each original frozen primitive with the original canvas clamp and original composite weight. Component cost, directional gradient, and directional HVP sums are asserted to reproduce the unmodified frozen composite to absolute 1e-14. These diagnostic wrappers do not modify the frozen source.'
    torch.set_default_dtype(torch.float32)
    for name, digest in immutable.items():
        assert original.sha256_file(original.REPO / name) == digest
    report['environment'] = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(), 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads()}
    report['total_wall_seconds'] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
