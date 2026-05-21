"""E115 — compare FastDiffProxy vs DiffProxyV3 baseline.

Usage:
  .venv/bin/python experiments/E115_triton_kernels/code/bench_fast.py [bench] [device] [n_steps]

Reports:
  - per-step wall (baseline vs fast)
  - per-component fwd timings (baseline vs fast)
  - per-component fwd+bwd timings
  - cost equivalence (within 0.5%)
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    str(_HERE),
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps, sdf_init

from smooth_global_placer_v3 import DiffProxyV3, loss_with_penalty_v3
from fast_proxy import FastDiffProxy, fast_loss_with_penalty


def time_step_loop(proxy, loss_fn, positions, fixed_mask, n_steps, device):
    """Time n_steps of Adam update on the proxy. Returns ms/step."""
    optimizer = torch.optim.Adam([positions], lr=10.0)
    # warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = loss_fn(proxy, positions, 50.0, include_congestion=True, boundary_lambda=50.0)
        loss.backward()
        with torch.no_grad():
            positions.grad[fixed_mask] = 0
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss, _ = loss_fn(proxy, positions, 50.0, include_congestion=True, boundary_lambda=50.0)
        loss.backward()
        with torch.no_grad():
            positions.grad[fixed_mask] = 0
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_steps


def time_component_fwd_bwd(name, fn, positions, n=20, device="cuda"):
    # warmup
    for _ in range(3):
        positions.grad = None
        v = fn(positions)
        if v.requires_grad:
            v.backward()
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        positions.grad = None
        v = fn(positions)
        if v.requires_grad:
            v.backward()
    if device.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def main(bench_name: str = "ibm17", device: str = "cuda", n_steps: int = 30):
    print(f"\n=== E115 fast vs baseline: bench={bench_name} device={device} steps={n_steps} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"  num_macros={benchmark.num_macros} num_hard={benchmark.num_hard_macros} "
          f"grid={benchmark.grid_rows}x{benchmark.grid_cols}", flush=True)

    init_pos = sdf_init(benchmark)
    init_pos, _ = project_overlaps(init_pos, benchmark)

    # ---------------- Baseline ----------------
    print("\n--- BASELINE (DiffProxyV3) ---", flush=True)
    proxy_base = DiffProxyV3(benchmark, plc, device=device, gamma_frac=5e-4)
    positions_b = init_pos.clone().detach().to(device).requires_grad_(True)
    fixed_mask = benchmark.macro_fixed.bool().to(device)
    ms_base = time_step_loop(proxy_base, loss_with_penalty_v3, positions_b, fixed_mask, n_steps, device)
    print(f"  per-step: {ms_base:.2f} ms", flush=True)

    # ---------------- Fast ----------------
    print("\n--- FAST (FastDiffProxy) ---", flush=True)
    proxy_fast = FastDiffProxy(benchmark, plc, device=device, gamma_frac=5e-4)
    positions_f = init_pos.clone().detach().to(device).requires_grad_(True)
    ms_fast = time_step_loop(proxy_fast, fast_loss_with_penalty, positions_f, fixed_mask, n_steps, device)
    print(f"  per-step: {ms_fast:.2f} ms", flush=True)
    print(f"  SPEEDUP: {ms_base / ms_fast:.2f}x  (saved {ms_base - ms_fast:.1f} ms/step)", flush=True)

    # ---------------- Per-component fwd+bwd ----------------
    print("\n--- Per-component fwd+bwd (cuda) ---", flush=True)
    # Reset positions
    positions = init_pos.clone().detach().to(device).requires_grad_(True)

    def make_baseline_fns(proxy):
        from diff_proxy import _lse_hpwl, _grid_density
        return {
            "wl": lambda p: _lse_hpwl(proxy._clamp_to_canvas(p), proxy.net_data, proxy.port_base, proxy.gamma) / proxy.wl_norm,
            "density": lambda p: _grid_density(proxy._clamp_to_canvas(p), proxy.sizes, proxy.cell_x_min, proxy.cell_x_max, proxy.cell_y_min, proxy.cell_y_max, proxy.cell_area, proxy.grid_rows, proxy.grid_cols),
            "cong": lambda p: proxy.trace.compute_congestion(proxy._clamp_to_canvas(p)),
            "overlap": lambda p: proxy.overlap_penalty(p),
            "boundary": lambda p: proxy.out_of_canvas_penalty(p),
        }

    def make_fast_fns(proxy):
        from fast_proxy import fast_lse_hpwl, fast_grid_density
        return {
            "wl": lambda p: fast_lse_hpwl(proxy._clamp_to_canvas(p), proxy.net_data, proxy.port_base, proxy.gamma) / proxy.wl_norm,
            "density": lambda p: fast_grid_density(proxy._clamp_to_canvas(p), proxy.sizes, proxy.cell_x_min, proxy.cell_x_max, proxy.cell_y_min, proxy.cell_y_max, proxy.cell_area, proxy.grid_rows, proxy.grid_cols),
            "cong": lambda p: proxy.trace.compute_congestion(proxy._clamp_to_canvas(p)),
            "overlap": lambda p: proxy.overlap_penalty(p),
            "boundary": lambda p: proxy.out_of_canvas_penalty(p),
        }

    base_fns = make_baseline_fns(proxy_base)
    fast_fns = make_fast_fns(proxy_fast)

    print(f"  {'comp':<10s} {'baseline ms':>15s} {'fast ms':>15s} {'speedup':>10s}", flush=True)
    component_results = {}
    for name in ("wl", "density", "cong", "overlap", "boundary"):
        ms_b = time_component_fwd_bwd(name, base_fns[name], positions.clone().detach().to(device).requires_grad_(True), n=15, device=device)
        ms_f = time_component_fwd_bwd(name, fast_fns[name], positions.clone().detach().to(device).requires_grad_(True), n=15, device=device)
        print(f"  {name:<10s} {ms_b:>15.2f} {ms_f:>15.2f} {ms_b / max(ms_f, 1e-9):>9.2f}x", flush=True)
        component_results[name] = {"baseline_ms": ms_b, "fast_ms": ms_f, "speedup": ms_b / max(ms_f, 1e-9)}

    # ---------------- Numerical equivalence ----------------
    print("\n--- Numerical equivalence ---", flush=True)
    pos_test = init_pos.clone().detach().to(device)
    proxy_base.set_gamma_frac(5e-4) if hasattr(proxy_base, "set_gamma_frac") else None
    proxy_fast.set_gamma_frac(5e-4)
    with torch.no_grad():
        b_cost, b_parts = proxy_base.cost(pos_test, include_congestion=True)
        f_cost, f_parts = proxy_fast.cost(pos_test, include_congestion=True)
    print(f"  baseline cost={b_cost.item():.5f}  parts: wl={b_parts['wl'].item():.5f} d={b_parts['density'].item():.5f} c={b_parts['cong'].item():.5f}", flush=True)
    print(f"  fast     cost={f_cost.item():.5f}  parts: wl={f_parts['wl'].item():.5f} d={f_parts['density'].item():.5f} c={f_parts['cong'].item():.5f}", flush=True)
    rel = abs(f_cost.item() - b_cost.item()) / max(abs(b_cost.item()), 1e-9) * 100
    print(f"  relative cost diff: {rel:+.4f}%", flush=True)

    results = {
        "bench": bench_name,
        "device": device,
        "n_steps": n_steps,
        "num_macros": int(benchmark.num_macros),
        "num_hard": int(benchmark.num_hard_macros),
        "grid_rows": int(benchmark.grid_rows),
        "grid_cols": int(benchmark.grid_cols),
        "baseline_step_ms": ms_base,
        "fast_step_ms": ms_fast,
        "step_speedup": ms_base / ms_fast,
        "components": component_results,
        "numerical_equivalence": {
            "baseline_cost": b_cost.item(),
            "fast_cost": f_cost.item(),
            "rel_diff_pct": rel,
        },
    }
    return results


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    r = main(bench, device, n)
    out_path = _HERE.parent / "results" / f"fast_vs_base_{bench}_{device}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\n  wrote {out_path}", flush=True)
