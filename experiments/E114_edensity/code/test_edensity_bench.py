"""Sanity test of ElectrostaticDensity on a real benchmark (ibm01).

Loads ibm01, creates the eDensity module, evaluates at SDF init.
Checks:
  - density is non-NaN, non-negative
  - gradient computes and is non-zero
  - magnitude is roughly comparable to grid_density
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import sdf_init, project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from diff_proxy import _grid_density
from edensity import ElectrostaticDensity


def main(bench_name: str = "ibm17"):
    print(f"=== Testing eDensity on {bench_name} ===")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"  canvas: {benchmark.canvas_width:.0f} x {benchmark.canvas_height:.0f}")
    print(f"  grid: {benchmark.grid_rows} x {benchmark.grid_cols}")
    print(f"  num_macros: {benchmark.num_macros} (hard={benchmark.num_hard_macros})")

    pos = sdf_init(benchmark)
    pos, _ = project_overlaps(pos, benchmark)
    pos = pos.detach().requires_grad_(True)

    # Canonical proxy
    canon = compute_proxy_cost(pos.detach(), benchmark, plc)
    print(f"\n  CANONICAL proxy={canon['proxy_cost']:.5f}")
    print(f"    wl={canon['wirelength_cost']:.4f}  d={canon['density_cost']:.4f}  c={canon['congestion_cost']:.4f}")

    # Grid-bin density (reference)
    gr, gc = benchmark.grid_rows, benchmark.grid_cols
    cell_w = float(benchmark.canvas_width) / gc
    cell_h = float(benchmark.canvas_height) / gr
    cell_x_min = torch.arange(gc, dtype=torch.float32) * cell_w
    cell_x_max = cell_x_min + cell_w
    cell_y_min = torch.arange(gr, dtype=torch.float32) * cell_h
    cell_y_max = cell_y_min + cell_h
    t0 = time.time()
    d_grid = _grid_density(
        pos, benchmark.macro_sizes,
        cell_x_min, cell_x_max, cell_y_min, cell_y_max,
        cell_w * cell_h, gr, gc,
    )
    t_grid = time.time() - t0
    print(f"\n  GRID density: {float(d_grid):.5f}  ({t_grid*1000:.0f} ms)")

    # eDensity (default config)
    ed = ElectrostaticDensity(benchmark, device="cpu")
    t0 = time.time()
    d_e = ed.compute_density(pos)
    t_e = time.time() - t0
    print(f"\n  eDENSITY (target=1.0, sigma_frac=0.5, scale=1.0):")
    print(f"    cost={float(d_e):.5f}  ({t_e*1000:.0f} ms)")

    # Diagnostic
    parts = ed.compute_density_with_parts(pos)
    print(f"    rho_max={parts['rho_max']:.3f}  rho_mean={parts['rho_mean']:.3f}")
    print(f"    chi_max={parts['chi_max']:.3f}  chi_sum={parts['chi_sum']:.3f}")
    print(f"    phi_max={parts['phi_max']:.3e}  phi_min={parts['phi_min']:.3e}")

    # Test gradient (should be non-zero)
    pos_t = pos.detach().clone().requires_grad_(True)
    d_e = ed.compute_density(pos_t)
    d_e.backward()
    grad_norm = pos_t.grad.norm()
    grad_max = pos_t.grad.abs().max()
    print(f"    grad_norm={grad_norm:.3e}  grad_max={grad_max:.3e}")

    # Try lower target (more overflow → more cost)
    ed2 = ElectrostaticDensity(benchmark, device="cpu", target_density=0.5)
    d_e2 = ed2.compute_density(pos.detach())
    print(f"\n  eDENSITY (target=0.5): cost={float(d_e2):.5f}")

    ed3 = ElectrostaticDensity(benchmark, device="cpu", target_density=0.0)
    d_e3 = ed3.compute_density(pos.detach())
    print(f"  eDENSITY (target=0.0): cost={float(d_e3):.5f}")


if __name__ == "__main__":
    import sys as _s
    name = _s.argv[1] if len(_s.argv) > 1 else "ibm17"
    main(name)
