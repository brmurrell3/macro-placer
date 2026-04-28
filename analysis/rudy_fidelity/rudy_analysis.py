#!/usr/bin/env python3
"""
RUDY vs PlacementCost congestion grid comparison for ibm01.

Runs DPO placer on ibm01, then compares the differentiable RUDY congestion
grid (used by DPO) against the real PlacementCost congestion grid cell by cell.

Goal: determine whether the RUDY error is (a) a uniform ~2x scaling or
(b) spatially varying / directional, requiring a better model.
"""

import sys
import os
import math
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, _set_placement


def get_dpo_placement(benchmark):
    """Run DPO placer on benchmark and return optimized positions."""
    from submissions.dpo.placer import DPOPlacer
    placer = DPOPlacer(seed=42)
    return placer.place(benchmark)


def compute_rudy_grid(positions, benchmark, plc):
    """Compute RUDY congestion grid using DPO's internal function.

    Returns:
        h_cong: [grid_rows, grid_cols] horizontal congestion
        v_cong: [grid_rows, grid_cols] vertical congestion
        combined: [grid_rows, grid_cols] h + v congestion
    """
    from submissions.dpo.placer import _extract_net_data, _rudy_congestion

    net_data = _extract_net_data(benchmark, plc)

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    cell_w = cw / grid_cols
    cell_h = ch / grid_rows

    cell_x_min = torch.arange(grid_cols, dtype=torch.float32) * cell_w
    cell_x_max = cell_x_min + cell_w
    cell_y_min = torch.arange(grid_rows, dtype=torch.float32) * cell_h
    cell_y_max = cell_y_min + cell_h

    grid_h_routes = cell_h * benchmark.hroutes_per_micron
    grid_v_routes = cell_w * benchmark.vroutes_per_micron

    port_base = torch.zeros(1, 2)
    gamma = 0.01 * cw  # Phase 1 gamma

    # We need to compute h_cong and v_cong separately (the function returns combined)
    # Let's extract the intermediate grids by reimplementing the core logic
    num_nets = len(net_data.weights)
    all_pos = torch.cat([positions.detach(), port_base], dim=0)
    pin_pos = all_pos[net_data.pin_macro_idx] + net_data.pin_offsets

    pin_x = pin_pos[:, :, 0]
    pin_y = pin_pos[:, :, 1]
    mask = net_data.mask
    big = 1e10

    # Smooth bounding boxes
    x_for_max = pin_x.clone(); x_for_max[~mask] = -big
    x_for_min = pin_x.clone(); x_for_min[~mask] = big
    y_for_max = pin_y.clone(); y_for_max[~mask] = -big
    y_for_min = pin_y.clone(); y_for_min[~mask] = big

    bbox_x_max = gamma * torch.logsumexp(x_for_max / gamma, dim=1)
    bbox_x_min = -gamma * torch.logsumexp(-x_for_min / gamma, dim=1)
    bbox_y_max = gamma * torch.logsumexp(y_for_max / gamma, dim=1)
    bbox_y_min = -gamma * torch.logsumexp(-y_for_min / gamma, dim=1)

    bbox_w = (bbox_x_max - bbox_x_min).clamp(min=1e-6)
    bbox_h = (bbox_y_max - bbox_y_min).clamp(min=1e-6)

    w = net_data.weights
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)

    h_cong = torch.zeros(grid_rows, grid_cols)
    v_cong = torch.zeros(grid_rows, grid_cols)
    batch_size = 2000

    for b_start in range(0, num_nets, batch_size):
        b_end = min(b_start + batch_size, num_nets)
        b = slice(b_start, b_end)

        x_ol = torch.clamp(
            torch.min(bbox_x_max[b].unsqueeze(1), cell_x_max.unsqueeze(0))
            - torch.max(bbox_x_min[b].unsqueeze(1), cell_x_min.unsqueeze(0)),
            min=0,
        )
        y_ol = torch.clamp(
            torch.min(bbox_y_max[b].unsqueeze(1), cell_y_max.unsqueeze(0))
            - torch.max(bbox_y_min[b].unsqueeze(1), cell_y_min.unsqueeze(0)),
            min=0,
        )
        ol_area = y_ol.unsqueeze(2) * x_ol.unsqueeze(1)
        h_cong = h_cong + (h_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
        v_cong = v_cong + (v_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)

    combined = h_cong + v_cong
    return h_cong.detach().numpy(), v_cong.detach().numpy(), combined.detach().numpy()


def compute_real_congestion_grid(positions, benchmark, plc):
    """Compute real PlacementCost congestion grid.

    Returns:
        h_cong: [grid_rows, grid_cols] horizontal congestion (normalized)
        v_cong: [grid_rows, grid_cols] vertical congestion (normalized)
        combined: [grid_rows, grid_cols] h + v congestion
    """
    # Set placement in PlacementCost
    _set_placement(plc, positions, benchmark)

    # Force recompute
    plc.FLAG_UPDATE_CONGESTION = True
    plc.get_routing()

    grid_rows = plc.grid_row
    grid_cols = plc.grid_col

    h_cong = np.array(plc.H_routing_cong).reshape(grid_rows, grid_cols)
    v_cong = np.array(plc.V_routing_cong).reshape(grid_rows, grid_cols)
    combined = h_cong + v_cong

    return h_cong, v_cong, combined


def analyze_grids(rudy_grid, real_grid, name="combined"):
    """Compare two congestion grids and report statistics."""
    print(f"\n{'='*70}")
    print(f"  {name.upper()} CONGESTION COMPARISON")
    print(f"{'='*70}")

    # Basic stats
    print(f"\n  Grid shape: {rudy_grid.shape}")
    print(f"  RUDY  - mean: {rudy_grid.mean():.4f}, std: {rudy_grid.std():.4f}, "
          f"max: {rudy_grid.max():.4f}, min: {rudy_grid.min():.4f}")
    print(f"  Real  - mean: {real_grid.mean():.4f}, std: {real_grid.std():.4f}, "
          f"max: {real_grid.max():.4f}, min: {real_grid.min():.4f}")

    # ABU-5% comparison (this is what get_congestion_cost uses)
    rudy_flat = rudy_grid.flatten()
    real_flat = real_grid.flatten()
    k5 = max(1, int(0.05 * len(rudy_flat)))
    k10 = max(1, int(0.10 * len(rudy_flat)))

    rudy_top5 = np.sort(rudy_flat)[::-1][:k5].mean()
    real_top5 = np.sort(real_flat)[::-1][:k5].mean()
    rudy_top10 = np.sort(rudy_flat)[::-1][:k10].mean()
    real_top10 = np.sort(real_flat)[::-1][:k10].mean()

    print(f"\n  ABU-5%:  RUDY={rudy_top5:.4f}  Real={real_top5:.4f}  "
          f"Ratio(Real/RUDY)={real_top5/max(rudy_top5, 1e-10):.3f}")
    print(f"  ABU-10%: RUDY={rudy_top10:.4f}  Real={real_top10:.4f}  "
          f"Ratio(Real/RUDY)={real_top10/max(rudy_top10, 1e-10):.3f}")

    # Per-cell ratio analysis (only where both are non-negligible)
    threshold = 0.01  # skip near-zero cells
    valid = (rudy_grid > threshold) & (real_grid > threshold)
    if valid.sum() > 0:
        ratios = real_grid[valid] / rudy_grid[valid]
        print(f"\n  Per-cell ratio (Real/RUDY) where both > {threshold}:")
        print(f"    N cells:  {valid.sum()}")
        print(f"    Mean:     {ratios.mean():.3f}")
        print(f"    Median:   {np.median(ratios):.3f}")
        print(f"    Std:      {ratios.std():.3f}")
        print(f"    Min:      {ratios.min():.3f}")
        print(f"    Max:      {ratios.max():.3f}")
        print(f"    P10:      {np.percentile(ratios, 10):.3f}")
        print(f"    P25:      {np.percentile(ratios, 25):.3f}")
        print(f"    P75:      {np.percentile(ratios, 75):.3f}")
        print(f"    P90:      {np.percentile(ratios, 90):.3f}")

        # Coefficient of variation of ratios
        cv = ratios.std() / ratios.mean()
        print(f"    CoV:      {cv:.3f}  ({'uniform' if cv < 0.3 else 'SPATIALLY VARYING'})")

    # Where is real >> RUDY? (top disagreements)
    diff = real_grid - rudy_grid
    abs_diff = np.abs(diff)

    print(f"\n  Absolute difference (Real - RUDY):")
    print(f"    Mean: {diff.mean():.4f}")
    print(f"    Std:  {diff.std():.4f}")
    print(f"    Cells where Real > RUDY: {(diff > 0).sum()} / {diff.size}")
    print(f"    Cells where Real < RUDY: {(diff < 0).sum()} / {diff.size}")

    # Correlation
    if rudy_flat.std() > 0 and real_flat.std() > 0:
        corr = np.corrcoef(rudy_flat, real_flat)[0, 1]
        print(f"\n  Pearson correlation: {corr:.4f}")

    # Rank correlation (Spearman) - do they agree on which cells are worst?
    from scipy.stats import spearmanr
    rho, pval = spearmanr(rudy_flat, real_flat)
    print(f"  Spearman rank corr: {rho:.4f} (p={pval:.2e})")

    return {
        'rudy_top5': rudy_top5,
        'real_top5': real_top5,
        'ratio_top5': real_top5 / max(rudy_top5, 1e-10),
        'correlation': np.corrcoef(rudy_flat, real_flat)[0, 1] if rudy_flat.std() > 0 else 0,
        'spearman': rho,
        'diff': diff,
        'valid_ratios': ratios if valid.sum() > 0 else None,
    }


def spatial_analysis(rudy_combined, real_combined, benchmark):
    """Analyze spatial patterns of RUDY error."""
    print(f"\n{'='*70}")
    print(f"  SPATIAL PATTERN ANALYSIS")
    print(f"{'='*70}")

    grid_rows, grid_cols = rudy_combined.shape
    diff = real_combined - rudy_combined

    # Quadrant analysis
    mid_r = grid_rows // 2
    mid_c = grid_cols // 2

    quadrants = {
        'Top-Left':     diff[:mid_r, :mid_c],
        'Top-Right':    diff[:mid_r, mid_c:],
        'Bottom-Left':  diff[mid_r:, :mid_c],
        'Bottom-Right': diff[mid_r:, mid_c:],
    }

    print(f"\n  Quadrant analysis (mean Real-RUDY difference):")
    for name, q in quadrants.items():
        print(f"    {name:15s}: mean={q.mean():+.4f}  max={q.max():.4f}")

    # Row-wise analysis (top-to-bottom gradient?)
    print(f"\n  Row-wise mean difference (top rows = row 0):")
    n_bands = min(5, grid_rows)
    band_size = grid_rows // n_bands
    for i in range(n_bands):
        r_start = i * band_size
        r_end = (i + 1) * band_size if i < n_bands - 1 else grid_rows
        band_diff = diff[r_start:r_end, :].mean()
        band_rudy = rudy_combined[r_start:r_end, :].mean()
        band_real = real_combined[r_start:r_end, :].mean()
        print(f"    Rows {r_start:3d}-{r_end:3d}: diff={band_diff:+.4f}  "
              f"RUDY={band_rudy:.4f}  Real={band_real:.4f}")

    # Column-wise analysis (left-to-right gradient?)
    print(f"\n  Column-wise mean difference (left cols = col 0):")
    n_bands = min(5, grid_cols)
    band_size = grid_cols // n_bands
    for i in range(n_bands):
        c_start = i * band_size
        c_end = (i + 1) * band_size if i < n_bands - 1 else grid_cols
        band_diff = diff[:, c_start:c_end].mean()
        band_rudy = rudy_combined[:, c_start:c_end].mean()
        band_real = real_combined[:, c_start:c_end].mean()
        print(f"    Cols {c_start:3d}-{c_end:3d}: diff={band_diff:+.4f}  "
              f"RUDY={band_rudy:.4f}  Real={band_real:.4f}")

    # Edge vs center analysis
    edge_width = max(1, min(grid_rows, grid_cols) // 5)
    edge_mask = np.zeros_like(diff, dtype=bool)
    edge_mask[:edge_width, :] = True   # top
    edge_mask[-edge_width:, :] = True  # bottom
    edge_mask[:, :edge_width] = True   # left
    edge_mask[:, -edge_width:] = True  # right
    center_mask = ~edge_mask

    print(f"\n  Edge vs Center (edge width = {edge_width} cells):")
    print(f"    Edge   ({edge_mask.sum():4d} cells): "
          f"mean_diff={diff[edge_mask].mean():+.4f}  "
          f"mean_RUDY={rudy_combined[edge_mask].mean():.4f}  "
          f"mean_Real={real_combined[edge_mask].mean():.4f}")
    print(f"    Center ({center_mask.sum():4d} cells): "
          f"mean_diff={diff[center_mask].mean():+.4f}  "
          f"mean_RUDY={rudy_combined[center_mask].mean():.4f}  "
          f"mean_Real={real_combined[center_mask].mean():.4f}")

    # Macro proximity analysis
    positions = benchmark.macro_positions.numpy()
    sizes = benchmark.macro_sizes.numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / grid_cols
    cell_h = ch / grid_rows

    # Compute min distance from each grid cell to any hard macro edge
    n_hard = benchmark.num_hard_macros
    cell_cx = np.arange(grid_cols) * cell_w + cell_w / 2
    cell_cy = np.arange(grid_rows) * cell_h + cell_h / 2

    # Create a macro density map (1 = macro present, 0 = no macro)
    macro_density = np.zeros((grid_rows, grid_cols), dtype=float)
    for i in range(n_hard):
        mx, my = positions[i]
        mw, mh = sizes[i]
        # Grid cells covered by this macro
        c_min = max(0, int((mx - mw/2) / cell_w))
        c_max = min(grid_cols - 1, int((mx + mw/2) / cell_w))
        r_min = max(0, int((my - mh/2) / cell_h))
        r_max = min(grid_rows - 1, int((my + mh/2) / cell_h))
        macro_density[r_min:r_max+1, c_min:c_max+1] = 1.0

    macro_cells = macro_density > 0
    nonmacro_cells = ~macro_cells

    print(f"\n  Macro-occupied vs non-macro cells:")
    print(f"    Macro cells   ({macro_cells.sum():4d}): "
          f"mean_diff={diff[macro_cells].mean():+.4f}  "
          f"mean_RUDY={rudy_combined[macro_cells].mean():.4f}  "
          f"mean_Real={real_combined[macro_cells].mean():.4f}")
    if nonmacro_cells.sum() > 0:
        print(f"    Non-macro     ({nonmacro_cells.sum():4d}): "
              f"mean_diff={diff[nonmacro_cells].mean():+.4f}  "
              f"mean_RUDY={rudy_combined[nonmacro_cells].mean():.4f}  "
              f"mean_Real={real_combined[nonmacro_cells].mean():.4f}")

    # Ratio in macro vs non-macro cells
    threshold = 0.01
    valid_macro = macro_cells & (rudy_combined > threshold) & (real_combined > threshold)
    valid_nonmacro = nonmacro_cells & (rudy_combined > threshold) & (real_combined > threshold)

    if valid_macro.sum() > 0:
        r_macro = real_combined[valid_macro] / rudy_combined[valid_macro]
        print(f"\n  Real/RUDY ratio in macro-occupied cells: "
              f"mean={r_macro.mean():.3f} std={r_macro.std():.3f}")
    if valid_nonmacro.sum() > 0:
        r_nonmacro = real_combined[valid_nonmacro] / rudy_combined[valid_nonmacro]
        print(f"  Real/RUDY ratio in non-macro cells:      "
              f"mean={r_nonmacro.mean():.3f} std={r_nonmacro.std():.3f}")


def top_cells_analysis(rudy_combined, real_combined, benchmark):
    """Compare which cells are in the top-5% for RUDY vs real."""
    print(f"\n{'='*70}")
    print(f"  TOP-CELL AGREEMENT ANALYSIS")
    print(f"{'='*70}")

    n_cells = rudy_combined.size
    k5 = max(1, int(0.05 * n_cells))

    rudy_flat = rudy_combined.flatten()
    real_flat = real_combined.flatten()

    rudy_top5_idx = set(np.argsort(rudy_flat)[-k5:])
    real_top5_idx = set(np.argsort(real_flat)[-k5:])

    overlap = rudy_top5_idx & real_top5_idx
    jaccard = len(overlap) / len(rudy_top5_idx | real_top5_idx)

    print(f"\n  Top-5% cells (k={k5}):")
    print(f"    RUDY top-5% and Real top-5% overlap: {len(overlap)}/{k5} = {len(overlap)/k5*100:.1f}%")
    print(f"    Jaccard similarity: {jaccard:.3f}")

    # What about top-10%?
    k10 = max(1, int(0.10 * n_cells))
    rudy_top10_idx = set(np.argsort(rudy_flat)[-k10:])
    real_top10_idx = set(np.argsort(real_flat)[-k10:])
    overlap10 = rudy_top10_idx & real_top10_idx
    print(f"\n  Top-10% cells (k={k10}):")
    print(f"    RUDY top-10% and Real top-10% overlap: {len(overlap10)}/{k10} = {len(overlap10)/k10*100:.1f}%")

    # Are there cells that are in real top-5% but NOT in RUDY top-10%?
    # These are cells RUDY completely misses
    missed = real_top5_idx - rudy_top10_idx
    print(f"\n  Cells in Real top-5% but NOT in RUDY top-10%: {len(missed)} ({len(missed)/k5*100:.1f}%)")

    # Are there cells in RUDY top-5% that are not even in real top-25%?
    k25 = max(1, int(0.25 * n_cells))
    real_top25_idx = set(np.argsort(real_flat)[-k25:])
    false_alarms = rudy_top5_idx - real_top25_idx
    print(f"  Cells in RUDY top-5% but NOT in Real top-25%: {len(false_alarms)} ({len(false_alarms)/k5*100:.1f}%)")


def hv_decomposition_analysis(rudy_h, rudy_v, real_h, real_v):
    """Compare H vs V congestion separately to find directional bias."""
    print(f"\n{'='*70}")
    print(f"  H vs V DIRECTIONAL ANALYSIS")
    print(f"{'='*70}")

    # ABU-5% for each direction
    k5 = max(1, int(0.05 * rudy_h.size))

    for name, rudy, real in [("Horizontal", rudy_h, real_h), ("Vertical", rudy_v, real_v)]:
        r_flat = rudy.flatten()
        re_flat = real.flatten()
        r_top5 = np.sort(r_flat)[::-1][:k5].mean()
        re_top5 = np.sort(re_flat)[::-1][:k5].mean()
        ratio = re_top5 / max(r_top5, 1e-10)

        valid = (rudy > 0.01) & (real > 0.01)
        if valid.sum() > 0:
            ratios = real[valid] / rudy[valid]
            print(f"\n  {name}:")
            print(f"    ABU-5%: RUDY={r_top5:.4f}  Real={re_top5:.4f}  Ratio={ratio:.3f}")
            print(f"    Per-cell ratio: mean={ratios.mean():.3f}  std={ratios.std():.3f}  "
                  f"median={np.median(ratios):.3f}")

            corr = np.corrcoef(r_flat, re_flat)[0, 1] if r_flat.std() > 0 else 0
            print(f"    Correlation: {corr:.4f}")


def macro_blockage_analysis(rudy_combined, real_combined, benchmark, plc):
    """Analyze the macro blockage component that RUDY misses."""
    print(f"\n{'='*70}")
    print(f"  MACRO BLOCKAGE ANALYSIS")
    print(f"{'='*70}")

    grid_rows = plc.grid_row
    grid_cols = plc.grid_col

    # Extract macro routing congestion separately
    h_macro = np.array(plc.H_macro_routing_cong).reshape(grid_rows, grid_cols)
    v_macro = np.array(plc.V_macro_routing_cong).reshape(grid_rows, grid_cols)
    macro_total = h_macro + v_macro

    # Routing-only (without macro blockage)
    h_routing_only = np.array(plc.H_routing_cong).reshape(grid_rows, grid_cols) - h_macro
    v_routing_only = np.array(plc.V_routing_cong).reshape(grid_rows, grid_cols) - v_macro
    routing_only = h_routing_only + v_routing_only

    print(f"\n  Macro blockage congestion:")
    print(f"    Mean: {macro_total.mean():.4f}")
    print(f"    Max:  {macro_total.max():.4f}")
    print(f"    Non-zero cells: {(macro_total > 0).sum()}/{macro_total.size}")

    # What fraction of total real congestion comes from macro blockage?
    total_real = real_combined.sum()
    total_macro = macro_total.sum()
    total_routing = routing_only.sum()

    print(f"\n  Congestion budget (summed over all cells):")
    print(f"    Total real:     {total_real:.1f}")
    print(f"    Net routing:    {total_routing:.1f} ({total_routing/total_real*100:.1f}%)")
    print(f"    Macro blockage: {total_macro:.1f} ({total_macro/total_real*100:.1f}%)")

    # Compare RUDY (no blockage) to routing-only component of real
    k5 = max(1, int(0.05 * rudy_combined.size))
    rudy_top5 = np.sort(rudy_combined.flatten())[::-1][:k5].mean()
    routing_top5 = np.sort(routing_only.flatten())[::-1][:k5].mean()
    real_top5 = np.sort(real_combined.flatten())[::-1][:k5].mean()

    print(f"\n  ABU-5% decomposition:")
    print(f"    RUDY (no blockage, no smoothing): {rudy_top5:.4f}")
    print(f"    Real routing-only (no blockage):  {routing_top5:.4f}")
    print(f"    Real total (routing + blockage):  {real_top5:.4f}")
    print(f"    Ratio Real-routing / RUDY:        {routing_top5/max(rudy_top5,1e-10):.3f}")
    print(f"    Ratio Real-total / RUDY:          {real_top5/max(rudy_top5,1e-10):.3f}")

    # Correlation: RUDY vs routing-only (stripping out blockage effect)
    rudy_flat = rudy_combined.flatten()
    routing_flat = routing_only.flatten()
    if rudy_flat.std() > 0 and routing_flat.std() > 0:
        corr = np.corrcoef(rudy_flat, routing_flat)[0, 1]
        print(f"\n  Correlation RUDY vs routing-only (no macro blockage): {corr:.4f}")


def smoothing_analysis(rudy_combined, benchmark, plc):
    """Analyze the effect of PlacementCost's smoothing on congestion."""
    print(f"\n{'='*70}")
    print(f"  SMOOTHING ANALYSIS")
    print(f"{'='*70}")

    smooth_range = plc.smooth_range
    hrouting_alloc = plc.hrouting_alloc
    vrouting_alloc = plc.vrouting_alloc

    print(f"\n  PlacementCost parameters:")
    print(f"    smooth_range:   {smooth_range}")
    print(f"    hrouting_alloc: {hrouting_alloc}")
    print(f"    vrouting_alloc: {vrouting_alloc}")
    print(f"    grid_rows:      {plc.grid_row}")
    print(f"    grid_cols:      {plc.grid_col}")
    print(f"    grid_h_routes:  {plc.grid_h_routes:.2f}")
    print(f"    grid_v_routes:  {plc.grid_v_routes:.2f}")


def net_decomposition_analysis(benchmark, plc):
    """Analyze how net decomposition differs between RUDY and real routing."""
    print(f"\n{'='*70}")
    print(f"  NET DECOMPOSITION ANALYSIS")
    print(f"{'='*70}")

    # Count net sizes in PlacementCost
    pin_counts = []
    for driver_name, sink_names in plc.nets.items():
        total_pins = 1 + len(sink_names)
        pin_counts.append(total_pins)

    pin_counts = np.array(pin_counts)

    print(f"\n  Net statistics:")
    print(f"    Total nets: {len(pin_counts)}")
    print(f"    2-pin nets: {(pin_counts == 2).sum()} ({(pin_counts == 2).sum()/len(pin_counts)*100:.1f}%)")
    print(f"    3-pin nets: {(pin_counts == 3).sum()} ({(pin_counts == 3).sum()/len(pin_counts)*100:.1f}%)")
    print(f"    >3-pin nets: {(pin_counts > 3).sum()} ({(pin_counts > 3).sum()/len(pin_counts)*100:.1f}%)")
    print(f"    Max fan-out: {pin_counts.max()}")
    print(f"    Mean pins/net: {pin_counts.mean():.2f}")

    # For >3-pin nets, PlacementCost decomposes into 2-pin nets (star from source)
    # while RUDY uses full bounding box. This overestimates demand for large nets.
    large_nets = pin_counts[pin_counts > 3]
    if len(large_nets) > 0:
        print(f"\n  Large nets (>3 pins):")
        print(f"    Count: {len(large_nets)}")
        print(f"    Mean size: {large_nets.mean():.1f}")
        print(f"    Max size: {large_nets.max()}")
        # RUDY error for a large net: RUDY fills the entire bounding box uniformly
        # Real routing decomposes into 2-pin L-routes from source to each sink
        # The RUDY estimate is proportional to bbox_area, while real routing
        # traces specific L-paths that may not fill the whole bbox
        print(f"    These nets contribute bbox-area-proportional RUDY demand")
        print(f"    but only L-shaped routing demand in PlacementCost")


def main():
    print("=" * 70)
    print("  RUDY vs PlacementCost Congestion Analysis — ibm01")
    print("=" * 70)

    # Load benchmark
    print("\nLoading ibm01...")
    benchmark_dir = os.path.join(project_root, "external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)
    print(f"  {benchmark}")
    print(f"  Grid: {benchmark.grid_rows} x {benchmark.grid_cols}")
    print(f"  Hroutes/um: {benchmark.hroutes_per_micron}, Vroutes/um: {benchmark.vroutes_per_micron}")

    # Run DPO to get optimized positions
    print("\nRunning DPO placer...")
    dpo_positions = get_dpo_placement(benchmark)

    # Compute proxy cost for reference
    print("\nComputing reference proxy cost...")
    result = compute_proxy_cost(dpo_positions, benchmark, plc)
    print(f"  Proxy cost: {result['proxy_cost']:.4f}")
    print(f"  WL: {result['wirelength_cost']:.4f}, Den: {result['density_cost']:.4f}, "
          f"Cong: {result['congestion_cost']:.4f}")
    print(f"  Overlaps: {result['overlap_count']}")

    # Get smoothing/routing allocation parameters
    smoothing_analysis(None, benchmark, plc)

    # Compute RUDY grid
    print("\nComputing RUDY congestion grid...")
    rudy_h, rudy_v, rudy_combined = compute_rudy_grid(dpo_positions, benchmark, plc)

    # Compute real congestion grid
    print("Computing real PlacementCost congestion grid...")
    # Need to reload plc to get fresh state (get_routing was already called by compute_proxy_cost)
    benchmark2, plc2 = load_benchmark_from_dir(benchmark_dir)
    real_h, real_v, real_combined = compute_real_congestion_grid(dpo_positions, benchmark2, plc2)

    # Analysis
    stats_combined = analyze_grids(rudy_combined, real_combined, "Combined (H+V)")
    stats_h = analyze_grids(rudy_h, real_h, "Horizontal only")
    stats_v = analyze_grids(rudy_v, real_v, "Vertical only")

    hv_decomposition_analysis(rudy_h, rudy_v, real_h, real_v)
    spatial_analysis(rudy_combined, real_combined, benchmark)
    top_cells_analysis(rudy_combined, real_combined, benchmark)
    macro_blockage_analysis(rudy_combined, real_combined, benchmark, plc2)
    net_decomposition_analysis(benchmark, plc)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    ratio = stats_combined['ratio_top5']
    corr = stats_combined['correlation']
    spearman = stats_combined['spearman']

    print(f"\n  Key findings:")
    print(f"    1. ABU-5% ratio (Real/RUDY): {ratio:.3f}")
    print(f"    2. Pearson correlation:       {corr:.4f}")
    print(f"    3. Spearman rank correlation: {spearman:.4f}")

    if stats_combined['valid_ratios'] is not None:
        cv = stats_combined['valid_ratios'].std() / stats_combined['valid_ratios'].mean()
        print(f"    4. Ratio CoV:                {cv:.3f}")

        if cv < 0.3:
            print(f"\n  CONCLUSION: RUDY error is APPROXIMATELY UNIFORM (~{ratio:.1f}x)")
            print(f"  -> A simple weight scaling could partially compensate")
        else:
            print(f"\n  CONCLUSION: RUDY error is SPATIALLY VARYING (CoV={cv:.3f})")
            print(f"  -> Weight scaling alone cannot fix this")
            print(f"  -> Need a structurally better congestion model")

    print()


if __name__ == "__main__":
    main()
