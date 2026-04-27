"""
DPO Ablation base — configurable variant of DPOPlacer for ablation experiments.
Not a standalone placer; imported by ablation_*.py scripts.
"""

import sys
import importlib.util
import torch
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass

from macro_place.benchmark import Benchmark

# Import components from main DPO placer via importlib (not a proper package)
_dpo_path = Path(__file__).parent / "placer.py"
_spec = importlib.util.spec_from_file_location("dpo_placer", str(_dpo_path))
_dpo_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo_mod)

_load_plc = _dpo_mod._load_plc
_extract_net_data = _dpo_mod._extract_net_data
_legalize = _dpo_mod._legalize
_count_overlaps = _dpo_mod._count_overlaps
_lse_hpwl = _dpo_mod._lse_hpwl
_grid_density = _dpo_mod._grid_density
_rudy_congestion = _dpo_mod._rudy_congestion
_overlap_penalty = _dpo_mod._overlap_penalty
NetData = _dpo_mod.NetData


class AblationDPOPlacer:
    """DPO with configurable ablation flags."""

    def __init__(self, seed=42, use_sdf_init=True,
                 skip_congestion=False, skip_density=False,
                 max_phases=3,
                 congestion_weight=0.5, density_weight=0.5):
        self.seed = seed
        self.use_sdf_init = use_sdf_init
        self.skip_congestion = skip_congestion
        self.skip_density = skip_density
        self.max_phases = max_phases
        self.congestion_weight = congestion_weight
        self.density_weight = density_weight

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        net_data = _extract_net_data(benchmark, plc)

        if self.use_sdf_init:
            init_pos = self._sdf_init(benchmark)
        else:
            init_pos = self._random_init(benchmark)

        optimized = self._optimize(init_pos, benchmark, net_data)
        result = self._do_legalize(optimized, benchmark)
        return result

    def _sdf_init(self, benchmark):
        import importlib.util
        sdf_path = Path(__file__).parent.parent / "polyhedra" / "init" / "sdf.py"
        spec = importlib.util.spec_from_file_location("sdf", str(sdf_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SDFPlacer(seed=self.seed).place(benchmark)

    def _random_init(self, benchmark):
        """Random uniform initialization within canvas bounds."""
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        n = benchmark.num_macros

        pos = torch.zeros(n, 2)
        movable = benchmark.get_movable_mask()
        fixed = ~movable

        # Fixed macros stay fixed
        pos[fixed] = benchmark.macro_positions[fixed]

        # Random positions for movable macros
        for i in range(n):
            if movable[i]:
                pos[i, 0] = half_w[i] + torch.rand(1).item() * (cw - sizes[i, 0])
                pos[i, 1] = half_h[i] + torch.rand(1).item() * (ch - sizes[i, 1])

        return pos

    def _optimize(self, init_pos, benchmark, net_data):
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes
        half_sizes = sizes / 2
        movable = benchmark.get_movable_mask()
        fixed_mask = ~movable

        wl_norm = (cw + ch) * net_data.total_net_count

        grid_rows = benchmark.grid_rows
        grid_cols = benchmark.grid_cols
        cell_w = cw / grid_cols
        cell_h = ch / grid_rows
        cell_area = cell_w * cell_h
        cell_x_min = torch.arange(grid_cols, dtype=torch.float32) * cell_w
        cell_x_max = cell_x_min + cell_w
        cell_y_min = torch.arange(grid_rows, dtype=torch.float32) * cell_h
        cell_y_max = cell_y_min + cell_h

        grid_h_routes = cell_h * benchmark.hroutes_per_micron
        grid_v_routes = cell_w * benchmark.vroutes_per_micron

        port_base = torch.zeros(1, 2)

        n_movable = int(movable.sum().item())
        num_nets = len(net_data.weights)
        cached_cong = 0.0

        cong_grad_freq = 1 if num_nets < 3000 else (3 if num_nets < 8000 else 5)

        pos = init_pos.clone().detach().requires_grad_(True)
        complexity = n_movable * num_nets * grid_rows * grid_cols
        if complexity > 1e9:
            step_scale = 0.25
        elif complexity > 5e8:
            step_scale = 0.45
        elif complexity > 1e8:
            step_scale = 0.65
        else:
            step_scale = 1.3

        def s(n): return max(40, int(n * step_scale))

        all_phases = [
            (s(250), 0.01,   1.0,   0.5),
            (s(300), 0.002,  50.0,  0.2),
            (s(150), 0.0005, 500.0, 0.1),
        ]
        phases = all_phases[:self.max_phases]

        best_pos = init_pos.clone()
        best_proxy = float("inf")

        for phase_idx, (n_steps, gamma_frac, lam, lr) in enumerate(phases):
            gamma = gamma_frac * cw

            optimizer = torch.optim.Adam([pos], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, n_steps, eta_min=lr * 0.05
            )

            for step in range(n_steps):
                optimizer.zero_grad()

                with torch.no_grad():
                    pos.data[fixed_mask] = init_pos[fixed_mask]

                clamped = torch.stack([
                    pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                    pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
                ], dim=1)

                wl = _lse_hpwl(clamped, net_data, port_base, gamma) / wl_norm

                if self.skip_density:
                    density = torch.tensor(0.0)
                else:
                    density = _grid_density(clamped, sizes,
                                            cell_x_min, cell_x_max,
                                            cell_y_min, cell_y_max,
                                            cell_area, grid_rows, grid_cols)

                if self.skip_congestion:
                    congestion = torch.tensor(0.0)
                else:
                    if step % cong_grad_freq == 0:
                        congestion = _rudy_congestion(clamped, net_data, port_base,
                                                      gamma,
                                                      cell_x_min, cell_x_max,
                                                      cell_y_min, cell_y_max,
                                                      grid_h_routes, grid_v_routes,
                                                      grid_rows, grid_cols)
                        cached_cong = congestion.item()
                    else:
                        congestion = torch.tensor(cached_cong)

                overlap = _overlap_penalty(clamped[:n_hard], half_sizes[:n_hard])
                overlap_norm = overlap / (cw * ch)

                proxy = wl + self.density_weight * density + self.congestion_weight * congestion
                loss = proxy + lam * overlap_norm

                loss.backward()
                torch.nn.utils.clip_grad_norm_([pos], max_norm=20.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    proxy_val = proxy.item()
                    ov_val = overlap_norm.item()
                    score = proxy_val + max(1.0, lam * 0.1) * ov_val
                    if score < best_proxy:
                        best_proxy = score
                        bp = pos.data.clone()
                        bp[fixed_mask] = init_pos[fixed_mask]
                        bp[:, 0].clamp_(half_sizes[:, 0], cw - half_sizes[:, 0])
                        bp[:, 1].clamp_(half_sizes[:, 1], ch - half_sizes[:, 1])
                        best_pos = bp

                if step == 0:
                    with torch.no_grad():
                        ablation_tag = ""
                        if self.skip_congestion:
                            ablation_tag += " [no-cong]"
                        if self.skip_density:
                            ablation_tag += " [no-dens]"
                        if self.max_phases < 3:
                            ablation_tag += f" [phase-{phase_idx+1}-only]"
                        print(f"    P{phase_idx+1}{ablation_tag}: wl={wl.item():.3f} "
                              f"den={density.item():.3f} "
                              f"cong={congestion.item():.3f}")

            with torch.no_grad():
                ov_count = _count_overlaps(best_pos[:n_hard], half_sizes[:n_hard])
                print(f"  Ablation phase {phase_idx+1} done: best_score={best_proxy:.4f}  "
                      f"overlaps={ov_count}")

        return best_pos

    def _do_legalize(self, positions, benchmark):
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()

        pos_np = positions.detach().numpy().astype(np.float64)
        legal = _legalize(pos_np, sizes, movable, n_hard, cw, ch)

        result = positions.clone()
        result[:n_hard] = torch.tensor(legal[:n_hard], dtype=torch.float32)
        return result
