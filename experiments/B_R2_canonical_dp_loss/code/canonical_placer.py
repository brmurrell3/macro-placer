"""B-R2 canonical placer: pure-PyTorch Nesterov-momentum on canonical-aligned losses.

Avoids DREAMPlace integration entirely. Implements:
  - Smooth WL: LSE-HPWL over net bboxes
  - Top-K density: our canonical_losses.topk_density_loss
  - RUDY congestion: our canonical_losses.rudy_congestion_loss
  - Nesterov accelerated gradient (manual, not Adam)
  - Density weight ramp schedule (warmup → cap)
  - Periodic feasibility projection
  - Best-so-far tracking after each projection

This is the "B-R2 minus DREAMPlace" path. If it produces basin proxy
better than stock-DP (1.10-1.40 typical), B-R2 is alive without DP.
If it diverges (like E88), B-R2 needs DREAMPlace's machinery (back to
fixing the build).

Usage:
    python3 canonical_placer.py <bench> [budget_s] [lambda_topk] [lambda_rudy]
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from canonical_losses import topk_density_loss, rudy_congestion_loss
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
from macro_place.cd_core import project_overlaps, sdf_init


def build_netpin_csr(benchmark):
    """net_nodes indices include macros + ports.

    Returns CSR (netpin_start, flat_netpin) over the unified [macros, ports] space.
    """
    nets = benchmark.net_nodes
    flat = []
    starts = [0]
    for net in nets:
        flat.extend([int(x) for x in net.tolist()])
        starts.append(len(flat))
    return (torch.tensor(starts, dtype=torch.long),
            torch.tensor(flat, dtype=torch.long))


def smooth_hpwl_vectorized(positions, netpin_start, flat_netpin, net_weights, gamma):
    """LSE-HPWL via per-net LSE over pin positions.

    positions: [N_pins, 2] (full pin universe = macros + ports)
    netpin_start, flat_netpin: CSR
    net_weights: [N_nets]
    gamma: LSE smoothing
    """
    device = positions.device
    dtype = positions.dtype
    n_nets = netpin_start.shape[0] - 1
    total = torch.zeros((), dtype=dtype, device=device)
    for i in range(n_nets):
        s = int(netpin_start[i])
        e = int(netpin_start[i + 1])
        if e <= s + 1:
            continue
        pins = positions[flat_netpin[s:e]]
        bbox = (
            gamma * (torch.logsumexp(pins[:, 0] / gamma, dim=0) + torch.logsumexp(-pins[:, 0] / gamma, dim=0))
            + gamma * (torch.logsumexp(pins[:, 1] / gamma, dim=0) + torch.logsumexp(-pins[:, 1] / gamma, dim=0))
        )
        total = total + net_weights[i] * bbox
    return total


class CanonicalPlacer:
    """B-R2 placer: Nesterov on smooth WL + canonical density + RUDY."""

    def __init__(
        self,
        budget_seconds: float = 600.0,
        lambda_topk: float = 1.0,
        lambda_rudy: float = 0.5,
        density_weight_init: float = 0.001,
        density_weight_max: float = 10.0,
        density_weight_ramp_iters: int = 200,
        learning_rate: float = 1.0,
        momentum: float = 0.95,
        project_every: int = 20,
        log_every: int = 25,
        gamma_frac: float = 0.005,  # gamma = canvas_w * gamma_frac
        num_bins_x: int = 32,
        num_bins_y: int = 32,
        K_frac: float = 0.10,
        tau: float = 0.05,
        seed: int = 42,
    ):
        self.budget_seconds = budget_seconds
        self.lambda_topk = lambda_topk
        self.lambda_rudy = lambda_rudy
        self.density_weight_init = density_weight_init
        self.density_weight_max = density_weight_max
        self.density_weight_ramp_iters = density_weight_ramp_iters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.project_every = project_every
        self.log_every = log_every
        self.gamma_frac = gamma_frac
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.K_frac = K_frac
        self.tau = tau
        self.seed = seed

    def place(self, benchmark, plc=None, init_pos: torch.Tensor = None, log=None):
        if log is None:
            log = lambda s: print(s, flush=True)
        if plc is None:
            from macro_place.objective import compute_proxy_cost as _ggg  # ensure import
            # plc is needed for canon eval; load from benchmark dir if not given
            raise ValueError("plc required for canonical eval")

        n = benchmark.num_macros
        n_ports = int(benchmark.port_positions.shape[0])
        n_pins = n + n_ports
        W = float(benchmark.canvas_width)
        H = float(benchmark.canvas_height)
        gamma = W * self.gamma_frac

        torch.manual_seed(self.seed)

        # Init position: SDF or given
        if init_pos is None:
            init_pos = sdf_init(benchmark)
        sizes = benchmark.macro_sizes.detach()

        # Build full pin position tensor [macros, ports]; gradients only through macros
        pos_full = torch.zeros(n_pins, 2, dtype=torch.float32)
        pos_full[:n] = init_pos.detach()
        if n_ports > 0:
            pos_full[n:] = benchmark.port_positions
        pos_full = pos_full.requires_grad_(True)

        # Fixed mask: macros that are fixed + all ports
        fixed_mask = torch.cat([benchmark.macro_fixed, torch.ones(n_ports, dtype=torch.bool)])

        # netpin CSR
        netpin_start, flat_netpin = build_netpin_csr(benchmark)
        net_weights = benchmark.net_weights

        # Initial canonical eval
        init_canon = compute_proxy_cost(init_pos, benchmark, plc)
        log(f"[B-R2] {benchmark.name}: num_macros={n} num_hard={benchmark.num_hard_macros} canvas={W:.1f}x{H:.1f}")
        log(f"[B-R2] init canon: proxy={init_canon['proxy_cost']:.5f} d={init_canon['density_cost']:.4f} c={init_canon['congestion_cost']:.4f}")

        # Calibrate lambdas to balance term contributions at init
        with torch.no_grad():
            wl_i = smooth_hpwl_vectorized(pos_full, netpin_start, flat_netpin, net_weights, gamma)
            tk_i = topk_density_loss(pos_full[:n], sizes, 0.0, 0.0, W, H, self.num_bins_x, self.num_bins_y, self.K_frac, self.tau)
            rd_i = rudy_congestion_loss(pos_full, netpin_start, flat_netpin, net_weights, 0.0, 0.0, W, H, self.num_bins_x, self.num_bins_y, self.K_frac, self.tau)
        # Lambda calibration: each term should contribute ~lambda * wl to obj
        l_topk_eff = self.lambda_topk * float(wl_i) / max(float(tk_i), 1e-9)
        l_rudy_eff = self.lambda_rudy * float(wl_i) / max(float(rd_i), 1e-9)
        log(f"[B-R2] init wl={float(wl_i):.0f} topk={float(tk_i):.2f} rudy={float(rd_i):.2f}")
        log(f"[B-R2] calibrated lambdas: topk={l_topk_eff:.6f} rudy={l_rudy_eff:.6f}")

        # Nesterov state
        velocity = torch.zeros_like(pos_full.detach())

        t0 = time.time()
        iter_n = 0
        best_proxy = float('inf')
        best_pos = init_pos.detach().clone()

        while time.time() - t0 < self.budget_seconds:
            iter_n += 1

            # Density weight schedule
            ramp = min(1.0, iter_n / max(1, self.density_weight_ramp_iters))
            dw = self.density_weight_init + ramp * (self.density_weight_max - self.density_weight_init)

            # Forward
            pos_full.grad = None
            wl = smooth_hpwl_vectorized(pos_full, netpin_start, flat_netpin, net_weights, gamma)
            tk = topk_density_loss(pos_full[:n], sizes, 0.0, 0.0, W, H,
                                    self.num_bins_x, self.num_bins_y, self.K_frac, self.tau)
            rd = rudy_congestion_loss(pos_full, netpin_start, flat_netpin, net_weights,
                                       0.0, 0.0, W, H, self.num_bins_x, self.num_bins_y, self.K_frac, self.tau)
            obj = wl + dw * (l_topk_eff * tk + l_rudy_eff * rd)
            obj.backward()

            with torch.no_grad():
                # Mask fixed grads
                grad = pos_full.grad.clone()
                grad[fixed_mask] = 0.0

                # Nesterov: velocity = momentum * velocity - lr * grad; pos += velocity
                velocity.mul_(self.momentum).add_(grad, alpha=-self.learning_rate)
                pos_full.add_(velocity)

                # Clamp macros to canvas (account for size)
                pos_full[:n, 0].clamp_(sizes[:, 0] * 0.5, W - sizes[:, 0] * 0.5)
                pos_full[:n, 1].clamp_(sizes[:, 1] * 0.5, H - sizes[:, 1] * 0.5)

            if iter_n % self.project_every == 0 or iter_n == 1:
                with torch.no_grad():
                    macro_pos = pos_full[:n].detach().clone()
                    projected, _ = project_overlaps(macro_pos, benchmark)
                    pos_full.data[:n] = projected
                    velocity[:n] = 0.0  # reset velocity for projected macros
                    canon = compute_proxy_cost(projected, benchmark, plc)
                    ovl = compute_overlap_metrics(projected, benchmark)['overlap_count']
                    if canon['proxy_cost'] < best_proxy and ovl == 0:
                        best_proxy = canon['proxy_cost']
                        best_pos = projected.clone()
                    if iter_n % self.log_every == 0:
                        log(f"[B-R2] iter {iter_n} t={time.time()-t0:.0f}s "
                            f"obj={float(obj):.0f} wl={float(wl):.0f} dw={dw:.3f} "
                            f"tk={float(tk):.2f} rd={float(rd):.2f} | "
                            f"canon proxy={canon['proxy_cost']:.5f} ovl={ovl} "
                            f"BEST={best_proxy:.5f}")

        # Final eval on best
        final = compute_proxy_cost(best_pos, benchmark, plc)
        ovl = compute_overlap_metrics(best_pos, benchmark)['overlap_count']
        log(f"[B-R2] DONE iter={iter_n} wall={time.time()-t0:.0f}s")
        log(f"[B-R2] best canon: proxy={final['proxy_cost']:.5f} d={final['density_cost']:.4f} c={final['congestion_cost']:.4f} ovl={ovl}")
        log(f"[B-R2] vs init {init_canon['proxy_cost']:.5f}: Δ={(final['proxy_cost']-init_canon['proxy_cost'])/init_canon['proxy_cost']*100:+.2f}%")

        return best_pos


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 else 'ibm01'
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    lambda_topk = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    lambda_rudy = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    bench, plc = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{bench_name}')

    placer = CanonicalPlacer(
        budget_seconds=budget,
        lambda_topk=lambda_topk,
        lambda_rudy=lambda_rudy,
    )
    result = placer.place(bench, plc)
    print()
    final = compute_proxy_cost(result, bench, plc)
    print(json.dumps({
        'bench': bench_name,
        'final_proxy': final['proxy_cost'],
        'final_density': final['density_cost'],
        'final_congestion': final['congestion_cost'],
        'final_wl': final['wirelength_cost'],
        'overlaps': compute_overlap_metrics(result, bench)['overlap_count'],
    }, indent=2))


if __name__ == '__main__':
    main()
