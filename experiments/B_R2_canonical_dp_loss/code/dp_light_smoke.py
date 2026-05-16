"""DREAMPlace-light: Adam descent on (smooth WL + canonical losses) with periodic projection.

Stripped-down version of B-R2 — doesn't need full DREAMPlace build. Tests
whether the canonical_losses gradient direction is useful for optimization.

E88's failure was Adam on smooth_proxy alone (LSE-HPWL + Gaussian density + DPO-RUDY).
This adds:
  - Periodic project_overlaps to keep feasible during descent
  - top-K density (matches canonical's top-K)
  - RUDY (closer to canonical)

If this beats E25 init from scratch on canonical proxy, B-R2 likely works
when wired into real DP. If it diverges, B-R2 needs more than custom losses.

Usage:
    python3 dp_light_smoke.py <bench_name> [budget_seconds]
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from canonical_losses import topk_density_loss, rudy_congestion_loss
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
from macro_place.cd_core import project_overlaps


def build_netpin_csr(benchmark):
    """Build netpin_start (CSR) + flat_netpin from benchmark.net_nodes (List[Tensor])."""
    nets = benchmark.net_nodes
    flat = []
    starts = [0]
    for net in nets:
        flat.extend([int(x) for x in net.tolist()])
        starts.append(len(flat))
    return (torch.tensor(starts, dtype=torch.long),
            torch.tensor(flat, dtype=torch.long))


def smooth_hpwl(positions, sizes, netpin_start, flat_netpin, net_weights, gamma=0.1):
    """LSE-HPWL (smooth bounding box per net), differentiable.

    Per net i with pins at positions p_1..p_k:
      bbox_x = LSE(x_p; tau) - LSE(-x_p; tau)
    where LSE(z; tau) = tau * log(sum(exp(z/tau))).
    """
    device = positions.device
    n_nets = netpin_start.shape[0] - 1
    total = torch.zeros((), dtype=positions.dtype, device=device)
    for i in range(n_nets):
        s = int(netpin_start[i])
        e = int(netpin_start[i + 1])
        if e <= s + 1:
            continue
        pins = positions[flat_netpin[s:e]]
        # x
        lse_x_hi = gamma * torch.logsumexp(pins[:, 0] / gamma, dim=0)
        lse_x_lo = -gamma * torch.logsumexp(-pins[:, 0] / gamma, dim=0)
        # y
        lse_y_hi = gamma * torch.logsumexp(pins[:, 1] / gamma, dim=0)
        lse_y_lo = -gamma * torch.logsumexp(-pins[:, 1] / gamma, dim=0)
        bbox = (lse_x_hi - lse_x_lo) + (lse_y_hi - lse_y_lo)
        total = total + net_weights[i] * bbox
    return total


def run_smoke(bench_name: str, budget_s: float = 600.0, log=None):
    if log is None:
        log = lambda s: print(s, flush=True)

    bench, plc = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{bench_name}')
    n = bench.num_macros
    W = float(bench.canvas_width)
    H = float(bench.canvas_height)
    log(f"[smoke] {bench_name}: num_macros={n} num_hard={bench.num_hard_macros} canvas={W:.1f}x{H:.1f}")

    # Initial canonical
    init_costs = compute_proxy_cost(bench.macro_positions, bench, plc)
    log(f"[smoke] init canonical: proxy={init_costs['proxy_cost']:.5f} "
        f"d={init_costs['density_cost']:.5f} c={init_costs['congestion_cost']:.5f}")

    # netpin CSR — net_nodes indices reach into [macros, ports] space
    netpin_start, flat_netpin = build_netpin_csr(bench)
    net_weights = (bench.net_weights if hasattr(bench, 'net_weights')
                   else torch.ones(netpin_start.shape[0] - 1, dtype=torch.float32))

    num_ports = int(bench.port_positions.shape[0])
    n_total = n + num_ports

    # Random init (or use existing positions, lightly perturbed)
    torch.manual_seed(0)
    # Start from a random uniform init within canvas (DREAMPlace-style fresh start)
    # We optimize a full pos_full tensor of shape [n + num_ports, 2]
    pos_full = torch.empty(n_total, 2, dtype=torch.float32)
    pos_full[:n, 0] = torch.rand(n) * W * 0.6 + 0.2 * W
    pos_full[:n, 1] = torch.rand(n) * H * 0.6 + 0.2 * H
    # Preserve fixed macros
    pos_full[:n][bench.macro_fixed] = bench.macro_positions[bench.macro_fixed]
    # Ports are fixed at port_positions
    if num_ports > 0:
        pos_full[n:] = bench.port_positions
    pos_full = pos_full.requires_grad_(True)
    # The "pos" we surface to canonical_proxy is the macro slice
    pos = pos_full[:n]  # view; gradient propagates to pos_full

    # Sizes (macros only)
    sizes = bench.macro_sizes.detach()
    fixed_mask = torch.cat([bench.macro_fixed, torch.ones(num_ports, dtype=torch.bool)])

    # Calibrate lambdas at the initial random placement
    with torch.no_grad():
        wl_init = smooth_hpwl(pos_full, sizes, netpin_start, flat_netpin, net_weights, gamma=W * 0.005)
        macro_pos = pos_full[:n]
        topk_init = topk_density_loss(macro_pos, sizes, 0.0, 0.0, W, H, 32, 32)
        rudy_init = rudy_congestion_loss(pos_full, netpin_start, flat_netpin, net_weights,
                                          0.0, 0.0, W, H, 32, 32)
    lambda_topk = 1.0 * float(wl_init) / max(float(topk_init), 1e-9)
    lambda_rudy = 0.5 * float(wl_init) / max(float(rudy_init), 1e-9)
    log(f"[smoke] init wl={float(wl_init):.3f} topk={float(topk_init):.3f} "
        f"rudy={float(rudy_init):.3f}")
    log(f"[smoke] lambdas: topk={lambda_topk:.6f} rudy={lambda_rudy:.6f}")

    # Optimizer over the full pos tensor (macros + ports). Port grads zeroed.
    opt = torch.optim.Adam([pos_full], lr=W * 0.01)
    t0 = time.time()
    iter_n = 0
    PROJECT_EVERY = 50
    LOG_EVERY = 25

    best_proxy = float('inf')
    best_pos = pos_full[:n].detach().clone()

    while time.time() - t0 < budget_s:
        iter_n += 1
        opt.zero_grad()
        # Build smooth objective
        wl = smooth_hpwl(pos_full, sizes, netpin_start, flat_netpin, net_weights, gamma=W * 0.005)
        macro_pos = pos_full[:n]
        topk = topk_density_loss(macro_pos, sizes, 0.0, 0.0, W, H, 32, 32)
        rudy = rudy_congestion_loss(pos_full, netpin_start, flat_netpin, net_weights,
                                     0.0, 0.0, W, H, 32, 32)
        obj = wl + lambda_topk * topk + lambda_rudy * rudy
        obj.backward()
        # Mask fixed macros' + port grads
        if pos_full.grad is not None:
            pos_full.grad[fixed_mask] = 0.0
        opt.step()
        # Clamp macros to canvas (port positions stay fixed via zero grad)
        with torch.no_grad():
            pos_full[:n, 0].clamp_(sizes[:, 0] * 0.5, W - sizes[:, 0] * 0.5)
            pos_full[:n, 1].clamp_(sizes[:, 1] * 0.5, H - sizes[:, 1] * 0.5)

        if iter_n % PROJECT_EVERY == 0 or iter_n == 1:
            # Project to overlap-feasibility (macros only)
            with torch.no_grad():
                macro_only = pos_full[:n].detach().clone()
                projected, _ = project_overlaps(macro_only, bench)
                pos_full.data[:n] = projected
                canon = compute_proxy_cost(projected, bench, plc)
                ovl = compute_overlap_metrics(projected, bench)['overlap_count']
                if canon['proxy_cost'] < best_proxy and ovl == 0:
                    best_proxy = canon['proxy_cost']
                    best_pos = projected.clone()
                if iter_n % LOG_EVERY == 0:
                    log(f"[smoke] iter {iter_n} t={time.time()-t0:.0f}s "
                        f"obj={float(obj):.3f} wl={float(wl):.3f} "
                        f"topk={float(topk):.3f} rudy={float(rudy):.3f} "
                        f"| canon proxy={canon['proxy_cost']:.5f} d={canon['density_cost']:.4f} "
                        f"c={canon['congestion_cost']:.4f} ovl={ovl}")

    # Final eval
    final_costs = compute_proxy_cost(best_pos, bench, plc)
    ovl = compute_overlap_metrics(best_pos, bench)['overlap_count']
    log(f"[smoke] DONE iter={iter_n} wall={time.time()-t0:.0f}s")
    log(f"[smoke] best canonical: proxy={final_costs['proxy_cost']:.5f} "
        f"d={final_costs['density_cost']:.5f} c={final_costs['congestion_cost']:.5f} ovl={ovl}")
    log(f"[smoke] vs init {init_costs['proxy_cost']:.5f}: "
        f"Δ={(final_costs['proxy_cost']-init_costs['proxy_cost'])/init_costs['proxy_cost']*100:+.2f}%")

    return {
        'bench': bench_name,
        'init_proxy': init_costs['proxy_cost'],
        'final_proxy': final_costs['proxy_cost'],
        'overlaps': ovl,
        'iters': iter_n,
        'wall_s': time.time() - t0,
    }


if __name__ == '__main__':
    bench = sys.argv[1] if len(sys.argv) > 1 else 'ibm01'
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    run_smoke(bench, budget)
