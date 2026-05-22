"""Hessian saddle escape: from a CD plateau, follow the smallest-magnitude
eigenvector of the smooth proxy Hessian for ε, then run TILOS-proxy CD.

Theory: at a CD-LNS-SA plateau, the placement is a local minimum of the
TILOS proxy under the available move sets. The smooth proxy Hessian has
its smallest-magnitude eigenvector pointing along the "softest" direction
— the direction in which the surface curves least. Following ±ε along it
takes us off the plateau into a region where CD can resume making
progress.

This is the "dimer / gentlest-ascent" approach in saddle-point literature
(Henkelman & Jónsson 2000), specialized to placement via the DPO smooth
proxy as the differentiable surrogate.

Implementation:
  - Smooth proxy: use DPO's _lse_hpwl + _grid_density + _rudy_congestion
    (all torch-autograd-friendly).
  - Hessian-vector product: torch.autograd.functional.hvp.
  - Lanczos eigenvectors: scipy.sparse.linalg.eigsh with LinearOperator.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _p in (_ROOT, _HERE):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# DPO smooth-proxy primitives — bundled sibling in lib/.
from ablation_v2_steps import (  # noqa: E402
    _lse_hpwl,
    _grid_density,
    _rudy_congestion,
    _extract_net_data,
)

from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.cd_core import project_overlaps, run_cd_adaptive  # noqa: E402
from macro_place.incremental_evaluator import IncrementalProxyEvaluator  # noqa: E402
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost  # noqa: E402


class SmoothProxy:
    """Builds a torch-autograd-friendly composite proxy = wl + 0.5*den + 0.5*cong.

    Excludes overlap_penalty — at a feasible E48 plateau, overlap is 0 and
    Hessian doesn't need that term.
    """

    def __init__(self, benchmark: Benchmark, plc, gamma_frac: float = 0.0005):
        self.benchmark = benchmark
        self.plc = plc
        self.n_hard = benchmark.num_hard_macros
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.sizes = benchmark.macro_sizes
        self.half_sizes = self.sizes / 2.0

        # Net topology.
        self.net_data = _extract_net_data(benchmark, plc)
        self.gamma = gamma_frac * self.cw  # match DPO phase 3 sharpening

        # Grid setup.
        self.grid_rows = benchmark.grid_rows
        self.grid_cols = benchmark.grid_cols
        self.cell_w = self.cw / self.grid_cols
        self.cell_h = self.ch / self.grid_rows
        self.cell_area = self.cell_w * self.cell_h
        self.cell_x_min = torch.arange(self.grid_cols, dtype=torch.float32) * self.cell_w
        self.cell_x_max = self.cell_x_min + self.cell_w
        self.cell_y_min = torch.arange(self.grid_rows, dtype=torch.float32) * self.cell_h
        self.cell_y_max = self.cell_y_min + self.cell_h
        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron
        self.port_base = torch.zeros(1, 2)
        self.wl_norm = (self.cw + self.ch) * self.net_data.total_net_count

    def cost(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute smooth proxy at `positions`. Returns a scalar tensor."""
        # Clamp to canvas (preserves gradient via clamp's piecewise-linearity).
        clamped = torch.stack([
            positions[:, 0].clamp(self.half_sizes[:, 0], self.cw - self.half_sizes[:, 0]),
            positions[:, 1].clamp(self.half_sizes[:, 1], self.ch - self.half_sizes[:, 1]),
        ], dim=1)

        wl = _lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        density = _grid_density(
            clamped, self.sizes,
            self.cell_x_min, self.cell_x_max,
            self.cell_y_min, self.cell_y_max,
            self.cell_area, self.grid_rows, self.grid_cols,
        )
        congestion = _rudy_congestion(
            clamped, self.net_data, self.port_base, self.gamma,
            self.cell_x_min, self.cell_x_max,
            self.cell_y_min, self.cell_y_max,
            self.grid_h_routes, self.grid_v_routes,
            self.grid_rows, self.grid_cols,
        )
        return wl + 0.5 * density + 0.5 * congestion


def hessian_vector_product(
    smooth_proxy: SmoothProxy,
    state: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute H @ v where H = ∂²f/∂p² evaluated at `state`.

    Uses torch.autograd.functional.hvp.

    Args:
      state: [N, 2] tensor of macro positions (no grad needed on input).
      v: [N, 2] tensor of perturbation direction.

    Returns:
      H @ v as [N, 2] tensor (same shape as state).
    """
    state_flat = state.detach().clone().requires_grad_(True)
    v_flat = v.detach().clone()

    # torch.autograd.functional.hvp computes (f, H @ v) given f and v.
    f_value, hv = torch.autograd.functional.hvp(
        smooth_proxy.cost, state_flat, v_flat, create_graph=False, strict=False,
    )
    return hv.detach()


def find_softest_eigenvectors(
    smooth_proxy: SmoothProxy,
    state: torch.Tensor,
    movable_mask: np.ndarray,
    *,
    k: int = 5,
    tol: float = 1e-4,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find k eigenvectors with smallest-magnitude eigenvalues.

    Operates on the MOVABLE macros only (fixed macros' positions don't
    contribute to the search space). The Hessian is restricted to movable
    coordinates by zeroing the v entries for fixed macros before HVP.

    Returns:
      eigvals: [k] float — eigenvalues
      eigvecs: [n_movable_dim, k] float — eigenvectors as 1D (flattened
        position vectors restricted to movable coords)
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_macros = state.shape[0]
    n_dim = n_macros * 2  # x, y per macro

    # Build mask flat for [n_dim] vector.
    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_mask):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True
    n_free = int(mask_flat.sum())
    log(f"[hessian] n_macros={n_macros}, n_free={n_free} (movable coords)")

    state_full = state.detach().clone()

    def matvec(v_free: np.ndarray) -> np.ndarray:
        # v_free is the [n_free] movable component.
        # Embed into full [n_dim] vector with fixed coords = 0.
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = v_free
        v_tensor = torch.tensor(v_full.reshape(n_macros, 2), dtype=torch.float32)
        hv = hessian_vector_product(smooth_proxy, state_full, v_tensor)
        # Extract movable coords.
        hv_flat = hv.cpu().numpy().reshape(n_dim)
        return hv_flat[mask_flat]

    op = spla.LinearOperator(
        shape=(n_free, n_free), matvec=matvec, dtype=np.float64,
    )

    log(f"[hessian] running Lanczos eigsh for {k} smallest-algebraic eigenvalues...")
    t0 = time.time()
    # 'SA' = smallest algebraic (most-negative if any; else smallest positive).
    # This is what we want for saddle escape — soft modes have smallest
    # curvature. Far more numerically stable than 'SM' for indefinite H.
    eigvals = []
    eigvecs = np.zeros((n_free, 0))
    try:
        eigvals, eigvecs = spla.eigsh(
            op, k=k, which="SA", tol=tol, maxiter=500, ncv=min(2 * k + 20, n_free),
        )
    except spla.ArpackNoConvergence as e:
        log(f"[hessian] 'SA' didn't converge after maxiter; "
            f"got {len(e.eigenvalues)} of {k} eigenvalues")
        if len(e.eigenvalues) > 0:
            eigvals = e.eigenvalues
            eigvecs = e.eigenvectors

    if len(eigvals) == 0:
        log(f"[hessian] FALLBACK: trying single-vector power iteration on -H")
        # Shifted-power-iteration fallback: find smallest eigenvalue via
        # iterating on (αI - H) for α large. Compute α via random probe.
        v_random = np.random.randn(n_free).astype(np.float64)
        v_random /= np.linalg.norm(v_random)
        # Estimate ||H|| via 5 power iterations.
        v_est = v_random.copy()
        h_norm_est = 1.0
        for _ in range(5):
            v_new = matvec(v_est)
            h_norm_est = np.linalg.norm(v_new)
            if h_norm_est < 1e-12:
                break
            v_est = v_new / h_norm_est
        log(f"[hessian]   estimated ||H|| ~ {h_norm_est:.4f}")
        alpha = 2.0 * h_norm_est + 1.0

        # Shifted power iteration on (αI - H): largest eigvec corresponds to
        # smallest eigvec of H.
        v = np.random.randn(n_free).astype(np.float64)
        v /= np.linalg.norm(v)
        for it in range(200):
            v_new = alpha * v - matvec(v)
            n_new = np.linalg.norm(v_new)
            if n_new < 1e-12:
                break
            v = v_new / n_new
        # Compute corresponding eigenvalue: λ = v^T H v / (v^T v)
        hv = matvec(v)
        lam = float(np.dot(v, hv) / (np.dot(v, v) + 1e-12))
        eigvals = np.array([lam])
        eigvecs = v.reshape(-1, 1)
        log(f"[hessian]   power iteration found λ_min ≈ {lam:.4e}")

    log(f"[hessian] Lanczos done in {time.time() - t0:.1f}s; eigvals = {eigvals}")
    return eigvals, eigvecs


def saddle_escape(
    state: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    n_eigvecs: int = 5,
    epsilon_values: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0, 2.0),
    cd_polish_budget: float = 600.0,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Saddle escape from a plateau state.

    Builds smooth proxy → finds k softest eigenvectors → for each ε,
    perturbs state by ±ε·v_k → project_overlaps → CD polish → check
    proxy. Returns best result and stats.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    smooth = SmoothProxy(benchmark, plc)
    log(f"[saddle] smooth-proxy built; n_macros={state.shape[0]}, n_nets={len(smooth.net_data.weights)}")

    movable_np = (~benchmark.macro_fixed.cpu().numpy())
    eigvals, eigvecs = find_softest_eigenvectors(
        smooth, state, movable_np, k=n_eigvecs, log=log,
    )

    # Compute starting proxy.
    start_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[saddle] starting proxy: {start_proxy:.5f}")

    n_macros = state.shape[0]
    n_dim = n_macros * 2
    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_np):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True

    state_np = state.detach().cpu().numpy()
    best_state = state.detach().clone()
    best_proxy = start_proxy
    attempts = []

    for k in range(eigvecs.shape[1]):
        v_free = eigvecs[:, k]
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = v_free
        v_2d = v_full.reshape(n_macros, 2)
        # Normalize per-macro displacement to unit RMS.
        v_norm = np.linalg.norm(v_2d) + 1e-12
        v_2d_unit = v_2d / v_norm

        for sign in [+1.0, -1.0]:
            for eps in epsilon_values:
                t_attempt = time.time()
                perturbed_np = state_np + sign * eps * v_2d_unit
                # Clamp to canvas.
                hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
                hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
                perturbed_np[:, 0] = np.clip(perturbed_np[:, 0], hw_np, smooth.cw - hw_np)
                perturbed_np[:, 1] = np.clip(perturbed_np[:, 1], hh_np, smooth.ch - hh_np)
                perturbed = torch.tensor(perturbed_np, dtype=torch.float32)
                # Restore fixed.
                perturbed[~torch.tensor(movable_np)] = state[~torch.tensor(movable_np)]

                # Legalize.
                perturbed, _ = project_overlaps(perturbed, benchmark)
                ovl = compute_overlap_metrics(perturbed, benchmark)["overlap_count"]
                if ovl > 0:
                    log(f"[saddle] eig{k} sign={sign:+.0f} eps={eps:.2f}: infeasible after project (residual {ovl})")
                    attempts.append({
                        "eig": k, "sign": sign, "eps": eps,
                        "feasible": False, "residual": ovl,
                    })
                    continue

                # CD polish.
                evaluator = IncrementalProxyEvaluator(benchmark, plc, perturbed.clone())
                fixed = benchmark.macro_fixed.cpu().numpy()
                movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
                run_cd_adaptive(
                    evaluator, benchmark, plc, movable_idx,
                    min_time_s=30.0, hard_cap_s=cd_polish_budget,
                    patience=3, plateau_threshold=0.001, log_fn=None,
                )
                polished = evaluator.placement.detach().clone().to(torch.float32)
                polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
                polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]

                attempt_wall = time.time() - t_attempt
                log(f"[saddle] eig{k} sign={sign:+.0f} eps={eps:.2f}: polished={polished_proxy:.5f} "
                    f"overlap={polished_overlap} wall={attempt_wall:.0f}s")
                attempts.append({
                    "eig": k, "sign": sign, "eps": eps,
                    "feasible": True, "polished_proxy": polished_proxy,
                    "polished_overlap": polished_overlap,
                    "wall": attempt_wall,
                })

                if polished_overlap == 0 and polished_proxy < best_proxy - 1e-7:
                    best_proxy = polished_proxy
                    best_state = polished.detach().clone()
                    log(f"[saddle] NEW BEST: {best_proxy:.5f} (Δ vs start = {best_proxy - start_proxy:+.5f})")

    return best_state, {
        "starting_proxy": start_proxy,
        "best_proxy": best_proxy,
        "improvement": start_proxy - best_proxy,
        "improvement_frac": (start_proxy - best_proxy) / start_proxy if start_proxy > 0 else 0.0,
        "n_eigvecs": n_eigvecs,
        "eigenvalues": eigvals.tolist() if hasattr(eigvals, "tolist") else list(eigvals),
        "attempts": attempts,
    }
