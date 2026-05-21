"""E120 — ContinuousHessianSaddle: Hessian saddle escape INSIDE Adam descent.

Novel formulation: applies the cascade saddle escape (E84) to the
**continuous** smooth-proxy landscape instead of the **combinatorial** CD
plateau. After Adam converges to a basin, we compute the Hessian of the
V3 smooth proxy at the current position, find the smallest-algebraic
eigenvector v (the "softest" direction in the landscape), perturb
positions ±ε along v, and resume Adam from each perturbed state. The
best resulting smooth proxy becomes the new basin.

The two saddle escapes (E84 combinatorial, E120 continuous) operate on
entirely different landscapes and address different failure modes:
- E84: CD-LNS-SA settles at a discrete local minimum under available
  moves. Saddle escape on the smooth proxy at that point picks the
  most-curving direction; ε-perturbation moves us off the plateau.
- E120: Adam descent settles at a continuous local minimum of the
  smooth proxy. The Hessian eigenvector gives a *principled* direction
  to escape the basin (vs random restart / multi-init).

Architecture:
  Phase A: V3 Adam descent (~300 steps)              → settle in basin
  Phase B: Hessian saddle escape (eigsh + perturb)   → escape basin
           Adam resume from each candidate (~150 steps)
           Pick best resulting smooth proxy
  Phase C: 1-2 more saddle escapes if budget remains
  Phase D: greedy_macro_legalize + CD polish (standard)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import DiffProxyV3, loss_with_penalty_v3
from macro_legalizer import greedy_macro_legalize


# ============================================================
# Hessian eigenvector primitives (continuous-space, V3 proxy)
# ============================================================

def _hvp_smooth_proxy(
    proxy: DiffProxyV3,
    state: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Hessian-vector product on the V3 smooth proxy (wl + 0.5d + 0.5c).

    Note: this uses ONLY the smooth proxy (no overlap penalty, no
    boundary penalty). The basin we want to escape is defined by the
    smooth proxy alone; overlap/boundary terms only matter for
    feasibility (handled by clamping and project_overlaps).
    """
    state_var = state.detach().clone().requires_grad_(True)
    v_var = v.detach().clone()

    def scalar_cost(s: torch.Tensor) -> torch.Tensor:
        c, _ = proxy.cost(s, include_congestion=True)
        return c

    _, hv = torch.autograd.functional.hvp(
        scalar_cost, state_var, v_var, create_graph=False, strict=False,
    )
    return hv.detach()


def find_softest_eigenvector_smooth(
    proxy: DiffProxyV3,
    state: torch.Tensor,
    movable_mask_np: np.ndarray,
    *,
    k: int = 1,
    tol: float = 1e-3,
    maxiter: int = 300,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the smallest-algebraic eigenvalue/vector of the V3 smooth proxy's Hessian.

    Restricts the operator to MOVABLE coords. Returns:
      eigvals: [k]  float
      eigvecs: [n_free, k]  float  (movable coords flattened)
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_macros = state.shape[0]
    n_dim = n_macros * 2

    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_mask_np):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True
    n_free = int(mask_flat.sum())

    state_full = state.detach().clone()

    def matvec(v_free: np.ndarray) -> np.ndarray:
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = v_free
        v_tensor = torch.tensor(
            v_full.reshape(n_macros, 2), dtype=torch.float32, device=state_full.device,
        )
        hv = _hvp_smooth_proxy(proxy, state_full, v_tensor)
        hv_flat = hv.cpu().numpy().reshape(n_dim)
        return hv_flat[mask_flat]

    op = spla.LinearOperator(
        shape=(n_free, n_free), matvec=matvec, dtype=np.float64,
    )

    t0 = time.time()
    try:
        ncv = min(2 * k + 20, n_free)
        eigvals, eigvecs = spla.eigsh(
            op, k=k, which="SA", tol=tol, maxiter=maxiter, ncv=ncv,
        )
    except spla.ArpackNoConvergence as e:
        log(f"[saddle] eigsh did not converge after {maxiter}; "
            f"got {len(e.eigenvalues)}/{k}")
        if len(e.eigenvalues) > 0:
            eigvals, eigvecs = e.eigenvalues, e.eigenvectors
        else:
            # Fallback: shifted power iteration on (alpha*I - H).
            log(f"[saddle] FALLBACK power iteration (alpha = 2 * ||H||_est + 1)")
            v = np.random.RandomState(42).randn(n_free).astype(np.float64)
            v /= np.linalg.norm(v)
            # estimate ||H|| via 5 power iters
            for _ in range(5):
                v_new = matvec(v)
                n_new = np.linalg.norm(v_new)
                if n_new < 1e-12:
                    break
                v = v_new / n_new
            h_norm_est = max(1.0, n_new)
            alpha = 2.0 * h_norm_est + 1.0
            v = np.random.RandomState(43).randn(n_free).astype(np.float64)
            v /= np.linalg.norm(v)
            for _ in range(150):
                v_new = alpha * v - matvec(v)
                n_new = np.linalg.norm(v_new)
                if n_new < 1e-12:
                    break
                v = v_new / n_new
            hv = matvec(v)
            lam = float(np.dot(v, hv) / (np.dot(v, v) + 1e-12))
            eigvals = np.array([lam])
            eigvecs = v.reshape(-1, 1)

    log(f"[saddle] eigsh done in {time.time() - t0:.1f}s; eigvals = {eigvals}")
    return eigvals, eigvecs


# ============================================================
# Adam descent helpers (matches E111 / V3Min behavior)
# ============================================================

def _make_optimizer(positions: torch.Tensor, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam([positions], lr=lr)


def _adam_descend(
    proxy: DiffProxyV3,
    init_positions: torch.Tensor,
    fixed_mask: torch.Tensor,
    *,
    num_steps: int,
    lr: float,
    gamma_start_frac: float,
    gamma_end_frac: float,
    overlap_lambda_start: float,
    overlap_lambda_end: float,
    overlap_ramp_pct: float,
    boundary_lambda: float,
    log: Callable[[str], None],
    log_every: int = 100,
    starting_step: int = 0,
    total_steps_for_anneal: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Run Adam for num_steps. Returns final (positions_detached, stats).

    starting_step / total_steps_for_anneal allow continuing the γ-anneal
    and overlap-λ ramp across stages (so resumption from a saddle uses
    the same γ_end / λ_end the original descent would have at this point).
    """
    if total_steps_for_anneal is None:
        total_steps_for_anneal = num_steps

    positions = init_positions.clone().detach().to(proxy.device).requires_grad_(True)
    optimizer = _make_optimizer(positions, lr)

    history = []
    last_loss = None
    last_parts = None

    for local_step in range(num_steps):
        global_step = starting_step + local_step
        t_anneal = min(1.0, global_step / max(1, total_steps_for_anneal - 1))
        gamma_frac = gamma_start_frac + t_anneal * (gamma_end_frac - gamma_start_frac)
        proxy.set_gamma_frac(gamma_frac)

        ramp_t = min(1.0, global_step / max(1, total_steps_for_anneal * overlap_ramp_pct))
        overlap_lambda = overlap_lambda_start + ramp_t * (overlap_lambda_end - overlap_lambda_start)

        optimizer.zero_grad()
        loss, parts = loss_with_penalty_v3(
            proxy, positions, overlap_lambda,
            include_congestion=True,
            boundary_lambda=boundary_lambda,
        )
        loss.backward()
        with torch.no_grad():
            if positions.grad is not None:
                positions.grad[fixed_mask] = 0.0
        optimizer.step()

        last_loss = float(loss.item())
        last_parts = parts
        if (local_step % log_every) == 0 or local_step == num_steps - 1:
            log(
                f"    step {global_step:4d}  loss={last_loss:.5f}  "
                f"smooth={parts['smooth_cost'].item():.5f}  "
                f"ovl_area={parts['overlap_area_raw'].item():.0f}  "
                f"γ={gamma_frac:.5f}  λ={overlap_lambda:.1f}"
            )
            history.append({
                "global_step": global_step,
                "loss": last_loss,
                "smooth": float(parts["smooth_cost"].item()),
                "wl": float(parts["wl"].item()),
                "density": float(parts["density"].item()),
                "cong": float(parts["cong"].item()),
                "overlap_area": float(parts["overlap_area_raw"].item()),
                "gamma_frac": gamma_frac,
                "overlap_lambda": overlap_lambda,
            })

    return positions.detach().cpu(), {
        "history": history,
        "final_loss": last_loss,
        "final_smooth": float(last_parts["smooth_cost"].item()) if last_parts else None,
        "final_overlap_area": float(last_parts["overlap_area_raw"].item()) if last_parts else None,
        "n_steps": num_steps,
    }


# ============================================================
# Main placer
# ============================================================

class ContinuousHessianSaddlePlacer:
    """V3 Adam descent + Hessian-eigvec saddle escape + CD polish.

    Default budget targets the same wall as V3Min ovl10 720s baseline so
    we can do a fair head-to-head: the saddle stages eat into the Adam
    budget rather than extending it.
    """

    def __init__(
        self,
        budget_seconds: float = 720.0,
        # Adam descent
        num_steps_phaseA: int = 300,
        num_steps_resume: int = 150,
        max_saddle_stages: int = 2,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_start: float = 0.0,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        boundary_lambda: float = 50.0,
        # Saddle escape
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),  # × cell-size
        eigval_tolerance: float = 1e-3,
        eigsh_maxiter: int = 300,
        # CD polish
        cd_polish_s: float = 300.0,
        cd_plateau_threshold: float = 0.001,
        # Misc
        init: str = "sdf",
        device: str = "cpu",
        rng_seed: int = 42,
        verbose: bool = True,
        log_every: int = 100,
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        trace_kwargs: Optional[Dict] = None,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps_phaseA = num_steps_phaseA
        self.num_steps_resume = num_steps_resume
        self.max_saddle_stages = max_saddle_stages
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_start = overlap_lambda_start
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.boundary_lambda = boundary_lambda
        self.eps_values = eps_values
        self.eigval_tolerance = eigval_tolerance
        self.eigsh_maxiter = eigsh_maxiter
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.init = init
        self.device = device
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.log_every = log_every
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.trace_kwargs = trace_kwargs

        self.last_run_stats: Optional[Dict] = None

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _init_positions(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        if self.init == "sdf":
            pos = sdf_init(benchmark)
            pos, n_iter = project_overlaps(pos, benchmark)
            self._log(f"  init=sdf: project_overlaps n_iter={n_iter}")
            return pos
        elif self.init == "center":
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            pos = torch.zeros(benchmark.num_macros, 2)
            pos[:, 0] = cw / 2.0
            pos[:, 1] = ch / 2.0
            mask = benchmark.macro_fixed.bool()
            pos[mask] = benchmark.macro_positions[mask]
            return pos
        else:
            raise ValueError(f"Unknown init: {self.init!r}")

    def _try_saddle_escape(
        self,
        proxy: DiffProxyV3,
        current_positions: torch.Tensor,
        fixed_mask: torch.Tensor,
        benchmark: Benchmark,
        *,
        stage: int,
        total_anneal_steps: int,
        starting_global_step: int,
        lr: float,
        deadline: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """One saddle escape stage.

        Returns (best_positions, stats). best_positions is the
        Adam-resumed candidate with lowest smooth proxy after resume; if
        nothing beat the current basin, returns current_positions
        unchanged with the original smooth proxy.
        """
        t_stage = time.time()
        movable_np = (~benchmark.macro_fixed.cpu().numpy())
        n_macros = current_positions.shape[0]

        # Smooth-proxy at the current basin.
        with torch.no_grad():
            basin_pos = current_positions.detach().to(proxy.device)
            basin_cost, _ = proxy.cost(basin_pos, include_congestion=True)
            basin_smooth = float(basin_cost.item())
        self._log(
            f"[stage {stage}] basin smooth_cost = {basin_smooth:.5f} "
            f"(remaining {deadline - time.time():.0f}s)"
        )

        # Hessian smallest-algebraic eigenvector.
        eigvals, eigvecs = find_softest_eigenvector_smooth(
            proxy, basin_pos, movable_np,
            k=1, tol=1e-3, maxiter=self.eigsh_maxiter,
            log=self._log,
        )
        if len(eigvals) == 0 or eigvecs.shape[1] == 0:
            self._log(f"[stage {stage}] no eigvecs returned; skipping")
            return current_positions, {
                "stage": stage,
                "skipped": "no_eigvec",
                "wall_s": time.time() - t_stage,
            }
        lam_min = float(eigvals[0])
        self._log(f"[stage {stage}] λ_min = {lam_min:.4e}")

        # Reconstruct full-state eigvec.
        n_dim = n_macros * 2
        mask_flat = np.zeros(n_dim, dtype=bool)
        for i, m in enumerate(movable_np):
            if bool(m):
                mask_flat[2 * i] = True
                mask_flat[2 * i + 1] = True
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = eigvecs[:, 0]
        v_2d = v_full.reshape(n_macros, 2)
        v_norm = np.linalg.norm(v_2d) + 1e-12
        v_unit = v_2d / v_norm

        # Define ε in MICRONS: ε × min(cell_w, cell_h).
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        cell_w = cw / gc
        cell_h = ch / gr
        cell_size = min(cell_w, cell_h)
        # v_unit is a unit vector in flat space; per-macro displacement
        # is small. Scale so the LARGEST per-macro displacement equals
        # ε × cell_size, matching the cascade saddle's intuition.
        per_macro_disp = np.linalg.norm(v_unit, axis=1)
        max_disp = per_macro_disp.max() + 1e-12

        attempts: List[Dict] = []
        best_smooth_after_resume = basin_smooth
        best_pos_after_resume = current_positions.detach().clone()

        basin_pos_np = current_positions.detach().cpu().numpy()
        fixed_mask_np = benchmark.macro_fixed.cpu().numpy()
        half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
        half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()

        deadline_reached = False
        for sign in (+1.0, -1.0):
            if deadline_reached:
                break
            for eps in self.eps_values:
                if time.time() > deadline - 30.0:
                    self._log(f"[stage {stage}] near deadline; skipping remaining (sign={sign} eps={eps})")
                    deadline_reached = True
                    break
                # Scale: ε × cell_size = max per-macro displacement.
                scale = (eps * cell_size) / max_disp
                disp = sign * scale * v_unit
                cand_np = basin_pos_np + disp
                # Clamp to canvas (matches V3 proxy clamping).
                cand_np[:, 0] = np.clip(cand_np[:, 0], half_w, cw - half_w)
                cand_np[:, 1] = np.clip(cand_np[:, 1], half_h, ch - half_h)
                # Restore fixed positions exactly.
                cand_np[fixed_mask_np] = basin_pos_np[fixed_mask_np]
                cand_t = torch.tensor(cand_np, dtype=torch.float32)

                # Resume Adam from cand_t for num_steps_resume.
                resume_t0 = time.time()
                resumed, resume_stats = _adam_descend(
                    proxy, cand_t, fixed_mask,
                    num_steps=self.num_steps_resume,
                    lr=lr,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_start=self.overlap_lambda_start,
                    overlap_lambda_end=self.overlap_lambda_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    boundary_lambda=self.boundary_lambda,
                    log=lambda s: None,  # silent inside the resume
                    log_every=10_000,
                    starting_step=starting_global_step,
                    total_steps_for_anneal=total_anneal_steps,
                )
                with torch.no_grad():
                    sc, _ = proxy.cost(resumed.to(proxy.device), include_congestion=True)
                    resumed_smooth = float(sc.item())
                resume_wall = time.time() - resume_t0

                is_best = resumed_smooth < best_smooth_after_resume - 1e-7
                self._log(
                    f"[stage {stage}] sign={sign:+.0f} eps={eps:.1f} → "
                    f"smooth={resumed_smooth:.5f} "
                    f"(Δ vs basin={resumed_smooth - basin_smooth:+.5f}) "
                    f"wall={resume_wall:.0f}s{' NEW BEST' if is_best else ''}"
                )
                attempts.append({
                    "sign": sign,
                    "eps": eps,
                    "smooth_after_resume": resumed_smooth,
                    "delta_vs_basin": resumed_smooth - basin_smooth,
                    "wall_s": resume_wall,
                    "is_best": is_best,
                })
                if is_best:
                    best_smooth_after_resume = resumed_smooth
                    best_pos_after_resume = resumed.clone()

        return best_pos_after_resume, {
            "stage": stage,
            "lam_min": lam_min,
            "basin_smooth": basin_smooth,
            "best_smooth_after_resume": best_smooth_after_resume,
            "improvement": basin_smooth - best_smooth_after_resume,
            "attempts": attempts,
            "wall_s": time.time() - t_stage,
        }

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else float("inf")
        self._log(f"=== ContinuousHessianSaddlePlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds:.0f}s phaseA_steps={self.num_steps_phaseA} "
            f"resume_steps={self.num_steps_resume} max_saddle={self.max_saddle_stages} "
            f"eps_values={self.eps_values}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        device = torch.device(self.device)
        proxy = DiffProxyV3(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
        )

        init_pos = self._init_positions(benchmark, plc=plc)
        fixed_mask = benchmark.macro_fixed.bool().to(device)

        # Anneal across (phaseA + max_saddle_stages * resume_steps) so γ
        # reaches gamma_end_frac at the very end of the last Adam stage.
        total_anneal_steps = (
            self.num_steps_phaseA
            + self.max_saddle_stages * self.num_steps_resume
        )

        # ----- PHASE A: settle into a basin -----
        self._log(f"\n--- Phase A: Adam descent for {self.num_steps_phaseA} steps ---")
        phaseA_pos, phaseA_stats = _adam_descend(
            proxy, init_pos, fixed_mask,
            num_steps=self.num_steps_phaseA,
            lr=lr,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_start=self.overlap_lambda_start,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            boundary_lambda=self.boundary_lambda,
            log=self._log,
            log_every=self.log_every,
            starting_step=0,
            total_steps_for_anneal=total_anneal_steps,
        )
        self._log(
            f"  Phase A done: smooth={phaseA_stats['final_smooth']:.5f} "
            f"ovl_area={phaseA_stats['final_overlap_area']:.0f} "
            f"wall={time.time() - t0:.0f}s"
        )

        current_pos = phaseA_pos
        with torch.no_grad():
            sc, _ = proxy.cost(current_pos.to(proxy.device), include_congestion=True)
            current_smooth = float(sc.item())
        starting_global_step = self.num_steps_phaseA

        # ----- PHASE B & C: saddle escapes -----
        saddle_stage_stats: List[Dict] = []
        for stage_idx in range(self.max_saddle_stages):
            remaining = deadline - time.time()
            # Budget for one full stage ~= eigsh + 6 * (resume Adam wall).
            # Bail if remaining < (saddle_polish + CD polish reserve).
            min_required = (
                self.cd_polish_s
                + 60.0  # eigsh
                + 6 * 30.0  # 6 perturb-and-resume attempts at ~30s each, rough
            )
            if remaining < min_required:
                self._log(
                    f"\n[stage {stage_idx + 1}] remaining {remaining:.0f}s "
                    f"< min_required {min_required:.0f}s; skipping further saddles"
                )
                break

            self._log(f"\n--- Stage {stage_idx + 1}: Hessian saddle escape ---")
            cand_pos, stage_stats = self._try_saddle_escape(
                proxy, current_pos, fixed_mask, benchmark,
                stage=stage_idx + 1,
                total_anneal_steps=total_anneal_steps,
                starting_global_step=starting_global_step,
                lr=lr,
                deadline=deadline,
            )
            saddle_stage_stats.append(stage_stats)
            cand_smooth = stage_stats.get("best_smooth_after_resume", current_smooth)

            if cand_smooth < current_smooth - 1e-7:
                self._log(
                    f"[stage {stage_idx + 1}] ACCEPT: smooth "
                    f"{current_smooth:.5f} → {cand_smooth:.5f} "
                    f"(Δ={cand_smooth - current_smooth:+.5f})"
                )
                current_pos = cand_pos
                current_smooth = cand_smooth
                starting_global_step += self.num_steps_resume
            else:
                self._log(
                    f"[stage {stage_idx + 1}] REJECT: no improvement "
                    f"(best resume {cand_smooth:.5f} >= basin {current_smooth:.5f}); "
                    f"stopping saddle cascade"
                )
                break

        # ----- PHASE D: legalize + CD polish -----
        self._log(f"\n--- Phase D: legalize + CD polish ---")
        leg_t0 = time.time()
        legal_pos, leg_stats = greedy_macro_legalize(
            current_pos, benchmark,
            search_radius_steps=self.legalize_radius_steps,
            step_size_frac=self.legalize_step_frac,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(
                f"  greedy_legalize left {ovl} overlaps "
                f"(moved={leg_stats.get('n_moved')} failed={leg_stats.get('n_failed')}); "
                f"running project_overlaps"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        self._log(f"  legalize done: ovl={ovl} wall={time.time() - leg_t0:.0f}s")

        if ovl > 0:
            raise RuntimeError(f"E120 produced {ovl} overlaps after legalize+project")

        # CD polish
        remaining = max(30.0, deadline - time.time() - 5.0)
        cd_budget = min(remaining, self.cd_polish_s)
        self._log(f"  CD polish budget = {cd_budget:.0f}s")
        evaluator = IncrementalProxyEvaluator(benchmark, plc, legal_pos)
        movable_idx = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable_idx,
            min_time_s=cd_budget * 0.5,
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time() - t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"E120 final has {final_ovl} overlaps")

        # Save run stats
        self.last_run_stats = {
            "bench": benchmark.name,
            "phaseA_stats": {k: v for k, v in phaseA_stats.items() if k != "history"},
            "saddle_stage_stats": saddle_stage_stats,
            "final_proxy": final_proxy,
            "final_overlap": final_ovl,
            "total_wall_s": time.time() - t0,
        }
        return final


def main_smoke():
    """Smoke test on ibm17.

    Run:
        cd <repo>
        uv run python experiments/E120_continuous_saddle/code/continuous_saddle_placer.py
    """
    BENCH = os.environ.get("E120_BENCH", "ibm17")
    BUDGET = float(os.environ.get("E120_BUDGET", "720"))
    bench_dir = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"Loaded {BENCH}: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}, "
        f"canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
        flush=True,
    )

    placer = ContinuousHessianSaddlePlacer(
        budget_seconds=BUDGET,
        num_steps_phaseA=300,
        num_steps_resume=150,
        max_saddle_stages=2,
        overlap_lambda_end=10.0,
        cd_polish_s=300.0,
        verbose=True,
        log_every=100,
    )
    t0 = time.time()
    final = placer.place(benchmark)
    print(
        f"\n=== {BENCH} DONE ===\n"
        f"  final_proxy: {placer.last_run_stats['final_proxy']:.5f}\n"
        f"  wall: {time.time() - t0:.0f}s\n"
        f"  reference V3Min ovl10 720s baseline: 1.20\n",
        flush=True,
    )


if __name__ == "__main__":
    main_smoke()
