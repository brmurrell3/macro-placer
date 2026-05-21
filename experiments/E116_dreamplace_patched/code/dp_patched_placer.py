"""E116 — DREAMPlace-style optimizer with challenge-proxy loss.

Pure-PyTorch port of DREAMPlace's Nesterov + BB descent, plugged into
our challenge proxy (LSE-HPWL + 0.5·top-K-density + 0.5·PerNetTrace
congestion). Replicates the optimization machinery the top-2 leader
team uses, but on the EXACT challenge proxy instead of WL + eDensity.

Pipeline:
  1. SDF init → project_overlaps for a legal start
  2. Nesterov + BB on challenge proxy
  3. Density weight ramping (Lgamma loop), small overlap penalty ramp
  4. Detach, run greedy_macro_legalize
  5. CD polish (run_cd_adaptive, budget-bounded)

Hypothesis: Nesterov+BB > Adam on this loss surface, giving
ibm01 < 0.85 (current 0.843) and ibm17 < 1.20 (current 1.200).

If smooth basin doesn't beat E111 V3, kill.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from diff_proxy_v2 import DiffProxyV2  # noqa: E402
from diff_proxy import _grid_density, _lse_hpwl  # noqa: E402
from macro_legalizer import greedy_macro_legalize  # noqa: E402
from per_net_trace_proxy import PerNetTraceCongestion  # noqa: E402

from nesterov_optimizer import DreamPlaceNesterov  # noqa: E402


# ---------------------------------------------------------------------------
# Challenge-proxy loss
# ---------------------------------------------------------------------------


class ChallengeProxyV3:
    """Differentiable challenge proxy: LSE-HPWL + 0.5·density + 0.5·cong.

    Same as DiffProxyV3 in E111 but factored out so we don't pay the
    overhead of building two PerNetTraceCongestion objects.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        plc,
        device: str = "cpu",
        gamma_frac: float = 5e-3,
        trace_kwargs: Optional[Dict] = None,
    ):
        self.diff = DiffProxyV2(benchmark, plc, device=device, gamma_frac=gamma_frac)
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.num_macros = int(benchmark.num_macros)
        self.num_hard = int(benchmark.num_hard_macros)
        self.macro_sizes = benchmark.macro_sizes.to(self.device)
        self.half_sizes = self.macro_sizes / 2.0
        self.macro_areas = self.macro_sizes[:, 0] * self.macro_sizes[:, 1]
        # Per-macro pin count (DREAMPlace's sum_pin_weights_in_nodes for our
        # graph: count pins per macro across nets, all net weights = 1
        # in our benchmark since we don't track per-net weights here).
        nd = self.diff.net_data
        pin_macro_idx = nd.pin_macro_idx.to(self.device).long()  # [n_nets, max_pins]
        mask = nd.mask.to(self.device)
        weights = nd.weights.to(self.device).to(torch.float32)
        # Per-pin weight = net's weight, then scatter-add by macro index.
        pin_w = weights.unsqueeze(1).expand_as(mask).clone()
        pin_w[~mask] = 0.0
        # macro indices in pin_macro_idx are [0, num_macros]; index num_macros = port pseudo.
        # We only count macros < num_macros.
        pin_macro_idx_clamped = pin_macro_idx.clone()
        pin_macro_idx_clamped[pin_macro_idx >= self.num_macros] = 0
        pin_w[pin_macro_idx >= self.num_macros] = 0.0
        sum_pin_w = torch.zeros(self.num_macros, device=self.device, dtype=torch.float32)
        sum_pin_w.scatter_add_(0, pin_macro_idx_clamped.reshape(-1), pin_w.reshape(-1))
        self.macro_pin_weights = sum_pin_w
        self.trace = PerNetTraceCongestion(
            benchmark, plc, device=device, **(trace_kwargs or {})
        )

    def precondition_gradient(
        self, grad: torch.Tensor, density_weight: float, alpha: float = 1.0
    ) -> torch.Tensor:
        """DREAMPlace-style gradient preconditioning.

        precond[i] = pin_weight[i] + alpha * density_weight * area[i]
        grad[i] /= max(precond[i], 1.0)
        """
        with torch.no_grad():
            precond = self.macro_pin_weights + alpha * density_weight * self.macro_areas
            precond.clamp_(min=1.0)
            return grad / precond.unsqueeze(1)

    def set_gamma_frac(self, frac: float) -> None:
        self.diff.set_gamma_frac(frac)

    def _clamp(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                positions[:, 0].clamp(self.half_sizes[:, 0], self.cw - self.half_sizes[:, 0]),
                positions[:, 1].clamp(self.half_sizes[:, 1], self.ch - self.half_sizes[:, 1]),
            ],
            dim=1,
        )

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True):
        clamped = self._clamp(positions)
        wl = _lse_hpwl(clamped, self.diff.net_data, self.diff.port_base, self.diff.gamma) / self.diff.wl_norm
        density = _grid_density(
            clamped,
            self.diff.sizes,
            self.diff.cell_x_min,
            self.diff.cell_x_max,
            self.diff.cell_y_min,
            self.diff.cell_y_max,
            self.diff.cell_area,
            self.diff.grid_rows,
            self.diff.grid_cols,
        )
        if include_congestion:
            cong = self.trace.compute_congestion(clamped)
        else:
            cong = torch.tensor(0.0, device=self.device)
        # WL + 0.5*density + 0.5*cong, where _grid_density already
        # returns 0.5 * top_10% (mirroring canonical get_density_cost),
        # so the formula is WL + 0.5 * (0.5*top10%) + 0.5*cong = canonical.
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }

    def overlap_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        return self.diff.overlap_penalty(positions)

    def boundary_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        return self.diff.out_of_canvas_penalty(positions)


# ---------------------------------------------------------------------------
# Placer
# ---------------------------------------------------------------------------


class DreamPlacePatchedPlacer:
    """DREAMPlace-style Nesterov + BB on challenge proxy.

    `place(benchmark)` returns a [num_macros, 2] tensor of centers.

    Pipeline:
      SDF init → Nesterov descent (density-weight ramp) → greedy
      legalize → CD polish.
    """

    def __init__(
        self,
        # Descent budget
        num_steps: int = 500,
        # Precondition gradient (DP-style: divide by pin_weight + α·dw·area)
        use_preconditioner: bool = False,
        precond_alpha: float = 1.0,
        # Initial learning rate as fraction of canvas width
        lr_frac: float = 0.005,
        # γ-anneal trajectory (LSE smoothing temperature)
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        # Overlap penalty ramp (in units of normalized canvas-area)
        overlap_lambda_start: float = 0.0,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        # Boundary keep-in penalty
        boundary_lambda: float = 50.0,
        # Density-weight ramping (DREAMPlace-style outer loop)
        density_weight_start: float = 1.0,
        density_weight_end: float = 1.0,
        # Optimizer flavor: "nesterov_bb" (DP-style), "sgd_momentum", "adam"
        optimizer: str = "nesterov_bb",
        sgd_momentum: float = 0.9,
        # CD polish
        cd_budget_s: float = 720.0,
        cd_min_time_s: float = 240.0,
        # Init flavor
        init: str = "sdf",
        # PerNetTraceCongestion knobs (passed through)
        trace_kwargs: Optional[Dict] = None,
        # Misc
        device: str = "cpu",
        rng_seed: int = 42,
        verbose: bool = True,
        log_every: int = 50,
        # Wall budget per benchmark (matches submissions/ defaults)
        budget_seconds: float = 720.0,
    ):
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_start = overlap_lambda_start
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.boundary_lambda = boundary_lambda
        self.density_weight_start = density_weight_start
        self.density_weight_end = density_weight_end
        self.use_preconditioner = use_preconditioner
        self.precond_alpha = precond_alpha
        self.optimizer = optimizer
        self.sgd_momentum = sgd_momentum
        self.cd_budget_s = cd_budget_s
        self.cd_min_time_s = cd_min_time_s
        self.init = init
        self.trace_kwargs = trace_kwargs
        self.device = device
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.log_every = log_every
        self.budget_seconds = budget_seconds

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _init_positions(self, benchmark: Benchmark) -> torch.Tensor:
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

    def descend(
        self, benchmark: Benchmark, plc,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        init_pos = self._init_positions(benchmark)
        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        device = torch.device(self.device)
        proxy = ChallengeProxyV3(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
        )
        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)
        canvas_area = proxy.cw * proxy.ch

        # State trackers shared between obj_and_grad_fn (called multiple
        # times per step by BB optimizer) and the outer descent loop.
        state = {
            "gamma_frac": self.gamma_start_frac,
            "overlap_lambda": self.overlap_lambda_start,
            "density_weight": self.density_weight_start,
            "boundary_lambda": self.boundary_lambda,
            "last_parts": None,
            "last_obj": None,
        }

        def constraint_fn(p: torch.Tensor) -> None:
            """Project fixed macros back to their original positions.

            Called by Nesterov after each look-ahead update. The fixed
            macros must never drift.
            """
            with torch.no_grad():
                p.data[fixed_mask] = init_pos.to(device)[fixed_mask]

        def obj_and_grad_fn(p: torch.Tensor):
            """Compute loss and gradient at p.

            Nesterov's `step_bb` calls this multiple times per step (at
            v_k, v_{k-1}, v_{k+1}). Each call uses CURRENT state values.
            """
            # Fresh leaf with autograd. Nesterov passes Variables.
            if p.grad is not None:
                p.grad = None
            if not p.requires_grad:
                p.requires_grad_(True)
            proxy.set_gamma_frac(state["gamma_frac"])

            smooth_cost, parts = proxy.cost(p, include_congestion=True)
            penalty = proxy.overlap_penalty(p)
            boundary = proxy.boundary_penalty(p)
            penalty_norm = penalty / canvas_area
            boundary_norm = boundary / canvas_area
            # Density weight multiplies the density portion. Default 1.0
            # keeps the canonical 0.5 coefficient; setting <1 underweights
            # density (more WL-focused), >1 overweights density.
            total = (
                state["density_weight"] * 0.5 * parts["wl"]
                + 0.5 * parts["density"]
                + 0.5 * parts["cong"]
                + state["overlap_lambda"] * penalty_norm
                + state["boundary_lambda"] * boundary_norm
            )
            # Recompute total with original WL since parts['wl'] is detached.
            # Actually, we want the smooth_cost from proxy.cost (which has
            # autograd graph), not a sum of detached parts.
            total = smooth_cost \
                + state["overlap_lambda"] * penalty_norm \
                + state["boundary_lambda"] * boundary_norm

            total.backward()
            grad = p.grad.data.clone()
            # DP-style preconditioning
            if self.use_preconditioner:
                grad = proxy.precondition_gradient(
                    grad, state["density_weight"], alpha=self.precond_alpha
                )
            # Zero gradient on fixed macros
            grad[fixed_mask] = 0.0

            state["last_parts"] = parts
            state["last_obj"] = float(total.item())
            state["last_penalty"] = float(penalty.item())
            state["last_boundary"] = float(boundary.item())
            return total.detach(), grad

        if self.optimizer == "nesterov_bb":
            optimizer = DreamPlaceNesterov(
                [positions], lr=lr,
                obj_and_grad_fn=obj_and_grad_fn,
                constraint_fn=constraint_fn,
            )
            need_obj_call = False  # optimizer calls it internally
        elif self.optimizer == "sgd_momentum":
            optimizer = torch.optim.SGD([positions], lr=lr,
                                         momentum=self.sgd_momentum,
                                         nesterov=True)
            need_obj_call = True
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam([positions], lr=lr)
            need_obj_call = True
        elif self.optimizer == "adabelief":
            import torch_optimizer
            optimizer = torch_optimizer.AdaBelief([positions], lr=lr, eps=1e-8, betas=(0.9, 0.999))
            need_obj_call = True
        elif self.optimizer == "radam":
            import torch_optimizer
            optimizer = torch_optimizer.RAdam([positions], lr=lr)
            need_obj_call = True
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW([positions], lr=lr, weight_decay=0.0)
            need_obj_call = True
        elif self.optimizer == "nadam":
            optimizer = torch.optim.NAdam([positions], lr=lr)
            need_obj_call = True
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Run num_steps Nesterov steps. State (gamma, overlap_lambda,
        # density_weight) is interpolated linearly over [0, num_steps].
        history = []
        deadline = t0 + max(1.0, self.budget_seconds - self.cd_budget_s - 30.0)
        for step in range(self.num_steps):
            t = step / max(1, self.num_steps - 1)
            state["gamma_frac"] = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
            ramp_t = min(1.0, step / max(1, self.num_steps * self.overlap_ramp_pct))
            state["overlap_lambda"] = self.overlap_lambda_start + ramp_t * (
                self.overlap_lambda_end - self.overlap_lambda_start
            )
            state["density_weight"] = self.density_weight_start + t * (
                self.density_weight_end - self.density_weight_start
            )

            if need_obj_call:
                # SGD/Adam: call obj_and_grad_fn ourselves to populate gradient
                optimizer.zero_grad()
                _, grad = obj_and_grad_fn(positions)
                with torch.no_grad():
                    positions.grad = grad
                optimizer.step()
                # Constraint: keep fixed macros pinned
                constraint_fn(positions)
            else:
                optimizer.step()

            if step % self.log_every == 0 or step == self.num_steps - 1:
                lp = state["last_parts"]
                self._log(
                    f"  step {step:4d}  obj={state['last_obj']:.5f}  "
                    f"wl={lp['wl'].item():.4f}  d={lp['density'].item():.4f}  "
                    f"c={lp['cong'].item():.4f}  "
                    f"ovl_area={state['last_penalty']:.0f}  "
                    f"γ={state['gamma_frac']:.4f}  "
                    f"λ_ovl={state['overlap_lambda']:.1f}  "
                    f"dw={state['density_weight']:.2f}"
                )
                history.append({
                    "step": step,
                    "obj": state["last_obj"],
                    "wl": lp["wl"].item(),
                    "density": lp["density"].item(),
                    "cong": lp["cong"].item(),
                    "overlap_area": state["last_penalty"],
                    "boundary": state["last_boundary"],
                    "gamma_frac": state["gamma_frac"],
                    "overlap_lambda": state["overlap_lambda"],
                    "density_weight": state["density_weight"],
                })

            if time.time() > deadline:
                self._log(f"  early stop: deadline reached at step {step}")
                break

        positions_final = positions.detach().cpu()
        stats = {
            "history": history,
            "final_obj": state["last_obj"],
            "final_overlap_area": state["last_penalty"],
            "descend_wall_s": time.time() - t0,
        }
        return positions_final, stats

    def legalize(
        self, positions: torch.Tensor, benchmark: Benchmark,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        legal_pos, leg_stats = greedy_macro_legalize(
            positions, benchmark,
            search_radius_steps=80,
            step_size_frac=0.02,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(
                f"  greedy_legalize left {ovl} overlaps; running project_overlaps"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        leg_stats["final_overlaps"] = ovl
        leg_stats["legalize_wall_s"] = time.time() - t0
        return legal_pos, leg_stats

    def polish(
        self, positions: torch.Tensor, benchmark: Benchmark, plc, budget_s: float,
    ) -> Tuple[torch.Tensor, Dict]:
        from macro_place.cd_core import run_cd_adaptive
        from macro_place.incremental_evaluator import IncrementalProxyEvaluator

        t0 = time.time()
        ev = IncrementalProxyEvaluator(benchmark, plc, positions.clone())
        n_hard = benchmark.num_hard_macros
        fixed = benchmark.macro_fixed.cpu().numpy()
        movable = [i for i in range(n_hard) if not bool(fixed[i])]
        run_cd_adaptive(
            ev, benchmark, plc, movable,
            min_time_s=self.cd_min_time_s, hard_cap_s=budget_s,
            patience=3, plateau_threshold=0.001, log_fn=None,
        )
        polished = ev.placement.detach().clone().to(torch.float32)
        polished, _ = project_overlaps(polished, benchmark)
        stats = {
            "polish_wall_s": time.time() - t0,
            "final_overlaps": compute_overlap_metrics(polished, benchmark)["overlap_count"],
        }
        return polished, stats

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== DreamPlacePatchedPlacer ({benchmark.name}) ===")
        self._log(
            f"  config: steps={self.num_steps} lr_frac={self.lr_frac} "
            f"γ={self.gamma_start_frac}→{self.gamma_end_frac} "
            f"λ_ovl={self.overlap_lambda_start}→{self.overlap_lambda_end} "
            f"density_weight={self.density_weight_start}→{self.density_weight_end} "
            f"init={self.init} budget={self.budget_seconds:.0f}s"
        )
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        positions, descend_stats = self.descend(benchmark, plc)
        self._log(
            f"  descent: obj={descend_stats['final_obj']:.5f} "
            f"ovl_area={descend_stats['final_overlap_area']:.0f} "
            f"wall={descend_stats['descend_wall_s']:.1f}s"
        )

        legal_pos, leg_stats = self.legalize(positions, benchmark)
        raw_proxy = float(compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  legalize: raw_proxy={raw_proxy:.5f} "
            f"ovl={leg_stats['final_overlaps']} "
            f"n_moved={leg_stats.get('n_moved')} "
            f"max_disp={leg_stats.get('max_displacement', -1):.1f} "
            f"wall={leg_stats['legalize_wall_s']:.1f}s"
        )

        # Compute remaining budget for polish
        elapsed = time.time() - t0
        polish_budget = max(60.0, self.budget_seconds - elapsed - 10.0)
        polish_budget = min(polish_budget, self.cd_budget_s)

        polished, polish_stats = self.polish(legal_pos, benchmark, plc, polish_budget)
        final_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        self._log(
            f"  polish: final_proxy={final_proxy:.5f} "
            f"ovl={polish_stats['final_overlaps']} "
            f"wall={polish_stats['polish_wall_s']:.1f}s"
        )
        self._log(
            f"  TOTAL: {final_proxy:.5f} (raw {raw_proxy:.5f}) "
            f"wall={time.time()-t0:.1f}s"
        )
        return polished


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="?", default="ibm01")
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--budget", type=float, default=720.0)
    ap.add_argument("--cd-budget", type=float, default=400.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench_dir = find_benchmark_dir(args.bench)
    bench, plc = load_benchmark_from_dir(str(bench_dir))

    placer = DreamPlacePatchedPlacer(
        num_steps=args.num_steps,
        budget_seconds=args.budget,
        cd_budget_s=args.cd_budget,
        device=args.device,
    )
    placement = placer.place(bench)
    proxy = compute_proxy_cost(placement, bench, plc)
    print(f"\nFINAL: {args.bench} proxy={proxy['proxy_cost']:.5f} "
          f"wl={proxy['wirelength_cost']:.4f} d={proxy['density_cost']:.4f} "
          f"c={proxy['congestion_cost']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "bench": args.bench,
            "proxy": float(proxy["proxy_cost"]),
            "wl": float(proxy["wirelength_cost"]),
            "density": float(proxy["density_cost"]),
            "cong": float(proxy["congestion_cost"]),
            "overlaps": compute_overlap_metrics(placement, bench)["overlap_count"],
        }, indent=2))
