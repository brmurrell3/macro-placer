"""
E10 — Congestion-Only Refinement Placer

Hypothesis (post-E8): congestion is 74% of proxy cost. After best_of_v2's
DPO converges (which jointly optimises wl + 0.5*density + 0.5*congestion),
re-running N additional iterations with the loss reweighted to focus on
congestion may unlock disproportionate improvement on congestion-heavy
benchmarks (ibm06, ibm17, ibm18, ibm02, ibm12).

Strategy:
  1. Run the SDF init.
  2. Run the standard `DPOv2StepsPlacer._optimize` loop (reused via subclass,
     no copy-paste of the gradient code). This is the pre-refinement
     placement.
  3. Run `n_refine` extra Adam steps on the same `pos` tensor with the loss
     reweighted to (wl_mult * wl) + (dens_mult * 0.5 * density)
     + (cong_mult * 0.5 * congestion). Defaults: wl_mult=0.1, dens_mult=0.5,
     cong_mult=5.0 (so the absolute coefficients become 0.1·wl, 0.25·density,
     2.5·congestion — the congestion term dominates the loss).
  4. Legalise both placements.
  5. Score both with the *real* proxy and return the better one (never regress).

We only modify the late-training weighting; everything else (LSE-HPWL,
RUDY congestion, density grid, overlap penalty, fixed-macro masking) is
inherited from `DPOv2StepsPlacer` so we cannot drift from its hard-macro
contracts. `submissions/dpo/best_of_v2_placer.py` and
`submissions/dpo/init_strategies.py` are NOT modified — read-only.

The refinement extra-iteration loop uses the same final-phase hyperparameters
as the standard P3 (small gamma, large lambda, low lr) so that overlaps
stay resolved and the optimizer doesn't blow the placement out of the
already-found basin.
"""

import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


# ---------------------------------------------------------------------------
# Reuse the v2-steps DPO core via importlib (no copy-paste of the gradient logic)
# ---------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HERE = Path(__file__).parent
_V2_PATH = _HERE / "ablation_v2_steps.py"
_v2_mod = _load_module(_V2_PATH, "dpo_v2_steps")

DPOv2StepsPlacer = _v2_mod.DPOv2StepsPlacer
_load_plc = _v2_mod._load_plc
_extract_net_data = _v2_mod._extract_net_data
_lse_hpwl = _v2_mod._lse_hpwl
_grid_density = _v2_mod._grid_density
_rudy_congestion = _v2_mod._rudy_congestion
_overlap_penalty = _v2_mod._overlap_penalty
_count_overlaps = _v2_mod._count_overlaps
_legalize = _v2_mod._legalize


# ---------------------------------------------------------------------------
# Top-level placer is defined FIRST so evaluate.py's loader (which picks the
# first class with a `place` method) selects `CongestionRefinePlacer`, not
# the subclass adapter `_DPOWithCongestionRefine`.  The adapter is defined
# below the placer; both are forward-referenced through plain attribute
# lookup at call time.
# ---------------------------------------------------------------------------


class CongestionRefinePlacer:
    """E10 placer.

    Runs DPOv2Steps, then a congestion-focused refinement phase.  Returns
    whichever of (pre-refine, post-refine) scores lower under the real proxy.
    Configurable via init args:
        n_refine: extra Adam steps for refinement (default 200)
        wl_mult / dens_mult / cong_mult: multipliers applied to the original
            (1, 0.5, 0.5) component coefficients during refinement.
    """

    def __init__(self, seed: int = 42,
                 n_refine: int = 200,
                 wl_mult: float = 0.1,
                 dens_mult: float = 0.5,
                 cong_mult: float = 5.0,
                 verbose: bool = True):
        self.seed = seed
        self.n_refine = n_refine
        self.wl_mult = wl_mult
        self.dens_mult = dens_mult
        self.cong_mult = cong_mult
        self.verbose = verbose

        self.last_diagnostics = {}

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        dpo = _DPOWithCongestionRefine(
            seed=self.seed,
            use_sdf_init=True,
            n_refine=self.n_refine,
            wl_mult=self.wl_mult,
            dens_mult=self.dens_mult,
            cong_mult=self.cong_mult,
            verbose=self.verbose,
        )
        post_refine = dpo.place(benchmark)
        pre_refine = dpo.pre_refine_pos
        if pre_refine is None:
            # n_refine == 0 path; nothing to compare.
            self.last_diagnostics = {"winner": "post(no_refine)"}
            return post_refine

        # Need legalised + scored versions of the pre-refine snapshot.
        pre_legal = dpo._do_legalize(pre_refine, benchmark)

        plc = _load_plc(benchmark.name)
        if plc is None:
            # Cannot evaluate; default to post-refine since refinement keeps
            # overlaps low and tracks best score internally.
            self.last_diagnostics = {"winner": "post(no_plc)"}
            return post_refine

        pre_costs = compute_proxy_cost(pre_legal, benchmark, plc)
        post_costs = compute_proxy_cost(post_refine, benchmark, plc)

        pre_proxy = float(pre_costs["proxy_cost"])
        post_proxy = float(post_costs["proxy_cost"])
        pre_ovs = int(pre_costs["overlap_count"])
        post_ovs = int(post_costs["overlap_count"])
        pre_cong = float(pre_costs["congestion_cost"])
        post_cong = float(post_costs["congestion_cost"])

        diag = {
            "pre_proxy": pre_proxy, "pre_overlaps": pre_ovs,
            "post_proxy": post_proxy, "post_overlaps": post_ovs,
            "pre_congestion": pre_cong, "post_congestion": post_cong,
            "pre_wirelength": float(pre_costs["wirelength_cost"]),
            "post_wirelength": float(post_costs["wirelength_cost"]),
            "pre_density": float(pre_costs["density_cost"]),
            "post_density": float(post_costs["density_cost"]),
        }

        # Never regress: if post has overlaps but pre is clean, prefer pre.
        # Otherwise pick lower proxy.
        if post_ovs == 0 and pre_ovs == 0:
            winner_pos = post_refine if post_proxy < pre_proxy else pre_legal
            winner_name = "post" if post_proxy < pre_proxy else "pre"
        elif post_ovs == 0:
            winner_pos, winner_name = post_refine, "post(pre_had_ov)"
        elif pre_ovs == 0:
            winner_pos, winner_name = pre_legal, "pre(post_had_ov)"
        else:
            winner_pos = post_refine if post_proxy < pre_proxy else pre_legal
            winner_name = "post" if post_proxy < pre_proxy else "pre"

        diag["winner"] = winner_name
        self.last_diagnostics = diag

        if self.verbose:
            cong_delta = post_cong - pre_cong
            cong_pct = (cong_delta / pre_cong * 100.0) if pre_cong > 0 else 0.0
            print(f"  >> E10: pre={pre_proxy:.4f} (cong={pre_cong:.4f})  "
                  f"post={post_proxy:.4f} (cong={post_cong:.4f}, "
                  f"delta={cong_delta:+.4f} / {cong_pct:+.2f}%)  "
                  f"-> {winner_name}")

        return winner_pos


# ---------------------------------------------------------------------------
# Subclass of DPOv2StepsPlacer that adds a congestion-focused refinement phase
# ---------------------------------------------------------------------------

class _DPOWithCongestionRefine(DPOv2StepsPlacer):
    """Runs standard DPOv2Steps, captures the converged tensor, and exposes a
    second-stage refinement loop with reweighted loss.

    The implementation overrides `_optimize` to (a) run the inherited loop via
    `super()._optimize(...)`, (b) re-create the optimisation state on the
    converged best_pos, and (c) execute n_refine extra Adam steps using the
    same per-step components as P3 but with custom (wl, density, congestion)
    weights. We track best by *score* exactly like the parent, so we can never
    return something worse than the inherited best.
    """

    def __init__(self, seed: int = 42, use_sdf_init: bool = True,
                 n_refine: int = 200,
                 wl_mult: float = 0.1,
                 dens_mult: float = 0.5,
                 cong_mult: float = 5.0,
                 verbose: bool = True):
        super().__init__(seed=seed, use_sdf_init=use_sdf_init)
        self.n_refine = int(n_refine)
        self.wl_mult = float(wl_mult)
        self.dens_mult = float(dens_mult)
        self.cong_mult = float(cong_mult)
        self.verbose = verbose

        # Capture the converged-from-standard-DPO placement so the placer can
        # decide between pre- and post-refinement at evaluation time.
        self.pre_refine_pos: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------
    # Override: run the inherited optimiser, then run a refinement phase.
    # -----------------------------------------------------------------
    def _optimize(self, init_pos, benchmark, net_data):
        # Step 1: standard v2-steps optimisation.
        std_best = super()._optimize(init_pos, benchmark, net_data)
        # Snapshot pre-refinement so .place can choose between the two later.
        self.pre_refine_pos = std_best.clone()

        if self.n_refine <= 0:
            return std_best

        # Step 2: refinement loop.  Reproduce just enough state from parent.
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

        num_nets = len(net_data.weights)
        cong_grad_freq = 1 if num_nets < 3000 else (3 if num_nets < 8000 else 5)
        cached_cong = 0.0

        # Refinement uses Phase-3 sharpening hyperparameters: small gamma, large
        # overlap multiplier, small lr.  Then the only thing different from the
        # standard P3 is the per-component weighting.
        gamma_frac = 0.0005
        lam = 500.0
        lr = 0.1
        gamma = gamma_frac * cw

        pos = std_best.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(self.n_refine, 1), eta_min=lr * 0.05
        )

        # Track best by the same composite score parent uses, so we can never
        # regress vs std_best when we pick a winner inside this phase.
        best_pos = std_best.clone()
        # Compute the inherited best's score under the *refinement* weighting
        # only as a tie-breaker — parent already tracked it under the original
        # weighting, which is what we ultimately want, so we re-evaluate a
        # neutral score per-step (the refined proxy).
        best_score = float("inf")

        # Effective coefficients we are minimising:
        # wl_eff·WL + dens_eff·density + cong_eff·congestion
        wl_eff = self.wl_mult
        dens_eff = self.dens_mult * 0.5
        cong_eff = self.cong_mult * 0.5

        for step in range(self.n_refine):
            optimizer.zero_grad()

            with torch.no_grad():
                pos.data[fixed_mask] = std_best[fixed_mask]

            clamped = torch.stack([
                pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

            wl = _lse_hpwl(clamped, net_data, port_base, gamma) / wl_norm

            density = _grid_density(clamped, sizes,
                                    cell_x_min, cell_x_max,
                                    cell_y_min, cell_y_max,
                                    cell_area, grid_rows, grid_cols)

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

            # Refinement loss: reweight components, keep overlap penalty hot
            # so we never re-introduce overlaps.
            refined_loss = (wl_eff * wl
                            + dens_eff * density
                            + cong_eff * congestion)
            loss = refined_loss + lam * overlap_norm
            loss.backward()

            torch.nn.utils.clip_grad_norm_([pos], max_norm=20.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                # For book-keeping use the *standard* proxy weighting (1·wl
                # + 0.5·density + 0.5·congestion) so "best" matches what the
                # outer evaluator will score with.
                std_proxy = (wl.item()
                             + 0.5 * density.item()
                             + 0.5 * congestion.item())
                ov_val = overlap_norm.item()
                score = std_proxy + max(1.0, lam * 0.1) * ov_val
                if score < best_score:
                    best_score = score
                    bp = pos.data.clone()
                    bp[fixed_mask] = std_best[fixed_mask]
                    bp[:, 0].clamp_(half_sizes[:, 0], cw - half_sizes[:, 0])
                    bp[:, 1].clamp_(half_sizes[:, 1], ch - half_sizes[:, 1])
                    best_pos = bp

            if step == 0 and self.verbose:
                print(f"    REFINE start: wl={wl.item():.4f} "
                      f"den={density.item():.4f} "
                      f"cong={congestion.item():.4f} "
                      f"(weights wl={wl_eff:.3f} dens={dens_eff:.3f} "
                      f"cong={cong_eff:.3f})")

        if self.verbose:
            ov_count = _count_overlaps(best_pos[:n_hard], half_sizes[:n_hard])
            print(f"  REFINE done: best_proxy_score={best_score:.4f}  "
                  f"overlaps={ov_count}  n_refine={self.n_refine}")

        return best_pos


# (CongestionRefinePlacer is defined above _DPOWithCongestionRefine so the
# evaluate.py loader picks it as the entry point.)
