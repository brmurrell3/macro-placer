"""E146 — Hyperparameter plateau-pick.

Run 3 different HP configs in series, score each by canonical proxy,
pick best, then run final CD polish on the chosen basin.

Different from E129 multi-restart: E129 varies SEED (FP32 nondeterminism
explores slightly different basins around a fixed HP point); E146 varies
DESCENT HYPERPARAMETERS (lr_frac, gamma_end_frac, overlap_lambda_end)
to explore basin FAMILIES.

Pipeline per bench:
  for cfg in [A, B, C]:
    pos = V4+Gauss descent(cfg) + legalize + light CD polish (~200 s)
    if ovl == 0:
      cand[cfg] = (canonical_proxy(pos), pos)
  best = min(cand, key=proxy)
  return CD_polish(best, budget=600 s)

Configs (A is current default; B and C are perturbations):
  A: lr_frac=0.005, gamma_end_frac=5e-5, overlap_lambda_end=10.0
  B: lr_frac=0.003, gamma_end_frac=5e-5, overlap_lambda_end=5.0
  C: lr_frac=0.008, gamma_end_frac=2e-5, overlap_lambda_end=20.0

Total per-bench wall budget: 1800 s. Within partcl 60 min/bench cap.

DO NOT mutate E127 source — it has downstream callers.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian


# Three HP configs to plateau-pick across.
HP_CONFIGS = {
    "A": dict(
        lr_frac=0.005,
        gamma_end_frac=5e-5,
        overlap_lambda_end=10.0,
    ),
    "B": dict(
        lr_frac=0.003,
        gamma_end_frac=5e-5,
        overlap_lambda_end=5.0,
    ),
    "C": dict(
        lr_frac=0.008,
        gamma_end_frac=2e-5,
        overlap_lambda_end=20.0,
    ),
}


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    plateau_threshold: float = 0.001,
):
    """Run CD-adaptive polish under a hard wall budget."""
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=min(30.0, budget_s * 0.5),
        hard_cap_s=budget_s,
        patience=5,
        plateau_threshold=plateau_threshold,
        log_fn=None,
    )
    return evaluator.placement.detach().clone().to(torch.float32)


class E146HPPlateauPlacer:
    """V4+Gaussian descent under 3 different HP configs; plateau-pick + CD polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1800.0,
        config_polish_s: float = 200.0,
        final_polish_s: float = 600.0,
        num_steps: int = 500,
        gamma_start_frac: float = 5e-3,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.config_polish_s = config_polish_s
        self.final_polish_s = final_polish_s
        self.num_steps = num_steps
        self.gamma_start_frac = gamma_start_frac
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Phase records (populated by place()).
        self.last_config_basins: dict = {}      # cfg -> basin proxy (post-descent)
        self.last_config_polished: dict = {}    # cfg -> post-light-CD proxy
        self.last_config_walls: dict = {}       # cfg -> wall seconds
        self.last_chosen_config: Optional[str] = None
        self.last_final_proxy: Optional[float] = None
        self.last_total_wall: Optional[float] = None

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _descend_one(
        self,
        benchmark: Benchmark,
        plc,
        cfg_name: str,
        cfg: dict,
        device: str,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """One descent under a specific HP config. Returns (pos_legal, basin_proxy)."""
        try:
            descender = SmoothGlobalPlacerV4Gaussian(
                num_steps=self.num_steps,
                lr_frac=cfg["lr_frac"],
                gamma_start_frac=self.gamma_start_frac,
                gamma_end_frac=cfg["gamma_end_frac"],
                overlap_lambda_end=cfg["overlap_lambda_end"],
                overlap_ramp_pct=self.overlap_ramp_pct,
                init=self.init,
                device=device,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos = descender.place(benchmark)
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                pos, _ = project_overlaps(pos, benchmark)
                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                self._log(f"    cfg {cfg_name}: descent still has {ovl} overlaps; skipped")
                return None, float("inf")
            proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            return pos, proxy
        except Exception as exc:
            self._log(f"    cfg {cfg_name}: EXCEPTION: {exc}")
            return None, float("inf")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E146 HP-plateau ({benchmark.name}) configs={list(HP_CONFIGS)} ===")
        self._log(
            f"  budget={self.budget_seconds}s; per-config polish={self.config_polish_s}s "
            f"+ final polish={self.final_polish_s}s"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")

        # ------- Phase 1: descend + light CD polish under each HP config.
        results: List[Tuple[float, str, torch.Tensor]] = []
        cfg_count = len(HP_CONFIGS)
        for i, (cfg_name, cfg) in enumerate(HP_CONFIGS.items()):
            t_cfg = time.time()

            # Budget guard: if budget too tight to do remaining cfgs + final
            # polish, stop early.
            if deadline is not None:
                remaining = deadline - time.time()
                # We still need: (cfg_count - i) descents + final polish + slack.
                needed = (cfg_count - i) * 60.0 + self.final_polish_s + 30.0
                if remaining < needed:
                    self._log(
                        f"  cfg {cfg_name}: SKIPPED (remaining={remaining:.0f}s < "
                        f"needed={needed:.0f}s)"
                    )
                    continue

            self._log(
                f"  cfg {cfg_name}: lr_frac={cfg['lr_frac']} "
                f"gamma_end_frac={cfg['gamma_end_frac']} "
                f"overlap_lambda_end={cfg['overlap_lambda_end']}"
            )

            t_descent = time.time()
            pos, basin_proxy = self._descend_one(benchmark, plc, cfg_name, cfg, device)
            descent_wall = time.time() - t_descent
            self.last_config_basins[cfg_name] = basin_proxy
            if pos is None:
                self._log(f"    cfg {cfg_name}: descent FAILED (wall={descent_wall:.0f}s)")
                continue
            self._log(
                f"    cfg {cfg_name}: basin={basin_proxy:.5f} "
                f"descent_wall={descent_wall:.0f}s"
            )

            # Light CD polish on this basin (partial, to differentiate basins
            # that look identical post-descent but have different polish ceilings).
            if deadline is not None:
                remaining = deadline - time.time()
                reserve = (cfg_count - i - 1) * 60.0 + self.final_polish_s + 30.0
                cfg_polish_budget = max(30.0, min(self.config_polish_s, remaining - reserve))
            else:
                cfg_polish_budget = self.config_polish_s
            t_polish = time.time()
            self._log(f"    cfg {cfg_name}: light CD polish budget={cfg_polish_budget:.0f}s")
            pos = _cd_polish(pos, benchmark, plc, cfg_polish_budget, self.cd_plateau_threshold)
            polish_wall = time.time() - t_polish
            polished_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            polished_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            cfg_wall = time.time() - t_cfg

            self.last_config_polished[cfg_name] = polished_proxy
            self.last_config_walls[cfg_name] = cfg_wall

            if polished_ovl > 0:
                self._log(
                    f"    cfg {cfg_name}: CD polish produced {polished_ovl} ovl; "
                    f"running project_overlaps"
                )
                pos, _ = project_overlaps(pos, benchmark)
                polished_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
                if polished_ovl > 0:
                    self._log(
                        f"    cfg {cfg_name}: still {polished_ovl} ovl; SKIPPING this cfg"
                    )
                    continue
                polished_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
                self.last_config_polished[cfg_name] = polished_proxy

            results.append((polished_proxy, cfg_name, pos))
            self._log(
                f"  cfg {cfg_name} done: polished={polished_proxy:.5f} "
                f"(Δ={polished_proxy - basin_proxy:+.5f}) "
                f"polish_wall={polish_wall:.0f}s cfg_total={cfg_wall:.0f}s"
            )

        if not results:
            self._log("  ALL configs FAILED; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            chosen_name = "SDF_FALLBACK"
            chosen_proxy = float("inf")
        else:
            results.sort(key=lambda r: r[0])
            chosen_proxy, chosen_name, pos = results[0]
            self._log(
                f"  CHOSEN: cfg {chosen_name} polished={chosen_proxy:.5f} "
                f"(of {len(results)} eligible)"
            )
        self.last_chosen_config = chosen_name

        # ------- Phase 2: final CD polish on chosen basin.
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            final_budget = max(30.0, min(self.final_polish_s, remaining))
        else:
            final_budget = self.final_polish_s
        t_final = time.time()
        self._log(f"  Final CD polish budget={final_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, final_budget, self.cd_plateau_threshold)
        final_wall = time.time() - t_final

        final = pos.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        total_wall = time.time() - t0

        self.last_final_proxy = final_proxy
        self.last_total_wall = total_wall

        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"final_wall={final_wall:.0f}s total_wall={total_wall:.0f}s"
        )
        self._log(
            f"  Config basins (post-light-polish): "
            + " ".join(
                f"{k}={v:.5f}" for k, v in sorted(self.last_config_polished.items())
            )
            + f" chosen={chosen_name}"
        )

        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Alias for harness convenience.
Placer = E146HPPlateauPlacer
