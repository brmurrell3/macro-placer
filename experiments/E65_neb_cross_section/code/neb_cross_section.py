"""E65 placer: linear cross-section between E25 (SDF) and E41 (DPO) basins.

This isn't a placer in the optimization sense — it's a diagnostic that
runs E25, runs E41, then evaluates 11 linear interpolations between
their placements. Logs proxy along the path. Returns the lowest-proxy
output (which by E48 mechanism is essentially E48 — but interpolation
proxies and overlap counts are the *real* outputs of this experiment).

If the path between basins has a minimum below max(p_E25 proxy, p_E41
proxy): a third basin is reachable. Full NEB next.

If the path is monotone up: basins separated by a barrier. NEB
finds the saddle height (different attack mechanism needed).

Pipeline (per benchmark):
  1. Run E25 (CDLNSSAPlacer)  → p_E25, proxy_E25.
  2. Run E41 (CDLNSSADPOKJointPlacer) → p_E41, proxy_E41.
  3. For k in [0.0, 0.1, ..., 1.0]:
       p_k = (1-k) * p_E25 + k * p_E41
       p_k_legal = project_overlaps(p_k, benchmark)
       proxy_k_pre = compute_proxy_cost(p_k, ...)
       proxy_k_post = compute_proxy_cost(p_k_legal, ...)
       Record (k, proxy_k_pre, proxy_k_post, overlap_count_pre).
  4. Write `cross_section.jsonl` with all data points.
  5. Return the lowest-proxy among (p_E25, p_E41, p_k_legal for each k).

This guarantees the placer returns a valid placement (so harness's
proxy + overlap check passes), AND surfaces the cross-section data.

Reference:
- E25 — `submissions/cd_lns_sa/placer.py` (SDF basin pipeline).
- E41 — `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`
  (DPO + K-joint basin pipeline).
- Henkelman & Jónsson (2000), "Improved tangent estimate in the
  nudged elastic band method for finding minimum energy paths and
  saddle points," J. Chem. Phys. 113, 9978.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


class NEBCrossSectionPlacer:
    """E65 diagnostic: cross-section between E25 and E41 basins."""

    def __init__(
        self,
        n_steps: int = 11,
        log_path: str | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        # Filter kwargs that only apply to E41
        e41_only_keys = {"kjoint_K", "kjoint_top_N", "kjoint_budget_s", "kjoint_seed"}
        e41_kwargs = {k: v for k, v in kwargs.items() if k in e41_only_keys}
        common_kwargs = {k: v for k, v in kwargs.items() if k not in e41_only_keys}
        self._e25 = CDLNSSAPlacer(**common_kwargs)
        self._e41 = CDLNSSADPOKJointPlacer(**common_kwargs, **e41_kwargs)
        self.n_steps = int(n_steps)
        self.log_path = log_path
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.perf_counter()
        self._log(
            f"=== NEBCrossSectionPlacer ({benchmark.name}): "
            f"E25 + E41 + cross-section ({self.n_steps} steps) ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # 1. Run E25.
        t_e25 = time.perf_counter()
        p_E25 = self._e25.place(benchmark)
        proxy_E25 = compute_proxy_cost(p_E25, benchmark, plc)["proxy_cost"]
        ovl_E25 = compute_overlap_metrics(p_E25, benchmark)["overlap_count"]
        self._log(
            f"  E25 done: proxy={proxy_E25:.5f} overlaps={ovl_E25} "
            f"wall={time.perf_counter() - t_e25:.1f}s"
        )

        # 2. Run E41.
        t_e41 = time.perf_counter()
        p_E41 = self._e41.place(benchmark)
        proxy_E41 = compute_proxy_cost(p_E41, benchmark, plc)["proxy_cost"]
        ovl_E41 = compute_overlap_metrics(p_E41, benchmark)["overlap_count"]
        self._log(
            f"  E41 done: proxy={proxy_E41:.5f} overlaps={ovl_E41} "
            f"wall={time.perf_counter() - t_e41:.1f}s"
        )

        # 3. Cross-section: linear interpolations.
        self._log(
            f"  starting cross-section sweep ({self.n_steps} steps from "
            f"E25 to E41)"
        )

        p_E25_f64 = p_E25.detach().cpu().to(torch.float64)
        p_E41_f64 = p_E41.detach().cpu().to(torch.float64)
        n_hard = benchmark.num_hard_macros
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)

        cross_section = []
        candidates = []  # (name, placement, proxy)
        candidates.append(("E25", p_E25, proxy_E25))
        candidates.append(("E41", p_E41, proxy_E41))

        ks = np.linspace(0.0, 1.0, self.n_steps)
        for k in ks:
            t_step = time.perf_counter()

            # Linear interpolation in placement space.
            p_k = (1.0 - k) * p_E25_f64 + k * p_E41_f64
            # Preserve fixed macros.
            if fixed_mask.any():
                p_k[fixed_mask] = original_positions[fixed_mask]

            # Pre-legalization metrics.
            p_k_t = p_k.to(torch.float32)
            ovl_pre = compute_overlap_metrics(p_k_t, benchmark)
            proxy_pre = compute_proxy_cost(p_k_t, benchmark, plc)["proxy_cost"]

            # Legalize.
            p_k_legal, proj_iters = project_overlaps(p_k_t, benchmark)
            ovl_post = compute_overlap_metrics(p_k_legal, benchmark)
            proxy_post = compute_proxy_cost(p_k_legal, benchmark, plc)["proxy_cost"]

            row = {
                "k": float(k),
                "proxy_pre_legal": float(proxy_pre),
                "overlap_count_pre": int(ovl_pre["overlap_count"]),
                "proxy_post_legal": float(proxy_post) if ovl_post["overlap_count"] == 0 else None,
                "overlap_count_post": int(ovl_post["overlap_count"]),
                "proj_iters": int(proj_iters),
                "step_wall_s": float(time.perf_counter() - t_step),
            }
            cross_section.append(row)
            self._log(
                f"  k={k:.2f}: proxy_pre={proxy_pre:.5f} (ovl_pre="
                f"{ovl_pre['overlap_count']}) → "
                f"proxy_post={'%.5f' % proxy_post if ovl_post['overlap_count'] == 0 else 'INFEASIBLE'} "
                f"(ovl_post={ovl_post['overlap_count']}, proj_iters={proj_iters}, "
                f"wall={row['step_wall_s']:.1f}s)"
            )

            if ovl_post["overlap_count"] == 0:
                candidates.append((f"interp_k={k:.2f}", p_k_legal, proxy_post))

        # 4. Write cross-section log.
        log_dir = Path(__file__).resolve().parents[1] / "results"
        log_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"cross_section_{benchmark.name}_{ts}.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({
                "benchmark": benchmark.name,
                "n_hard_macros": n_hard,
                "n_steps": self.n_steps,
                "proxy_E25": float(proxy_E25),
                "proxy_E41": float(proxy_E41),
                "max_endpoint_proxy": float(max(proxy_E25, proxy_E41)),
                "min_endpoint_proxy": float(min(proxy_E25, proxy_E41)),
            }) + "\n")
            for row in cross_section:
                f.write(json.dumps(row) + "\n")
        self._log(f"  cross-section written to {log_file}")

        # 5. Return lowest-proxy valid placement.
        candidates.sort(key=lambda x: x[2])
        winner_name, winner_placement, winner_proxy = candidates[0]
        endpoints_str = f"E25={proxy_E25:.5f}, E41={proxy_E41:.5f}"
        self._log(
            f"  E65 winner: {winner_name} proxy={winner_proxy:.5f} "
            f"({endpoints_str}, total_wall={time.perf_counter() - t0:.1f}s)"
        )

        # Diagnostic summary.
        valid_proxies = [r["proxy_post_legal"] for r in cross_section
                         if r["proxy_post_legal"] is not None]
        if valid_proxies:
            min_path_proxy = min(valid_proxies)
            max_path_proxy = max(valid_proxies)
            self._log(
                f"  CROSS-SECTION SUMMARY: min_path_proxy={min_path_proxy:.5f}, "
                f"max_path_proxy={max_path_proxy:.5f}, "
                f"endpoint_min={min(proxy_E25, proxy_E41):.5f}, "
                f"interpolations_below_endpoint_min={sum(1 for p in valid_proxies if p < min(proxy_E25, proxy_E41))}/"
                f"{len(valid_proxies)}"
            )
            if any(p < min(proxy_E25, proxy_E41) for p in valid_proxies):
                self._log(
                    f"  ⚡ SIGNAL: at least one interpolation point is BELOW "
                    f"the better endpoint — third basin reachable by linear "
                    f"interpolation + legalization. NEB worth the build."
                )
            else:
                self._log(
                    f"  📊 PATH IS MONOTONE-UP (no interpolation beats endpoint min). "
                    f"Basins separated by barrier; need NEB to find saddle height "
                    f"and then non-local moves to bridge."
                )

        return winner_placement
