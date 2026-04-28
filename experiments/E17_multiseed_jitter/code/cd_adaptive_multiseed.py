"""
E17 — CDAdaptiveMultiSeed: best-of-K with deterministic seed + jittered seeds.

Wraps the E9 CDAdaptive pipeline (SDF init → project_overlaps → CD adaptive)
in a multi-seed loop. The pool always includes one deterministic run
(seed=42, jitter=0.0) — byte-identical to the E9 champion — plus K-1
jittered runs. Best-of-K is selected by canonical compute_proxy_cost.

The deterministic baseline guarantees best-of-K ≥ single-seed champion;
jittered seeds are pure-upside exploration. Cost is K× single-seed wall.

This is an EXPERIMENT placer, not a champion. Champions in
`submissions/cd_adaptive/placer.py` and `submissions/cd_lns_gridbin/placer.py`
remain single-seed by default.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import torch

# Repo root: experiments/E17_multiseed_jitter/code/<this>.py → up 3 → repo
_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from macro_place.sdf_init import SDFPlacer

from submissions.cd_adaptive.placer import run_cd_adaptive

# Reuse project_overlaps from the diagnostic; champions already do this.
import importlib.util

_DIAGNOSTIC_PATH = _ROOT / "scripts" / "cd_ibm10_diagnostic.py"
_diag_spec = importlib.util.spec_from_file_location(
    "cd_ibm10_diagnostic", str(_DIAGNOSTIC_PATH)
)
_diag = importlib.util.module_from_spec(_diag_spec)
sys.modules.setdefault("cd_ibm10_diagnostic", _diag)
_diag_spec.loader.exec_module(_diag)
project_overlaps = _diag.project_overlaps


class CDAdaptiveMultiSeedPlacer:
    """Best-of-K wrapper around CDAdaptive with jittered SDF init.

    Args:
        seeds_jittered: integer seeds for the JITTERED runs only. The
            deterministic (seed=42, jitter=0.0) baseline is always added
            implicitly, so total runs = 1 + len(seeds_jittered).
        init_jitter: stddev of init perturbation as fraction of canvas_diag,
            applied only to the jittered runs.
        min_time_s, hard_cap_s, patience, plateau_threshold: forwarded to
            run_cd_adaptive (defaults match E9 champion).
        verbose: print per-seed progress.
    """

    def __init__(
        self,
        seeds_jittered: Sequence[int] = (43, 44),
        init_jitter: float = 0.05,
        min_time_s: float = 300.0,
        hard_cap_s: float = 3600.0,
        patience: int = 3,
        plateau_threshold: float = 0.005,
        verbose: bool = True,
    ) -> None:
        self.seeds_jittered = tuple(int(s) for s in seeds_jittered)
        self.init_jitter = float(init_jitter)
        self.min_time_s = float(min_time_s)
        self.hard_cap_s = float(hard_cap_s)
        self.patience = int(patience)
        self.plateau_threshold = float(plateau_threshold)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _run_one_seed(
        self, benchmark: Benchmark, seed: int, jitter: float
    ) -> tuple[torch.Tensor, float]:
        self._log(f"  ── seed={seed} (jitter={jitter}) ──")

        # 1. SDF init. (seed=42, jitter=0.0) is byte-identical to E9 champion.
        sdf_placer = SDFPlacer(seed=seed, init_jitter=jitter)
        placement = sdf_placer.place(benchmark)

        # 2. Project residual overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"    projection: {proj_iters} iters, "
            f"residual overlaps={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator on a fresh PLC (SDF mutates its own copy).
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"    init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. Adaptive CD.
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.min_time_s,
            hard_cap_s=self.hard_cap_s,
            patience=self.patience,
            plateau_threshold=self.plateau_threshold,
            log_fn=None,
        )
        final_internal = evaluator.current_cost()
        self._log(
            f"    CD: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, "
            f"internal proxy={final_internal['proxy']:.5f}"
        )

        # 5. Pull placement; restore fixed; validate; canonical proxy.
        final_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        if fixed_mask.any():
            final_f64[fixed_mask] = benchmark.macro_positions.to(torch.float64)[fixed_mask]
        final_placement = final_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"seed={seed} produced {overlaps['overlap_count']} overlaps "
                f"on '{benchmark.name}' — CD legality bug."
            )

        canonical = compute_proxy_cost(final_placement, benchmark, plc)
        canonical_proxy = float(canonical["proxy_cost"])
        self._log(f"    canonical proxy={canonical_proxy:.5f}")
        return final_placement, canonical_proxy

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.perf_counter()
        # Pool: 1 deterministic baseline + K-1 jittered runs.
        pool: list[tuple[int, float]] = [(42, 0.0)]
        pool.extend((s, self.init_jitter) for s in self.seeds_jittered)
        self._log(
            f"=== CDAdaptiveMultiSeed ({benchmark.name}): "
            f"pool={pool}, cap={self.hard_cap_s:.0f}s ==="
        )

        best_placement = None
        best_proxy = float("inf")
        best_key = None
        for seed, jitter in pool:
            placement, proxy = self._run_one_seed(benchmark, seed, jitter)
            if proxy < best_proxy:
                best_proxy = proxy
                best_placement = placement
                best_key = (seed, jitter)

        wall = time.perf_counter() - t0
        self._log(
            f"=== {benchmark.name} multi-seed done: "
            f"best (seed,jitter)={best_key}, best proxy={best_proxy:.5f}, "
            f"total wall={wall:.1f}s ==="
        )
        return best_placement
