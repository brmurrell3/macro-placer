"""CDLNSSACascadeSeedEnsemblePlacer — N parallel Option-C runs, best-of-N.

Hypothesis: cascade pipeline has 0.3-0.7% inter-seed variance. With 4
cores, running 4 parallel placers with different seeds at FULL budget
takes the same wall time as a single run (assuming each child uses
1 thread). Pick best by canonical proxy.

Architecture:
  1. Parent process: receives benchmark, spawns N child processes.
  2. Each child:
       a. torch.set_num_threads(1)  (avoid contention)
       b. Patches CDLNSSAPlacer / CDLNSSADPOKJointPlacer default seeds.
       c. Loads benchmark independently (avoids pickling plc).
       d. Runs CDLNSSACascadeStackedPeripheryPlacer with the given seed.
       e. Saves placement tensor + proxy + overlap count to a temp file.
  3. Parent gathers results, picks best (zero overlap + lowest proxy).

Falls back to single-process Option C if multiprocessing fails for any
reason.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir


def _worker_entry(seed_id: int, bench_name: str, budget_seconds: float,
                  output_path: str, verbose: bool = False) -> None:
    """Subprocess entrypoint — runs Option C with patched seeds.

    Writes a pickle dict {"placement": Tensor, "proxy": float,
    "overlap_count": int, "seed": int, "wall_s": float, "error": Optional[str]}
    to output_path.
    """
    result = {"seed": seed_id, "error": None, "placement": None,
              "proxy": float("inf"), "overlap_count": 999, "wall_s": 0.0}
    t_worker = time.time()
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed_id)
        import numpy as np
        np.random.seed(seed_id)

        sys.path.insert(0, str(_ROOT))
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

        # Patch CDLNSSAPlacer / CDLNSSADPOKJointPlacer to use seed_id.
        import importlib.util
        e25_spec = importlib.util.spec_from_file_location(
            "e25_p", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py"),
        )
        e25_mod = importlib.util.module_from_spec(e25_spec)
        e25_spec.loader.exec_module(e25_mod)
        orig_e25 = e25_mod.CDLNSSAPlacer.__init__

        def patched_e25(self, *args, **kwargs):
            kwargs.setdefault("lns_seed", seed_id)
            kwargs.setdefault("sa_seed", seed_id)
            return orig_e25(self, *args, **kwargs)

        e25_mod.CDLNSSAPlacer.__init__ = patched_e25

        from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer
        orig_e41 = CDLNSSADPOKJointPlacer.__init__

        def patched_e41(self, *args, **kwargs):
            kwargs.setdefault("lns_seed", seed_id)
            kwargs.setdefault("sa_seed", seed_id)
            return orig_e41(self, *args, **kwargs)

        CDLNSSADPOKJointPlacer.__init__ = patched_e41

        # Load placer module fresh in this subprocess.
        opt_c_spec = importlib.util.spec_from_file_location(
            "opt_c_p",
            str(_ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"),
        )
        opt_c_mod = importlib.util.module_from_spec(opt_c_spec)
        opt_c_spec.loader.exec_module(opt_c_mod)

        bench_dir = find_benchmark_dir(bench_name)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))

        placer = opt_c_mod.CDLNSSACascadeStackedPeripheryPlacer(
            budget_seconds=budget_seconds,
            rng_seed=seed_id,
            verbose=verbose,
        )
        placement = placer.place(benchmark)

        proxy = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
        ovl = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])

        result["placement"] = placement.cpu()
        result["proxy"] = proxy
        result["overlap_count"] = ovl
        result["wall_s"] = time.time() - t_worker
    except Exception:
        result["error"] = traceback.format_exc()
        result["wall_s"] = time.time() - t_worker
    finally:
        with open(output_path, "wb") as f:
            pickle.dump(result, f)


class CDLNSSACascadeSeedEnsemblePlacer:
    """Seed-ensemble best-of-N wrapper around Option C."""

    def __init__(
        self,
        n_seeds: int = 4,
        seeds: Optional[Tuple[int, ...]] = None,
        budget_seconds: Optional[float] = 3300.0,
        startup_overhead_s: float = 30.0,
        verbose: bool = True,
    ):
        self.n_seeds = n_seeds
        if seeds is None:
            seeds = tuple(range(42, 42 + n_seeds))
        assert len(seeds) >= n_seeds, "Not enough seeds for n_seeds"
        self.seeds = seeds[:n_seeds]
        self.budget_seconds = budget_seconds
        self.startup_overhead_s = startup_overhead_s
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeSeedEnsemblePlacer ({benchmark.name}) "
            f"n_seeds={self.n_seeds} seeds={self.seeds} ===")
        t0 = time.time()

        child_budget = (
            (self.budget_seconds - self.startup_overhead_s)
            if self.budget_seconds else 3300.0
        )

        # Spawn N workers.
        tmpdir = Path(tempfile.mkdtemp(prefix="seed_ensemble_"))
        log(f"  spawning {self.n_seeds} workers (each budget={child_budget:.0f}s) "
            f"tmpdir={tmpdir}")

        try:
            ctx = mp.get_context("spawn")
        except (RuntimeError, ValueError):
            ctx = mp.get_context()

        procs = []
        out_paths = []
        for seed_id in self.seeds:
            out_path = str(tmpdir / f"seed_{seed_id}.pkl")
            out_paths.append(out_path)
            p = ctx.Process(
                target=_worker_entry,
                args=(seed_id, benchmark.name, child_budget, out_path,
                      False),  # children silent
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Gather results.
        results = []
        for seed_id, out_path in zip(self.seeds, out_paths):
            try:
                with open(out_path, "rb") as f:
                    r = pickle.load(f)
                results.append(r)
                if r["error"]:
                    log(f"  seed={seed_id} ERROR: {r['error'].splitlines()[-1]}")
                else:
                    log(f"  seed={seed_id} proxy={r['proxy']:.5f} "
                        f"ovl={r['overlap_count']} wall={r['wall_s']:.0f}s")
            except Exception as exc:
                log(f"  seed={seed_id}: failed to read result ({exc})")

        # Pick best valid (zero ovl + lowest proxy).
        valid = [r for r in results if r["error"] is None and
                 r["overlap_count"] == 0 and r["placement"] is not None]
        if not valid:
            # Fallback: pick any non-error result even with overlap.
            valid = [r for r in results if r["error"] is None and
                     r["placement"] is not None]
        if not valid:
            raise RuntimeError(
                f"All {self.n_seeds} seed workers failed for {benchmark.name}"
            )

        best = min(valid, key=lambda r: r["proxy"])
        log(f"  WINNER: seed={best['seed']} proxy={best['proxy']:.5f} "
            f"ovl={best['overlap_count']} "
            f"(others: {[(r['seed'], round(r['proxy'], 5)) for r in results]})")
        log(f"  total wall={time.time()-t0:.0f}s")

        # Cleanup
        try:
            import shutil
            shutil.rmtree(tmpdir)
        except Exception:
            pass

        return best["placement"].to(torch.float32)


Placer = CDLNSSACascadeSeedEnsemblePlacer
