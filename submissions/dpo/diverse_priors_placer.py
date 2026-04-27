"""
E11 — Diverse-Priors Best-of-N DPO Placer

Hypothesis: E5 falsified perturbed-SDF best-of-N (B=64 same basin as B=1).
Different *priors* should land DPO in different convergence basins, giving
genuine best-of-N value.

This placer runs DPO once per init strategy (default: SDF / Will-SA / greedy /
random-projected), then picks the placement with the lowest real proxy cost.

The DPO core is reused from `ablation_v2_steps.DPOv2StepsPlacer` by
subclassing and overriding `_sdf_init` so we can inject any precomputed init
without duplicating the gradient logic.

Coordination: `best_of_v2_placer.py` is owned by another agent and is NOT
modified. `incremental_evaluator.py` is owned by another agent — not touched.
"""

import importlib.util
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import List, Optional

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


# ---------------------------------------------------------------------------
# Reuse loaders / DPO core via importlib (no copy-paste of the gradient code)
# ---------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HERE = Path(__file__).parent
_V2_PATH = _HERE / "ablation_v2_steps.py"
_INIT_PATH = _HERE / "init_strategies.py"

_v2_mod = _load_module(_V2_PATH, "dpo_v2_steps")
_init_mod = _load_module(_INIT_PATH, "init_strategies")

DPOv2StepsPlacer = _v2_mod.DPOv2StepsPlacer
_load_plc = _v2_mod._load_plc
INIT_STRATEGIES = _init_mod.INIT_STRATEGIES


# ---------------------------------------------------------------------------
# Best-of-N over diverse priors
# (Defined first so the evaluate.py loader picks this class as the placer.
#  The DPO adapter `_DPOWithCustomInit` is defined below.)
# ---------------------------------------------------------------------------

class DiversePriorsPlacer:
    """Run DPO from each enabled prior, pick the best by real proxy cost.

    Default priors: ['sdf', 'will', 'greedy', 'random'].
    Pass `priors=['sdf']` to reproduce single-init behaviour for ablation.
    """

    def __init__(self, seed: int = 42,
                 priors: Optional[List[str]] = None,
                 verbose: bool = True):
        self.seed = seed
        self.priors = list(priors) if priors is not None else \
            ["sdf", "will", "greedy", "random"]
        self.verbose = verbose

        # Validate prior names eagerly
        for p in self.priors:
            if p not in INIT_STRATEGIES:
                raise KeyError(
                    f"Unknown prior '{p}'. Available: {sorted(INIT_STRATEGIES)}"
                )

        # Diagnostic state populated by .place()
        self.last_diagnostics = {}

    # -----------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)

        results = []  # list of (prior_name, placement, proxy_cost)
        diag = {"per_prior": {}}

        for prior_name in self.priors:
            init_fn = INIT_STRATEGIES[prior_name]

            if self.verbose:
                print(f"\n  [E11] === Prior: {prior_name} ===")

            # 1. Run init strategy. Suppress its stdout so the log isn't
            #    flooded with sub-placer prints.
            buf = StringIO()
            try:
                with redirect_stdout(buf):
                    init_pos = init_fn(benchmark, seed=self.seed)
            except TypeError:
                # Some inits (e.g. greedy) don't take seed
                with redirect_stdout(buf):
                    init_pos = init_fn(benchmark)

            # Validate the init produced the right shape
            if init_pos.shape != (benchmark.num_macros, 2):
                raise RuntimeError(
                    f"Init '{prior_name}' returned shape {init_pos.shape}, "
                    f"expected ({benchmark.num_macros}, 2)"
                )

            # 2. Run DPO with that init
            dpo = _DPOWithCustomInit(init_tensor=init_pos, seed=self.seed)
            placement = dpo.place(benchmark)

            # 3. Score with the real proxy
            if plc is None:
                cost = float("inf")
                ovs = -1
            else:
                costs = compute_proxy_cost(placement, benchmark, plc)
                cost = float(costs["proxy_cost"])
                ovs = int(costs["overlap_count"])

            if self.verbose:
                print(f"  [E11] prior={prior_name:>7s}  "
                      f"proxy={cost:.4f}  overlaps={ovs}")

            diag["per_prior"][prior_name] = {
                "proxy": cost, "overlaps": ovs,
            }
            results.append((prior_name, placement, cost, ovs))

        # 4. Pick best by proxy cost (skip placements with overlaps)
        clean = [r for r in results if r[3] == 0]
        if clean:
            best = min(clean, key=lambda r: r[2])
        else:
            # Fallback: pick whichever has the lowest cost despite overlaps
            best = min(results, key=lambda r: r[2])

        winner_name, winner_pos, winner_cost, winner_ovs = best
        diag["winner"] = winner_name
        diag["winner_proxy"] = winner_cost
        diag["winner_overlaps"] = winner_ovs
        self.last_diagnostics = diag

        if self.verbose:
            print(f"\n  [E11] ====> winner: {winner_name}  "
                  f"proxy={winner_cost:.4f}  ov={winner_ovs}")

        return winner_pos


# ---------------------------------------------------------------------------
# Adapter: DPO that takes an externally provided init tensor
# (Defined AFTER DiversePriorsPlacer so evaluate.py's loader picks
# DiversePriorsPlacer as the entry point — the loader picks the first class
# defined in this file with a `place` method.)
# ---------------------------------------------------------------------------

class _DPOWithCustomInit(DPOv2StepsPlacer):
    """Subclass of v2-steps DPO that injects a pre-computed init."""

    def __init__(self, init_tensor: torch.Tensor, seed: int = 42):
        super().__init__(seed=seed, use_sdf_init=True)
        self._injected_init = init_tensor

    def _sdf_init(self, benchmark):  # type: ignore[override]
        # Return the externally supplied init so we don't re-run SDF.
        return self._injected_init.clone()
