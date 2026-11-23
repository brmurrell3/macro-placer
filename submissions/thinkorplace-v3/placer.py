"""thinkorplace v3: seed pool, span screen, warm chain and stacked search under one deadline.

    uv run evaluate submissions/thinkorplace-v3/placer.py -b ibm01

See README.md in this folder for the method and its measured result.
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# evaluate loads this file by path, not as a package member: put this folder (which holds the thinkorplace_v3
# package) and the repository root (which holds macro_place and the thinkorplace runtime) on the import path
for _path in (HERE, HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

from thinkorplace.common.bench_paths import find_benchmark_dir  # noqa: E402
from thinkorplace.common.cd_core import sdf_init  # noqa: E402
from thinkorplace.common.runtime_safety import valid_placement  # noqa: E402
from thinkorplace_v3.legalize import is_legal, legalize  # noqa: E402
from thinkorplace_v3.scheduler import run_chains  # noqa: E402


class Placer:
    """``place(benchmark)`` returns a [num_macros, 2] float32 placement with no hard-macro overlap.

    The budget covers everything from the call to the return. The design is
    reloaded from its directory (found by name) because the chains need its
    netlist, which the Benchmark object does not carry. The number printed at
    the end is the incremental evaluator's; the reference scorer's may differ
    by up to about 0.005. If no chain publishes in time, the supplied placement
    is legalised and returned instead (or, failing that, the SDF initialiser's).
    """

    def __init__(self, budget_seconds: float = 720.0, work_dir: str | None = None, verbose: bool = True):
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("budget_seconds must be finite and positive")
        self.budget_seconds = budget_seconds
        self.work_dir = work_dir
        self.verbose = verbose
        self.last_run_metadata: dict = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _fallback(self, benchmark, reason: str) -> torch.Tensor:
        legal = legalize(benchmark.macro_positions.clone().to(torch.float32), benchmark)
        if legal is not None:
            self._log(f"v3 fallback ({reason}): returning the legalised supplied placement")
            return legal.to(torch.float32)
        sdf = sdf_init(benchmark).to(torch.float32)
        if is_legal(sdf, benchmark):
            self._log(f"v3 fallback ({reason}): returning the SDF initialiser's placement")
            return sdf
        raise RuntimeError(f"{reason}, and neither fallback placement is legal")

    def place(self, benchmark) -> torch.Tensor:
        t0 = time.time()
        if self.work_dir:  # a fresh subdirectory per call, so earlier runs' files are never read
            Path(self.work_dir).mkdir(parents=True, exist_ok=True)
            out = Path(tempfile.mkdtemp(prefix=f"{benchmark.name}_", dir=self.work_dir))
        else:
            out = Path(tempfile.mkdtemp(prefix="thinkorplace_v3_"))
        try:
            positions, summary = run_chains(str(find_benchmark_dir(benchmark.name)), out, t0, self.budget_seconds)
            self.last_run_metadata = summary
            if positions is None:
                return self._fallback(benchmark, "no chain published a placement before the deadline")
            valid, violations = valid_placement(positions, benchmark)
            if not valid:
                return self._fallback(benchmark, f"the chosen placement is invalid: {violations}")
            chosen = summary["chosen"]
            self._log(f"v3 chain {chosen['chain']} ({chosen['stage']}) evaluator proxy {chosen['proxy']:.6f}")
            return positions.to(torch.float32)
        finally:
            if not self.work_dir:
                shutil.rmtree(out, ignore_errors=True)
