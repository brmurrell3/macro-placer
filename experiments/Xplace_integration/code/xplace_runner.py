"""Xplace placer wrapper.

Runs CUHK Xplace via subprocess on a bookshelf-converted benchmark, parses
the output `.pl` back to our (N, 2) center-coordinate placement tensor, and
exposes a `.place(benchmark) -> Tensor` interface compatible with our harness.

Requirements:
  - Xplace built at $XPLACE_ROOT (default $HOME/Xplace)
  - Bookshelf files at <bookshelf_root>/<bench>/<bench>.aux
  - CUDA-capable GPU

Usage (programmatic):
  placer = XplacePlacer(budget_seconds=3300)
  placement = placer.place(benchmark)

Usage (CLI):
  uv run python xplace_runner.py <bench> [budget_s]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

# bookshelf converter (E76)
sys.path.insert(0, str(_ROOT / "experiments/E76_dreamplace_integration/code"))
from tilos_to_bookshelf import SCALE as BOOKSHELF_SCALE, main as _bksh_main  # noqa: E402

XPLACE_ROOT = Path(os.environ.get("XPLACE_ROOT", str(Path.home() / "Xplace")))
BOOKSHELF_ROOT = Path(
    os.environ.get(
        "BOOKSHELF_ROOT",
        str(_ROOT / "experiments/E76_dreamplace_integration/bookshelf_out"),
    )
)


def _ensure_bookshelf(bench_name: str) -> Path:
    """Convert TILOS → bookshelf if not already done. Returns aux path."""
    out_dir = BOOKSHELF_ROOT / bench_name
    aux = out_dir / f"{bench_name}.aux"
    if aux.exists():
        return aux
    out_dir.mkdir(parents=True, exist_ok=True)
    # call converter directly
    conv_script = _ROOT / "experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py"
    cmd = [sys.executable, str(conv_script), bench_name, str(out_dir)]
    subprocess.run(cmd, check=True)
    return aux


def _parse_pl(pl_path: Path) -> dict[str, tuple[float, float]]:
    """Parse a Bookshelf .pl file → {node_name: (llx, lly)}."""
    out: dict[str, tuple[float, float]] = {}
    with open(pl_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            # Expected: "<name> <llx> <lly> : N [/FIXED]"
            try:
                name = parts[0]
                llx = float(parts[1])
                lly = float(parts[2])
            except ValueError:
                continue
            out[name] = (llx, lly)
    return out


def _placement_from_pl(pl_path: Path, benchmark: Benchmark) -> torch.Tensor:
    """Convert Xplace output .pl → our (N,2) center-coordinate tensor."""
    pl = _parse_pl(pl_path)
    pos = benchmark.macro_positions.clone()
    sizes = benchmark.macro_sizes
    n = benchmark.num_macros
    # node names in our bookshelf are n<idx>
    for i in range(n):
        name = f"n{i}"
        if name in pl:
            llx, lly = pl[name]
            # bookshelf .pl coords were scaled by SCALE; divide back.
            # Then llx/y → center
            cx = llx / BOOKSHELF_SCALE + float(sizes[i, 0]) / 2.0
            cy = lly / BOOKSHELF_SCALE + float(sizes[i, 1]) / 2.0
            pos[i, 0] = cx
            pos[i, 1] = cy
    return pos


def run_xplace(bench_name: str, aux_path: Path, budget_s: int,
               exp_id: str = "tilos") -> Path:
    """Invoke Xplace; return path to output .pl. Raises on failure."""
    # Xplace cd; custom_path expects key:value pairs comma-separated
    custom = (
        f"benchmark:ispd2005,"
        f"design_name:{bench_name},"
        f"aux:{aux_path},"
        f"bookshelf_variety:ispd2005"
    )
    result_dir = XPLACE_ROOT / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(XPLACE_ROOT / "main.py"),
        "--custom_path", custom,
        "--exp_id", exp_id,
        "--result_dir", str(result_dir),
        "--global_placement", "True",
        "--detail_placement", "True",
        "--final_route_eval", "False",
        "--mixed_size", "True",
        "--gpu", "0",
        # Reasonable budget; Xplace usually converges in ~3000-5000 inner iters
        "--inner_iter", "10000",
        "--seed", "42",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(XPLACE_ROOT) + ":" + env.get("PYTHONPATH", "")
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(XPLACE_ROOT), env=env,
        timeout=budget_s, capture_output=True, text=True,
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Xplace failed (rc={proc.returncode}, wall={wall:.0f}s):\n"
            f"STDOUT tail:\n{proc.stdout[-2000:]}\n"
            f"STDERR tail:\n{proc.stderr[-2000:]}"
        )
    # Locate the output .pl. Xplace writes to:
    #   result/<exp_id>_<design>/output/<output_prefix>_<design>_<id>.pl
    candidates = list(result_dir.rglob(f"*{bench_name}*.pl"))
    if not candidates:
        raise FileNotFoundError(
            f"Xplace produced no .pl for {bench_name} in {result_dir}"
        )
    # Pick the most recent
    return max(candidates, key=lambda p: p.stat().st_mtime)


class XplacePlacer:
    """Run Xplace, return our placement tensor.

    Optional post-polish via cascade saddle if budget remains.
    """

    def __init__(self,
                 budget_seconds: float = 3300.0,
                 post_polish: bool = False,
                 verbose: bool = True):
        self.budget_seconds = budget_seconds
        self.post_polish = post_polish
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== XplacePlacer ({benchmark.name}) ===")
        t0 = time.time()

        aux = _ensure_bookshelf(benchmark.name)
        log(f"  bookshelf: {aux}")

        # Reserve 60s for parsing + polish
        xplace_budget = int(self.budget_seconds - 60)
        log(f"  invoking Xplace (budget={xplace_budget}s)")
        pl_out = run_xplace(benchmark.name, aux, xplace_budget)
        log(f"  Xplace output: {pl_out}")

        placement = _placement_from_pl(pl_out, benchmark)

        # Optional post-polish: feed Xplace placement into our cascade
        if self.post_polish and (time.time() - t0) < self.budget_seconds - 300:
            log("  post-polish: cascade saddle (TODO — wire in)")
            # TODO: wire in cd_lns_sa_cascade with placement-as-init
            pass

        wall = time.time() - t0
        log(f"  total wall: {wall:.0f}s")
        return placement


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 3300.0
    print(f"[xplace] bench={bench} budget={budget}s")
    bench_dir = find_benchmark_dir(bench)
    b, plc = load_benchmark_from_dir(str(bench_dir))
    placer = XplacePlacer(budget_seconds=budget, verbose=True)
    placement = placer.place(b)
    proxy = compute_proxy_cost(placement, b, plc)
    ovl = compute_overlap_metrics(placement, b)["overlap_count"]
    print(f"[xplace] proxy={proxy['proxy_cost']:.5f} ovl={ovl}")


if __name__ == "__main__":
    main()
