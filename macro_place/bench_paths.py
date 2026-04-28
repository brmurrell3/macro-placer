"""Generic benchmark-directory discovery for placer reload of PlacementCost.

Problem: the eval harness calls ``placer.place(benchmark)`` with only the
``Benchmark`` dataclass — no path, no plc. Many of our placers (CDAdaptive,
CDOnly, SDF init) need to reload the underlying ``PlacementCost`` to access
fields the dataclass doesn't carry (per-pin info, smoothing range, etc.).

This module supplies a single helper, ``find_benchmark_dir(name)``, that
locates the source directory for a benchmark in a generic way:

  1. ``$BENCH_ROOT`` environment variable, if set (eval harness or contest
     can override the search root explicitly).
  2. Standard testcase roots under ``external/MacroPlacement/Testcases``:
     ``ICCAD04`` (IBM benchmarks) and ``NG45`` (commercial designs).
  3. Shallow ``rglob`` fallback from the repo root looking for
     ``{name}/netlist.pb.txt``.

No benchmark-name-specific logic, no per-benchmark conditionals — by design
this works equally well on IBM and NG45 and survives hidden NG45 designs as
long as the contest places them in a discoverable location.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent


def _candidate_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("BENCH_ROOT")
    if env:
        roots.append(Path(env))
    base = _REPO_ROOT / "external" / "MacroPlacement" / "Testcases"
    roots.extend([base / "ICCAD04", base / "NG45"])
    return roots


def find_benchmark_dir(name: str) -> Path:
    """Return the source directory for a benchmark by name.

    Raises FileNotFoundError if no candidate path resolves. The error
    message lists every path tried so misconfigurations are obvious.
    """
    tried: List[str] = []
    for root in _candidate_roots():
        cand = root / name
        tried.append(str(cand))
        if (cand / "netlist.pb.txt").exists():
            return cand

    # Last resort: shallow recursive search from repo root. Bounded by the
    # `name` segment so we don't crawl arbitrary directories.
    for found in _REPO_ROOT.rglob(f"{name}/netlist.pb.txt"):
        return found.parent

    raise FileNotFoundError(
        f"Benchmark '{name}': no source dir found.\n"
        f"  Tried: {tried}\n"
        f"  Set $BENCH_ROOT to override the search root."
    )
