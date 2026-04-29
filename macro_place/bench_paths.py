"""Generic benchmark-directory discovery for placer reload of PlacementCost.

Problem: the eval harness calls ``placer.place(benchmark)`` with only the
``Benchmark`` dataclass — no path, no plc. Many of our placers (CDAdaptive,
CDOnly, SDF init) need to reload the underlying ``PlacementCost`` to access
fields the dataclass doesn't carry (per-pin info, smoothing range, etc.).

This module supplies a single helper, ``find_benchmark_dir(name)``, that
locates the source directory for a benchmark in a generic way. For each
candidate root we try two layouts:

  • ``<root>/<name>/``                                    (IBM ICCAD04)
  • ``<root>/<name>/netlist/output_CT_Grouping/``        (NG45 Flows)

Candidate roots, in order:

  1. ``$BENCH_ROOT`` environment variable, if set.
  2. ``external/MacroPlacement/Testcases/ICCAD04``       (IBM)
  3. ``external/MacroPlacement/Testcases/NG45``          (reserved)
  4. ``external/MacroPlacement/Flows/NanGate45``         (public NG45)

No rglob fallback: the TILOS submodule contains test fixtures with the same
``{name}/netlist.pb.txt`` layout (e.g. ``CodeElements/Plc_client/test/
ariane133/``) that have *different* grid/routing parameters than the real
designs. Falling through to rglob silently loaded the wrong netlist.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent  # macro_place/bench_paths.py → repo root

# Sub-paths to try beneath each candidate root. Empty string = root itself.
_LAYOUT_SUFFIXES = ("", "netlist/output_CT_Grouping")


def _candidate_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("BENCH_ROOT")
    if env:
        roots.append(Path(env))
    mp = _REPO_ROOT / "external" / "MacroPlacement"
    roots.extend([
        mp / "Testcases" / "ICCAD04",
        mp / "Testcases" / "NG45",
        mp / "Flows" / "NanGate45",
    ])
    return roots


def find_benchmark_dir(name: str) -> Path:
    """Return the source directory for a benchmark by name.

    Raises FileNotFoundError if no candidate path resolves. The error
    message lists every path tried so misconfigurations are obvious.
    """
    tried: List[str] = []
    for root in _candidate_roots():
        for suffix in _LAYOUT_SUFFIXES:
            cand = root / name / suffix if suffix else root / name
            tried.append(str(cand))
            if (cand / "netlist.pb.txt").exists():
                return cand

    raise FileNotFoundError(
        f"Benchmark '{name}': no source dir found.\n"
        f"  Tried: {tried}\n"
        f"  Set $BENCH_ROOT to override the search root."
    )
