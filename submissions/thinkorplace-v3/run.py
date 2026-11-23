"""Place one design with thinkorplace v3 and write <out>/result.json with the reference scorer's proxy.

    .venv/bin/python submissions/thinkorplace-v3/run.py --design ibm01 --out <dir> [--budget 720]

The clock starts when this file starts, so start-up, imports and loading are
inside the budget. After the deadline the chosen placement is validated and
scored by the reference scorer on a fresh load of the design.
"""
import time

T0 = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402
from thinkorplace.common.bench_paths import find_benchmark_dir  # noqa: E402
from thinkorplace.v3.chain import load_design, write_json  # noqa: E402
from thinkorplace.v3.scheduler import run_chains  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--budget", type=float, default=720.0, help="wall-clock seconds")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    directory = str(find_benchmark_dir(args.design))
    positions, summary = run_chains(directory, args.out, T0, args.budget)
    result = {"design": args.design, "budget_seconds": args.budget, **summary,
              "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}}
    if positions is None:
        result.update(status="no placement published before the deadline", proxy=None)
    else:
        np.save(args.out / "placement.npy", positions.numpy())
        bench, plc = load_design(directory)
        valid, violations = validate_placement(positions, bench)
        costs = compute_proxy_cost(positions, bench, plc)
        valid = valid and int(costs["overlap_count"]) == 0
        result.update(status="valid" if valid else "invalid", violations=[str(v) for v in violations][:5],
                      proxy=float(costs["proxy_cost"]) if valid else None,
                      wirelength=float(costs["wirelength_cost"]), density=float(costs["density_cost"]),
                      congestion=float(costs["congestion_cost"]), overlaps=int(costs["overlap_count"]))
    result["wall_seconds_with_audit"] = time.time() - T0
    write_json(args.out / "result.json", result)
    print(json.dumps({key: result.get(key) for key in ("design", "status", "proxy", "decided_at_seconds")}), flush=True)


if __name__ == "__main__":
    main()
