"""
Evaluation harness for macro placement submissions.

Loads a placer from a Python file, runs it on benchmarks, and prints results
with baseline comparisons.

Usage:
    uv run evaluate submissions/examples/greedy_row_placer.py
    uv run evaluate submissions/examples/greedy_row_placer.py --all
    uv run evaluate submissions/examples/greedy_row_placer.py -b ibm03
    uv run evaluate submissions/examples/greedy_row_placer.py --fast --json
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement

# ── IBM ICCAD04 benchmark list ──────────────────────────────────────────────

IBM_BENCHMARKS = [
    "ibm01",
    "ibm02",
    "ibm03",
    "ibm04",
    "ibm06",
    "ibm07",
    "ibm08",
    "ibm09",
    "ibm10",
    "ibm11",
    "ibm12",
    "ibm13",
    "ibm14",
    "ibm15",
    "ibm16",
    "ibm17",
    "ibm18",
]

# ── NG45 commercial designs ────────────────────────────────────────────────

NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}

BENCHMARKS = IBM_BENCHMARKS

# ── Published baselines ─────────────────────────────────────────────────────

SA_BASELINES = {
    "ibm01": 1.3166,
    "ibm02": 1.9072,
    "ibm03": 1.7401,
    "ibm04": 1.5037,
    "ibm06": 2.5057,
    "ibm07": 2.0229,
    "ibm08": 1.9239,
    "ibm09": 1.3875,
    "ibm10": 2.1108,
    "ibm11": 1.7111,
    "ibm12": 2.8261,
    "ibm13": 1.9141,
    "ibm14": 2.2750,
    "ibm15": 2.3000,
    "ibm16": 2.2337,
    "ibm17": 3.6726,
    "ibm18": 2.7755,
}

REPLACE_BASELINES = {
    "ibm01": 0.9976,
    "ibm02": 1.8370,
    "ibm03": 1.3222,
    "ibm04": 1.3024,
    "ibm06": 1.6187,
    "ibm07": 1.4633,
    "ibm08": 1.4285,
    "ibm09": 1.1194,
    "ibm10": 1.5009,
    "ibm11": 1.1774,
    "ibm12": 1.7261,
    "ibm13": 1.3355,
    "ibm14": 1.5436,
    "ibm15": 1.5159,
    "ibm16": 1.4780,
    "ibm17": 1.6446,
    "ibm18": 1.7722,
}

# ── Fast validation subset ─────────────────────────────────────────────────
# Selected for diversity: ibm01 (small, low baseline), ibm04 (medium),
# ibm09 (lowest baselines), ibm13 (mid-range, 382 macros).
# Covers the range of difficulty while keeping iteration under 15s.

FAST_BENCHMARKS = ["ibm01", "ibm04", "ibm09", "ibm13"]

# ── Placer loading ───────────────────────────────────────────────────────────


def _load_placer(path: Path):
    """Import a placer .py file and return an instance of its placer class.

    Convention: the first class defined in the file that has a ``place``
    method is treated as the placer.  It is instantiated with no arguments.
    """
    path = path.resolve()
    if spec := importlib.util.spec_from_file_location(path.stem, str(path)):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        raise RuntimeError(f"Failed to load placer from {path}")

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()

    raise RuntimeError(
        f"No placer class found in {path}.\n"
        "Expected a class with a  place(self, benchmark) -> Tensor  method."
    )


# ── Single-benchmark evaluation ─────────────────────────────────────────────


def evaluate_benchmark(placer, name: str, testcase_root: str, ng45_dir: str = None) -> dict:
    """Run *placer* on a single benchmark and return a results dict."""
    if ng45_dir:
        netlist_file = f"{ng45_dir}/netlist.pb.txt"
        plc_file = f"{ng45_dir}/initial.plc"
        benchmark, plc = load_benchmark(netlist_file, plc_file, name=name)
    else:
        benchmark_dir = f"{testcase_root}/{name}"
        benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    start = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - start

    is_valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)

    return {
        "name": name,
        "proxy_cost": costs["proxy_cost"],
        "wirelength": costs["wirelength_cost"],
        "density": costs["density_cost"],
        "congestion": costs["congestion_cost"],
        "overlaps": costs["overlap_count"],
        "runtime": runtime,
        "valid": is_valid,
        "sa_baseline": SA_BASELINES.get(name),
        "replace_baseline": REPLACE_BASELINES.get(name),
        "placement": placement,
        "benchmark": benchmark,
        "plc": plc,
    }


# ── Pretty-printing ─────────────────────────────────────────────────────────


def _print_summary_table(results):
    """Print a multi-benchmark comparison table."""
    has_baselines = any(r["sa_baseline"] is not None for r in results)

    print()
    print("-" * 80)
    if has_baselines:
        print(
            f"{'Benchmark':>13}  {'Proxy':>8}  {'SA':>8}  {'RePlAce':>8}"
            f"  {'vs SA':>8}  {'vs RePlAce':>10}  {'Overlaps':>8}"
        )
    else:
        print(
            f"{'Benchmark':>13}  {'Proxy':>8}  {'WL':>8}  {'Density':>8}"
            f"  {'Congestion':>10}  {'Overlaps':>8}"
        )
    print("-" * 80)

    for r in results:
        if has_baselines:
            vs_sa = (
                ((r["sa_baseline"] - r["proxy_cost"]) / r["sa_baseline"] * 100)
                if r["sa_baseline"]
                else 0
            )
            vs_rep = (
                ((r["replace_baseline"] - r["proxy_cost"]) / r["replace_baseline"] * 100)
                if r["replace_baseline"]
                else 0
            )
            sa_str = f"{r['sa_baseline']:>8.4f}" if r["sa_baseline"] else f"{'—':>8}"
            rep_str = f"{r['replace_baseline']:>8.4f}" if r["replace_baseline"] else f"{'—':>8}"
            print(
                f"{r['name']:>13}  {r['proxy_cost']:>8.4f}"
                f"  {sa_str}  {rep_str}"
                f"  {vs_sa:>+7.1f}%  {vs_rep:>+9.1f}%  {r['overlaps']:>8}"
            )
        else:
            print(
                f"{r['name']:>13}  {r['proxy_cost']:>8.4f}"
                f"  {r['wirelength']:>8.3f}  {r['density']:>8.3f}"
                f"  {r['congestion']:>10.3f}  {r['overlaps']:>8}"
            )

    avg_proxy = sum(r["proxy_cost"] for r in results) / len(results)
    total_overlaps = sum(r["overlaps"] for r in results)
    total_runtime = sum(r["runtime"] for r in results)

    if has_baselines:
        baselines_sa = [r for r in results if r["sa_baseline"] is not None]
        baselines_rep = [r for r in results if r["replace_baseline"] is not None]
        avg_sa = sum(r["sa_baseline"] for r in baselines_sa) / len(baselines_sa) if baselines_sa else 0
        avg_rep = sum(r["replace_baseline"] for r in baselines_rep) / len(baselines_rep) if baselines_rep else 0
        print("-" * 80)
        print(
            f"{'AVG':>13}  {avg_proxy:>8.4f}  {avg_sa:>8.4f}  {avg_rep:>8.4f}"
            f"  {(avg_sa - avg_proxy) / avg_sa * 100:>+7.1f}%"
            f"  {(avg_rep - avg_proxy) / avg_rep * 100:>+9.1f}%  {total_overlaps:>8}"
        )
    else:
        avg_wl = sum(r["wirelength"] for r in results) / len(results)
        avg_den = sum(r["density"] for r in results) / len(results)
        avg_cong = sum(r["congestion"] for r in results) / len(results)
        print("-" * 80)
        print(
            f"{'AVG':>13}  {avg_proxy:>8.4f}"
            f"  {avg_wl:>8.3f}  {avg_den:>8.3f}"
            f"  {avg_cong:>10.3f}  {total_overlaps:>8}"
        )

    print()
    print(f"Total runtime: {total_runtime:.2f}s")
    if total_overlaps > 0:
        print(f"⚠  DISQUALIFIED: {total_overlaps} total overlaps across benchmarks")
    print()


# ── Structured output ──────────────────────────────────────────────────────


def _build_json_output(results, placer_name: str, placer_path: str, mode: str) -> dict:
    """Build a structured dict from evaluation results."""
    benchmarks = []
    for r in results:
        benchmarks.append({
            "name": r["name"],
            "proxy_cost": float(r["proxy_cost"]),
            "wirelength": float(r["wirelength"]),
            "density": float(r["density"]),
            "congestion": float(r["congestion"]),
            "overlap_count": int(r["overlaps"]),
            "runtime_seconds": round(float(r["runtime"]), 4),
            "sa_baseline": float(r["sa_baseline"]) if r["sa_baseline"] is not None else None,
            "replace_baseline": float(r["replace_baseline"]) if r["replace_baseline"] is not None else None,
        })

    avg_proxy = float(sum(float(r["proxy_cost"]) for r in results) / len(results))
    total_overlaps = int(sum(int(r["overlaps"]) for r in results))
    total_runtime = float(sum(float(r["runtime"]) for r in results))

    return {
        "placer_name": placer_name,
        "placer_path": placer_path,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
        "avg_proxy_cost": round(avg_proxy, 6),
        "total_overlaps": total_overlaps,
        "total_runtime_seconds": round(total_runtime, 4),
        "qualified": total_overlaps == 0,
    }


def _write_json_result(data: dict, results_dir: Path):
    """Write structured results to results/<placer>_<timestamp>.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = data["placer_name"]
    out_path = results_dir / f"{name}_{ts}.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Results written to {out_path}")
    return out_path


def _append_experiment_log(data: dict, log_path: Path):
    """Append a single-line JSON record to the experiment log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "hypothesis": data.get("hypothesis", ""),
        "variant_name": data["placer_name"],
        "placer_path": data["placer_path"],
        "mode": data["mode"],
        "avg_proxy_cost": data["avg_proxy_cost"],
        "total_overlaps": data["total_overlaps"],
        "qualified": data["qualified"],
        "per_benchmark": {b["name"]: b["proxy_cost"] for b in data["benchmarks"]},
        "runtime_seconds": data["total_runtime_seconds"],
        "timestamp": data["timestamp"],
        "status": "exploring",
        "notes": "",
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged to {log_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate a macro-placement submission on IBM ICCAD04 benchmarks.",
    )
    parser.add_argument(
        "placer",
        help="Path to a placer .py file (e.g. submissions/examples/greedy_row_placer.py).",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default=None,
        help="Run on a specific benchmark (e.g. ibm01). Default: ibm01.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run on all 17 IBM benchmarks.",
    )
    parser.add_argument(
        "--ng45",
        action="store_true",
        help="Run on NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run on fast validation subset (4 predictive IBM benchmarks).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write structured JSON results to results/ and append to experiment log.",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="",
        help="Hypothesis name for experiment log tagging (e.g. 'sdf_density').",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize each placement after evaluation (saves to vis/<benchmark>.png).",
    )
    args = parser.parse_args()

    # ── resolve paths ────────────────────────────────────────────────────
    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")
    if not args.ng45 and not testcase_root.exists():
        print(f"Error: Testcases not found at {testcase_root}")
        print("Run: git submodule update --init external/MacroPlacement")
        sys.exit(1)

    # ── load placer ──────────────────────────────────────────────────────
    placer_path = Path(args.placer)
    placer = _load_placer(placer_path)
    placer_name = type(placer).__name__

    # ── determine which benchmarks to run ────────────────────────────────
    if args.ng45:
        benchmarks_to_run = list(NG45_BENCHMARKS.keys())
        mode = "ng45"
    elif args.all:
        benchmarks_to_run = BENCHMARKS
        mode = "all"
    elif args.fast:
        benchmarks_to_run = FAST_BENCHMARKS
        mode = "fast"
    else:
        benchmarks_to_run = [args.benchmark or "ibm01"]
        mode = "single"

    # ── run ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"evaluate · {placer_name}  ({placer_path})")
    print("=" * 80)
    print()

    results = []
    for name in benchmarks_to_run:
        print(f"  {name}...", end=" ", flush=True)
        ng45_dir = NG45_BENCHMARKS.get(name) if args.ng45 or name in NG45_BENCHMARKS else None
        result = evaluate_benchmark(placer, name, str(testcase_root), ng45_dir=ng45_dir)
        results.append(result)

        status = (
            "VALID"
            if result["overlaps"] == 0
            else f"INVALID ({result['overlaps']} overlaps)"
        )
        print(
            f"proxy={result['proxy_cost']:.4f}  "
            f"(wl={result['wirelength']:.3f} den={result['density']:.3f} cong={result['congestion']:.3f})  "
            f"{status}  [{result['runtime']:.2f}s]"
        )

        if args.vis:
            vis_dir = Path("vis")
            vis_dir.mkdir(exist_ok=True)
            save_path = str(vis_dir / f"{name}.png")
            visualize_placement(result["placement"], result["benchmark"], save_path=save_path, plc=result.get("plc"))

    if len(results) > 1:
        _print_summary_table(results)

    # ── structured output ───────────────────────────────────────────────
    if args.json:
        data = _build_json_output(results, placer_name, str(placer_path), mode)
        if args.hypothesis:
            data["hypothesis"] = args.hypothesis
        results_dir = Path("results")
        _write_json_result(data, results_dir)
        _append_experiment_log(data, results_dir / "experiment_log.jsonl")


if __name__ == "__main__":
    main()
