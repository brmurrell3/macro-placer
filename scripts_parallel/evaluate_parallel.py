"""
Parallel benchmark-level evaluator for macro placement submissions.

This is a standalone variant of ``macro_place.evaluate``. It runs different
benchmarks in separate worker processes while keeping each benchmark's
PlacementCost object private to its worker.

Examples:
    uv run python scripts_parallel/evaluate_parallel.py submissions/examples/greedy_row_placer.py --fast --jobs 4
    uv run python scripts_parallel/evaluate_parallel.py submissions/examples/greedy_row_placer.py --all --jobs 8
    uv run python scripts_parallel/evaluate_parallel.py submissions/examples/greedy_row_placer.py -b ibm01
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# Ensure repo root is importable when this file is run by path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from macro_place.evaluate import (  # noqa: E402
    BENCHMARKS,
    FAST_BENCHMARKS,
    NG45_BENCHMARKS,
    REPLACE_BASELINES,
    SA_BASELINES,
    _append_experiment_log,
    _build_json_output,
    _load_placer,
    _print_summary_table,
    _write_json_result,
)
from macro_place.loader import load_benchmark, load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import visualize_placement  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402


def evaluate_benchmark(placer, name: str, testcase_root: str, ng45_dir: str = None) -> dict:
    if ng45_dir:
        netlist_file = f"{ng45_dir}/netlist.pb.txt"
        plc_file = f"{ng45_dir}/initial.plc"
        benchmark, plc = load_benchmark(netlist_file, plc_file)
    else:
        benchmark_dir = f"{testcase_root}/{name}"
        benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    import time

    start = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - start

    is_valid, _violations = validate_placement(placement, benchmark)
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
    }


def _status_line(result: dict[str, Any]) -> str:
    status = "VALID" if result["overlaps"] == 0 else f"INVALID ({result['overlaps']} overlaps)"
    return (
        f"proxy={result['proxy_cost']:.4f}  "
        f"(wl={result['wirelength']:.3f} den={result['density']:.3f} "
        f"cong={result['congestion']:.3f})  "
        f"{status}  [{result['runtime']:.2f}s]"
    )


def _strip_heavy_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Remove tensors and Benchmark objects before returning from workers."""
    slim = dict(result)
    slim.pop("placement", None)
    slim.pop("benchmark", None)
    return slim


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one benchmark in a child process.

    The function is module-level so Windows multiprocessing can spawn it.
    """
    if task["torch_threads"] is not None:
        try:
            import torch

            torch.set_num_threads(int(task["torch_threads"]))
        except Exception:
            pass

    placer = _load_placer(Path(task["placer_path"]))
    result = evaluate_benchmark(
        placer,
        task["name"],
        task["testcase_root"],
        ng45_dir=task["ng45_dir"],
    )

    if task["vis"]:
        vis_dir = Path(task["vis_dir"])
        vis_dir.mkdir(exist_ok=True)
        visualize_placement(
            result["placement"],
            result["benchmark"],
            save_path=str(vis_dir / f"{task['name']}.png"),
        )

    return _strip_heavy_fields(result)


def _run_parallel(
    placer_path: Path,
    benchmarks_to_run: list[str],
    testcase_root: Path,
    jobs: int,
    vis: bool,
    torch_threads: int | None,
    ng45_mode: bool,
) -> list[dict[str, Any]]:
    tasks = []
    for name in benchmarks_to_run:
        ng45_dir = NG45_BENCHMARKS.get(name) if ng45_mode or name in NG45_BENCHMARKS else None
        tasks.append(
            {
                "placer_path": str(placer_path),
                "name": name,
                "testcase_root": str(testcase_root),
                "ng45_dir": ng45_dir,
                "vis": vis,
                "vis_dir": "vis",
                "torch_threads": torch_threads,
            }
        )

    results_by_name: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_worker, task): task["name"] for task in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"Benchmark {name} failed in worker") from exc
            results_by_name[name] = result
            print(f"  {name}... {_status_line(result)}", flush=True)

    return [results_by_name[name] for name in benchmarks_to_run]


def _run_serial(
    placer,
    benchmarks_to_run: list[str],
    testcase_root: Path,
    vis: bool,
    ng45_mode: bool,
) -> list[dict[str, Any]]:
    results = []
    for name in benchmarks_to_run:
        print(f"  {name}...", end=" ", flush=True)
        ng45_dir = NG45_BENCHMARKS.get(name) if ng45_mode or name in NG45_BENCHMARKS else None
        result = evaluate_benchmark(placer, name, str(testcase_root), ng45_dir=ng45_dir)
        print(_status_line(result))

        if vis:
            vis_dir = Path("vis")
            vis_dir.mkdir(exist_ok=True)
            visualize_placement(
                result["placement"],
                result["benchmark"],
                save_path=str(vis_dir / f"{name}.png"),
            )

        results.append(_strip_heavy_fields(result))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate_parallel",
        description="Evaluate a macro-placement submission with benchmark-level multiprocessing.",
    )
    parser.add_argument("placer", help="Path to a placer .py file.")
    parser.add_argument("--benchmark", "-b", type=str, default=None)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark list, e.g. ibm01,ibm02. Overrides --benchmark.",
    )
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--ng45", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hypothesis", type=str, default="")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 for serial behavior.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch.set_num_threads value inside each worker. Use 0 to leave unchanged.",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    torch_threads = None if args.torch_threads == 0 else args.torch_threads

    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")
    if not args.ng45 and not testcase_root.exists():
        print(f"Error: Testcases not found at {testcase_root}")
        print("Run: git submodule update --init external/MacroPlacement")
        sys.exit(1)

    placer_path = Path(args.placer)
    placer = _load_placer(placer_path)
    placer_name = type(placer).__name__

    if args.ng45:
        benchmarks_to_run = list(NG45_BENCHMARKS.keys())
        mode = "ng45"
    elif args.all:
        benchmarks_to_run = BENCHMARKS
        mode = "all"
    elif args.fast:
        benchmarks_to_run = FAST_BENCHMARKS
        mode = "fast"
    elif args.benchmarks:
        benchmarks_to_run = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
        if not benchmarks_to_run:
            parser.error("--benchmarks did not contain any benchmark names")
        mode = "custom"
    else:
        benchmarks_to_run = [args.benchmark or "ibm01"]
        mode = "single"

    print("=" * 80)
    print(f"evaluate_parallel - {placer_name} ({placer_path})")
    print(f"mode={mode} jobs={args.jobs} torch_threads={torch_threads}")
    print("=" * 80)
    print()

    if args.jobs == 1 or len(benchmarks_to_run) == 1:
        results = _run_serial(placer, benchmarks_to_run, testcase_root, args.vis, args.ng45)
    else:
        results = _run_parallel(
            placer_path,
            benchmarks_to_run,
            testcase_root,
            args.jobs,
            args.vis,
            torch_threads,
            args.ng45,
        )

    if len(results) > 1:
        _print_summary_table(results)

    if args.json:
        data = _build_json_output(results, placer_name, str(placer_path), mode)
        if args.hypothesis:
            data["hypothesis"] = args.hypothesis
        data["jobs"] = args.jobs
        data["torch_threads"] = torch_threads
        results_dir = Path("results")
        _write_json_result(data, results_dir)
        _append_experiment_log(data, results_dir / "experiment_log.jsonl")


if __name__ == "__main__":
    # Keep this guard: Windows uses spawn for multiprocessing.
    main()
