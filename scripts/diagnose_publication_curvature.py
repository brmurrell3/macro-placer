#!/usr/bin/env python3
"""Bounded curvature diagnostic for the frozen E74 IBM01 endpoint.

This script intentionally evaluates the differentiable E74 SmoothProxy at the
saved E74 output placement.  It does not rerun E74, legalize perturbations, or
apply coordinate-descent polishing.  The frozen publication source archive is
extracted into a temporary directory and imported from there so working-tree
changes cannot alter the proxy implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
import torch


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "writeup/data/publication/competition-source.tar"
SNAPSHOT = REPO / "writeup/data/publication/source_snapshot.json"
STATE_PT = REPO / "writeup/data/publication/raw/placements/hessian_ibm01.pt"
OUTPUT = REPO / "writeup/data/publication/curvature_diagnostics_ibm01.json"
TEX_OUTPUT = REPO / "writeup/latex/generated/curvature_diagnostics.tex"
E74_MEMBER = "experiments/E74_hessian_saddle/code/hessian_saddle.py"
DPO_MEMBER = "writeup/archive/submissions/dpo/ablation_v2_steps.py"
EPSILONS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SEED = 7401


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(header + b"\0" + value.numpy().tobytes(order="C"))


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(REPO), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def extract_frozen_source(destination: Path) -> None:
    with tarfile.open(ARCHIVE, "r") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)
    os.symlink(REPO / "external", destination / "external", target_is_directory=True)


def load_frozen_modules(root: Path):
    # The script directory, rather than REPO, is sys.path[0], but remove any
    # explicit repository-root entries as an additional isolation guard.
    repo_string = str(REPO)
    sys.path[:] = [entry for entry in sys.path if entry not in ("", repo_string)]
    sys.path.insert(0, str(root))

    module_path = root / E74_MEMBER
    spec = importlib.util.spec_from_file_location("publication_frozen_e74", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir

    benchmark_path = find_benchmark_dir("ibm01")
    benchmark, plc = load_benchmark_from_dir(str(benchmark_path))
    return module, benchmark, plc, benchmark_path


def value_and_gradient(proxy, state: torch.Tensor) -> Tuple[float, torch.Tensor, float]:
    started = time.perf_counter()
    x = state.detach().clone().requires_grad_(True)
    value = proxy.cost(x)
    gradient = torch.autograd.grad(value, x, create_graph=False)[0]
    return float(value.detach()), gradient.detach(), time.perf_counter() - started


def clamp_counts(proxy, state: torch.Tensor) -> Dict[str, int]:
    low_x = proxy.half_sizes[:, 0]
    high_x = proxy.cw - proxy.half_sizes[:, 0]
    low_y = proxy.half_sizes[:, 1]
    high_y = proxy.ch - proxy.half_sizes[:, 1]
    return {
        "x_below_or_at_lower": int((state[:, 0] <= low_x).sum()),
        "x_above_or_at_upper": int((state[:, 0] >= high_x).sum()),
        "y_below_or_at_lower": int((state[:, 1] <= low_y).sum()),
        "y_above_or_at_upper": int((state[:, 1] >= high_y).sum()),
    }


def finite_or_none(value: float) -> Any:
    return float(value) if math.isfinite(float(value)) else None


def relative_error(error: float, reference: float) -> Any:
    if reference == 0.0:
        return None
    return finite_or_none(error / abs(reference))


def direction_diagnostic(
    proxy,
    e74_module,
    state: torch.Tensor,
    baseline_value: float,
    baseline_gradient: torch.Tensor,
    movable: torch.Tensor,
    seed: int,
    epsilons: Tuple[float, ...],
) -> Tuple[Dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(int(movable.sum()) * 2)
    raw /= np.linalg.norm(raw)
    direction = torch.zeros_like(state)
    direction[movable] = torch.from_numpy(raw.reshape(-1, 2)).to(state.dtype)
    direction /= torch.linalg.vector_norm(direction)

    started = time.perf_counter()
    hvp = e74_module.hessian_vector_product(proxy, state, direction)
    hvp_wall = time.perf_counter() - started
    movable_gradient = baseline_gradient[movable]
    movable_hvp = hvp[movable]
    g_dot_v = float(torch.sum(movable_gradient * direction[movable]))
    v_h_v = float(torch.sum(direction[movable] * movable_hvp))
    hvp_norm = float(torch.linalg.vector_norm(movable_hvp))

    scales: List[Dict[str, Any]] = []
    for epsilon in epsilons:
        plus = state + epsilon * direction
        minus = state - epsilon * direction
        plus_value, plus_gradient, plus_wall = value_and_gradient(proxy, plus)
        minus_value, minus_gradient, minus_wall = value_and_gradient(proxy, minus)

        first_fd = (plus_value - minus_value) / (2.0 * epsilon)
        # Gradient central differences avoid the severe float32 cancellation of
        # scalar second differences, but both are retained as raw diagnostics.
        hvp_fd = (plus_gradient - minus_gradient) / (2.0 * epsilon)
        hvp_fd_movable = hvp_fd[movable]
        directional_curvature_grad_fd = float(
            torch.sum(direction[movable] * hvp_fd_movable)
        )
        scalar_second_fd = (
            plus_value - 2.0 * baseline_value + minus_value
        ) / (epsilon * epsilon)
        vector_error = hvp_fd_movable - movable_hvp
        vector_error_norm = float(torch.linalg.vector_norm(vector_error))
        fd_norm = float(torch.linalg.vector_norm(hvp_fd_movable))
        denom = hvp_norm * fd_norm
        cosine = (
            float(torch.sum(movable_hvp * hvp_fd_movable)) / denom
            if denom > 0.0
            else None
        )
        actual_plus = plus - state
        actual_minus = state - minus
        scales.append(
            {
                "epsilon_microns_l2": epsilon,
                "cost_plus": plus_value,
                "cost_base": baseline_value,
                "cost_minus": minus_value,
                "cost_central_first_derivative": first_fd,
                "autodiff_directional_gradient": g_dot_v,
                "directional_gradient_abs_error": abs(first_fd - g_dot_v),
                "directional_gradient_relative_error": relative_error(
                    abs(first_fd - g_dot_v), g_dot_v
                ),
                "directional_curvature_from_gradient_fd": directional_curvature_grad_fd,
                "directional_curvature_from_scalar_cost_fd": scalar_second_fd,
                "autodiff_directional_curvature_vHv": v_h_v,
                "directional_curvature_gradient_fd_abs_error": abs(
                    directional_curvature_grad_fd - v_h_v
                ),
                "directional_curvature_gradient_fd_relative_error": relative_error(
                    abs(directional_curvature_grad_fd - v_h_v), v_h_v
                ),
                "hvp_fd_l2_norm_movable": fd_norm,
                "autodiff_hvp_l2_norm_movable": hvp_norm,
                "hvp_vector_abs_error_l2": vector_error_norm,
                "hvp_vector_relative_error_l2": (
                    relative_error(vector_error_norm, hvp_norm)
                ),
                "hvp_vector_cosine": finite_or_none(cosine) if cosine is not None else None,
                "actual_plus_displacement_l2": float(
                    torch.linalg.vector_norm(actual_plus[movable])
                ),
                "actual_minus_displacement_l2": float(
                    torch.linalg.vector_norm(actual_minus[movable])
                ),
                "plus_clamp_counts": clamp_counts(proxy, plus),
                "minus_clamp_counts": clamp_counts(proxy, minus),
                "plus_value_and_gradient_wall_seconds": plus_wall,
                "minus_value_and_gradient_wall_seconds": minus_wall,
            }
        )

    result = {
        "seed": seed,
        "generator": "numpy.random.default_rng(seed).standard_normal; normalize in float64; cast to state dtype; renormalize",
        "direction_sha256": tensor_hash(direction),
        "direction_dtype": str(direction.dtype),
        "direction_l2_norm_all_coordinates": float(torch.linalg.vector_norm(direction)),
        "direction_nonzero_fixed_coordinates": int(torch.count_nonzero(direction[~movable])),
        "autodiff_directional_gradient": g_dot_v,
        "autodiff_hvp_l2_norm_movable": hvp_norm,
        "autodiff_directional_curvature_vHv": v_h_v,
        "autodiff_hvp_wall_seconds": hvp_wall,
        "scales": scales,
    }
    return result, hvp_wall


class DeadlineReached(RuntimeError):
    pass


def bounded_lanczos(
    proxy,
    e74_module,
    state: torch.Tensor,
    movable: torch.Tensor,
    seed: int,
    max_hvps: int,
    deadline_seconds: float,
    measured_hvp_seconds: float,
) -> Dict[str, Any]:
    projected = measured_hvp_seconds * max_hvps
    krylov_hvp_cap = max(1, max_hvps - 1)  # reserve one HVP for residual validation
    metadata: Dict[str, Any] = {
        "method": "full-reorthogonalized symmetric Lanczos; smallest Ritz value",
        "seed": seed,
        "hvp_cap": max_hvps,
        "deadline_seconds": deadline_seconds,
        "measured_mean_prior_hvp_seconds": measured_hvp_seconds,
        "projected_hvp_seconds": projected,
        "status": "not_run",
    }
    if projected > 0.85 * deadline_seconds:
        metadata["status"] = "skipped_projected_runtime"
        return metadata

    n_free = int(movable.sum()) * 2
    rng = np.random.default_rng(seed)
    q_np = rng.standard_normal(n_free)
    q_np /= np.linalg.norm(q_np)
    q = torch.from_numpy(q_np.reshape(-1, 2)).to(state.dtype)
    q /= torch.linalg.vector_norm(q)

    basis: List[torch.Tensor] = []
    alphas: List[float] = []
    betas: List[float] = []
    previous = torch.zeros_like(q)
    beta_previous = 0.0
    last_beta = None
    started = time.perf_counter()

    try:
        for iteration in range(krylov_hvp_cap):
            if time.perf_counter() - started >= deadline_seconds:
                raise DeadlineReached
            full_q = torch.zeros_like(state)
            full_q[movable] = q
            hv_full = e74_module.hessian_vector_product(proxy, state, full_q)
            z = hv_full[movable] - beta_previous * previous
            alpha = float(torch.sum(q * z))
            z = z - alpha * q
            # A second pass is cheap compared with an HVP and stabilizes the
            # small float32 Krylov basis used for this bounded diagnostic.
            for existing in basis:
                z = z - torch.sum(existing * z) * existing
            for existing in basis:
                z = z - torch.sum(existing * z) * existing
            beta = float(torch.linalg.vector_norm(z))

            basis.append(q.clone())
            alphas.append(alpha)
            last_beta = beta
            if iteration == krylov_hvp_cap - 1 or beta <= 1e-12:
                break
            betas.append(beta)
            previous = q
            q = z / beta
            beta_previous = beta
    except DeadlineReached:
        metadata["status"] = "deadline_reached"

    count = len(alphas)
    if count == 0:
        metadata.update({"hvp_count": 0, "wall_seconds": time.perf_counter() - started})
        return metadata
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if count > 1:
        offdiag = np.asarray(betas[: count - 1], dtype=np.float64)
        tridiagonal += np.diag(offdiag, 1) + np.diag(offdiag, -1)
    values, vectors = np.linalg.eigh(tridiagonal)
    index = int(np.argmin(values))
    coeff = vectors[:, index]
    ritz = sum(float(coeff[i]) * basis[i] for i in range(count))
    ritz /= torch.linalg.vector_norm(ritz)
    residual_estimate = abs(float(last_beta or 0.0) * float(coeff[-1]))
    if metadata["status"] == "not_run":
        metadata["status"] = "completed" if count == krylov_hvp_cap else "happy_breakdown"
    actual_rayleigh = None
    actual_residual = None
    total_hvps = count
    if time.perf_counter() - started < deadline_seconds and total_hvps < max_hvps:
        full_ritz = torch.zeros_like(state)
        full_ritz[movable] = ritz
        h_ritz = e74_module.hessian_vector_product(proxy, state, full_ritz)[movable]
        total_hvps += 1
        actual_rayleigh = float(torch.sum(ritz * h_ritz))
        actual_residual = float(
            torch.linalg.vector_norm(h_ritz - actual_rayleigh * ritz)
        )
    metadata.update(
        {
            "hvp_count": total_hvps,
            "krylov_hvp_count": count,
            "wall_seconds": time.perf_counter() - started,
            "smallest_ritz_value": float(values[index]),
            "ritz_residual_l2_estimate": residual_estimate,
            "validated_rayleigh_quotient": actual_rayleigh,
            "validated_residual_l2": actual_residual,
            "ritz_vector_sha256": tensor_hash(ritz),
            "tridiagonal_diagonal": [float(x) for x in alphas],
            "tridiagonal_off_diagonal": [float(x) for x in betas[: count - 1]],
            "all_ritz_values": [float(x) for x in values],
            "interpretation": "Approximate low-curvature Ritz pair from a capped Krylov subspace, not a converged extremal eigenpair unless the residual supports that reading.",
        }
    )
    return metadata


def write_latex(report: Dict[str, Any], path: Path) -> None:
    baseline = report["baseline"]
    lanczos = report["bounded_lanczos"]
    rows = []
    hvp_relative_errors = []
    hvp_cosines = []
    for direction_index, direction in enumerate(report["random_directions"], start=1):
        for scale in direction["scales"]:
            hvp_relative_errors.append(scale["hvp_vector_relative_error_l2"])
            hvp_cosines.append(scale["hvp_vector_cosine"])
            rows.append(
                f"{direction_index} & {scale['epsilon_microns_l2']:.0e} & "
                f"{direction['autodiff_directional_gradient']:.3e} & "
                f"{scale['cost_central_first_derivative']:.3e} & "
                f"{direction['autodiff_directional_curvature_vHv']:.3e} & "
                f"{scale['directional_curvature_from_gradient_fd']:.3e} \\\\"
            )
    residual = lanczos.get("validated_residual_l2")
    rayleigh = lanczos.get("validated_rayleigh_quotient")
    latex = (
        "\\paragraph{Bounded IBM01 endpoint curvature diagnostic.}\n"
        "The frozen legacy E74 proxy evaluated at the saved final E74 endpoint has "
        f"$J_B={baseline['smooth_proxy_value']:.8f}$ and raw movable-gradient norm "
        f"$\\lVert\\nabla J_B\\rVert_2={baseline['movable_gradient_l2_norm']:.6f}$. "
        f"A {lanczos.get('hvp_count', 0)}-HVP capped Lanczos probe returned a "
        f"Rayleigh quotient of ${rayleigh:.6f}$ with residual norm ${residual:.2e}$, "
        "checked using the same automatic-differentiation HVP operator. "
        "This is a negative-curvature mode of the automatic-differentiation active "
        "branch, not evidence that the endpoint is a constrained stationary saddle.\n\n"
        "\\begin{table}[htbp]\\centering\\small\n"
        "\\begin{tabular}{rrrrrr}\n"
        "\\toprule\n"
        "$v$ & $\\epsilon$ ($\\mu$m) & $\\nabla J_B^T v$ & cost FD & "
        "$v^T H v$ & gradient FD \\\\ \n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Central finite differences along two seeded random movable-coordinate "
        "unit directions at the saved final E74 IBM01 endpoint. Cost FD denotes the "
        "symmetric first difference; gradient FD denotes the directional projection "
        "of the symmetric gradient difference. The vector-HVP relative errors span "
        f"{min(hvp_relative_errors):.2f}--{max(hvp_relative_errors):.2f}, with cosine "
        f"agreement {min(hvp_cosines):.3f}--{max(hvp_cosines):.3f}. The cause of the "
        "discrepancy is unresolved: float32 cancellation, crossings of piecewise "
        "branches, and operator inconsistency have not been separated. These finite "
        "differences therefore do not independently validate the HVP operator.}\n"
        "\\label{tab:ibm01-curvature-diagnostic}\n"
        "\\end{table}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tex-output", type=Path, default=TEX_OUTPUT)
    parser.add_argument("--skip-lanczos", action="store_true")
    parser.add_argument("--lanczos-hvp-cap", type=int, default=14)
    parser.add_argument("--lanczos-deadline-seconds", type=float, default=105.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_threads)
    overall_started = time.perf_counter()

    snapshot = json.loads(SNAPSHOT.read_text())
    with tempfile.TemporaryDirectory(prefix="curvature-frozen-") as temporary:
        frozen_root = Path(temporary)
        extract_frozen_source(frozen_root)
        e74, benchmark, plc, benchmark_path = load_frozen_modules(frozen_root)

        saved = torch.load(STATE_PT, map_location="cpu", weights_only=False)
        state = saved["placement"].detach().cpu().contiguous()
        if tuple(state.shape) != (benchmark.num_macros, 2):
            raise RuntimeError(
                f"placement shape {tuple(state.shape)} != benchmark {(benchmark.num_macros, 2)}"
            )
        proxy = e74.SmoothProxy(benchmark, plc)
        movable = (~benchmark.macro_fixed).cpu()
        baseline_value, baseline_gradient, baseline_wall = value_and_gradient(proxy, state)
        movable_gradient = baseline_gradient[movable]

        direction_results = []
        hvp_times = []
        for offset in range(2):
            result, hvp_wall = direction_diagnostic(
                proxy,
                e74,
                state,
                baseline_value,
                baseline_gradient,
                movable,
                SEED + offset,
                EPSILONS,
            )
            direction_results.append(result)
            hvp_times.append(hvp_wall)

        if args.skip_lanczos:
            lanczos = {"status": "skipped_by_argument"}
        else:
            lanczos = bounded_lanczos(
                proxy,
                e74,
                state,
                movable,
                SEED + 100,
                args.lanczos_hvp_cap,
                args.lanczos_deadline_seconds,
                float(np.mean(hvp_times)),
            )

        imported_macro_place = sys.modules["macro_place"]
        imported_dpo = e74._dpo
        benchmark_files = {
            name: sha256_file(benchmark_path / name)
            for name in ("netlist.pb.txt", "initial.plc")
        }
        historical_stats = saved.get("stats", {})
        report: Dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "benchmark": "ibm01",
                "evaluation_point": "saved final E74 endpoint placement",
                "state_path": str(STATE_PT.relative_to(REPO)),
                "no_optimization_or_legalization": True,
                "claim_boundary": "Local numerical diagnostic of one surrogate at one saved endpoint; not a classical-Hessian proof or a multi-benchmark ablation.",
            },
            "provenance": {
                "repository_head": git_value("rev-parse", "HEAD"),
                "publication_snapshot_commit": snapshot["commit"],
                "archive_path": str(ARCHIVE.relative_to(REPO)),
                "archive_sha256": sha256_file(ARCHIVE),
                "snapshot_declared_archive_sha256": snapshot["archive_sha256"],
                "frozen_e74_member": E74_MEMBER,
                "frozen_e74_member_sha256": sha256_file(frozen_root / E74_MEMBER),
                "frozen_dpo_member": DPO_MEMBER,
                "frozen_dpo_member_sha256": sha256_file(frozen_root / DPO_MEMBER),
                "imported_e74_file": str(Path(e74.__file__).resolve().relative_to(frozen_root.resolve())),
                "imported_macro_place_file": str(Path(imported_macro_place.__file__).resolve().relative_to(frozen_root.resolve())),
                "imported_dpo_file": (
                    str(Path(imported_dpo.__file__).resolve().relative_to(frozen_root.resolve()))
                    if imported_dpo is not None else None
                ),
                "external_benchmark_symlink_target": str((REPO / "external").resolve()),
                "benchmark_directory": str(benchmark_path),
                "benchmark_input_sha256": benchmark_files,
                "state_pt_sha256": sha256_file(STATE_PT),
                "placement_tensor_sha256": tensor_hash(state),
            },
            "surrogate": {
                "class": "experiments/E74_hessian_saddle/code/hessian_saddle.py::SmoothProxy",
                "formula": "normalized LSE HPWL + 0.5 * rectangle-grid density + 0.5 * RUDY congestion",
                "gamma_fraction_of_canvas_width": 0.0005,
                "gamma_microns": float(proxy.gamma),
                "nonsmooth_operations": [
                    "canvas clamp",
                    "rectangle/cell min-max and zero clamps",
                    "RUDY bounding-box minimum-width clamps",
                    "density top-10% selection",
                    "congestion top-5% selection",
                ],
                "not_gaussian_density": True,
            },
            "state": {
                "shape": list(state.shape),
                "dtype": str(state.dtype),
                "device": str(state.device),
                "num_macros": int(benchmark.num_macros),
                "num_hard_macros": int(benchmark.num_hard_macros),
                "num_movable_macros": int(movable.sum()),
                "num_movable_coordinates": int(movable.sum()) * 2,
                "canvas_width_microns": float(benchmark.canvas_width),
                "canvas_height_microns": float(benchmark.canvas_height),
                "grid_rows": int(benchmark.grid_rows),
                "grid_columns": int(benchmark.grid_cols),
                "baseline_clamp_counts": clamp_counts(proxy, state),
            },
            "historical_saved_metadata": {
                "note": "These values were computed inside the original E74 run before its perturb-and-polish attempts; they do not describe the final saved endpoint evaluated here.",
                "starting_proxy": historical_stats.get("starting_proxy"),
                "best_proxy": historical_stats.get("best_proxy"),
                "eigenvalues_at_original_E74_start": historical_stats.get("eigenvalues"),
            },
            "baseline": {
                "smooth_proxy_value": baseline_value,
                "gradient_dtype": str(baseline_gradient.dtype),
                "full_gradient_l2_norm": float(torch.linalg.vector_norm(baseline_gradient)),
                "movable_gradient_l2_norm": float(torch.linalg.vector_norm(movable_gradient)),
                "movable_gradient_linf_norm": float(torch.linalg.vector_norm(movable_gradient, ord=float("inf"))),
                "fixed_gradient_l2_norm": float(torch.linalg.vector_norm(baseline_gradient[~movable])),
                "value_and_gradient_wall_seconds": baseline_wall,
                "interpretation": "A nonzero unconstrained raw surrogate gradient does not disprove constrained stationarity of the placement method or its discrete move set.",
            },
            "finite_difference_definition": {
                "epsilons_microns_l2": list(EPSILONS),
                "direction": "Euclidean unit vector over movable x/y coordinates; fixed coordinates are zero",
                "first_derivative": "[f(x + epsilon v) - f(x - epsilon v)] / (2 epsilon)",
                "hvp": "[grad f(x + epsilon v) - grad f(x - epsilon v)] / (2 epsilon)",
                "scalar_second_derivative": "[f(x + epsilon v) - 2 f(x) + f(x - epsilon v)] / epsilon^2",
                "arithmetic_dtype": str(state.dtype),
                "warning": "Finite differences can cross clamp, rectangle-grid, or top-k selection boundaries. At those piecewise boundaries, autodiff follows the active branch at x while a symmetric finite difference samples branches on both sides; neither comparison establishes a classical Hessian.",
            },
            "random_directions": direction_results,
            "bounded_lanczos": lanczos,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch_default_dtype": str(torch.get_default_dtype()),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
            },
            "total_wall_seconds": time.perf_counter() - overall_started,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_latex(report, args.tex_output)
    print(json.dumps({
        "output": str(args.output),
        "tex_output": str(args.tex_output),
        "smooth_proxy_value": report["baseline"]["smooth_proxy_value"],
        "movable_gradient_l2_norm": report["baseline"]["movable_gradient_l2_norm"],
        "directional_curvatures": [
            item["autodiff_directional_curvature_vHv"]
            for item in report["random_directions"]
        ],
        "lanczos_status": report["bounded_lanczos"]["status"],
        "smallest_ritz_value": report["bounded_lanczos"].get("smallest_ritz_value"),
        "total_wall_seconds": report["total_wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
