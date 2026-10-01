#!/usr/bin/env python3
"""Render the saved IBM01 endpoint-curvature diagnostic without recomputation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "writeup/data/publication/curvature_diagnostics_ibm01.json"
DEFAULT_FOLLOWUP = ROOT / "writeup/data/publication/curvature_followup_ibm01.json"
DEFAULT_HISTORICAL = ROOT / "writeup/data/publication/historical_summary.json"
DEFAULT_OUTPUT = ROOT / "writeup/latex/generated/curvature_diagnostics.tex"


def sci(value: Any, digits: int = 2) -> str:
    """Math-mode scientific notation, without a siunitx dependency."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cannot render a non-finite number")
    if number == 0.0:
        return "0"
    mantissa, exponent = ("%.*e" % (digits, number)).split("e")
    return r"%s\times10^{%d}" % (mantissa, int(exponent))


def historical_ibm01_eigenvalue(historical: dict[str, Any]) -> float:
    records = [r for r in historical["cascade_stage_records"] if r["name"] == "ibm01"]
    if len(records) != 1 or not records[0]["matched_endpoint"]:
        raise ValueError("expected one matched IBM01 cascade record")
    return float(records[0]["iter_log"][0]["lam_min"])


def branch_counts(first: int, second: int) -> str:
    if first == second:
        return "%d recorded min/max/clamp comparisons of the composite in each direction" % first
    return "%d and %d recorded min/max/clamp comparisons of the composite" % (first, second)


def render_followup(report: dict[str, Any]) -> str:
    guarded = sorted(
        (run for run in report["runs"]
         if run["mask_name"] == "guarded_interior" and run["dtype"] == "torch.float64"),
        key=lambda run: run["seed"],
    )
    components = sorted(
        (run for run in report["weighted_component_runs"]
         if run["mask_name"] == "guarded_interior"),
        key=lambda run: (run["seed"], run["component"]),
    )
    if len(guarded) != 2 or len(components) != 6:
        raise ValueError("expected two guarded float64 directions and three components each")
    for run in guarded:
        if set(run["forward_float_operation_output_counts"]) != {"torch.float64"}:
            raise ValueError("follow-up does not establish float64 forward arithmetic")
        for scale in run["scales"]:
            for comparison in ("plus_minus_branch_delta", "base_plus_branch_delta", "base_minus_branch_delta"):
                if scale[comparison]["canvas_clamp:changed_comparisons"] != 0:
                    raise ValueError("guarded follow-up contains a canvas branch change")

    def at(run: dict[str, Any], epsilon: float) -> dict[str, Any]:
        return next(scale for scale in run["scales"] if scale["epsilon_microns"] == epsilon)

    wirelength = [run for run in components if run["component"] == "wirelength"]
    names = {
        "wirelength": "Wirelength",
        "weighted_density": r"$\tfrac12\times$density",
        "weighted_congestion": r"$\tfrac12\times$RUDY",
    }
    lines = [
        r"\paragraph{Precision and branch follow-up.} "
        "A separate saved probe restricts both directions to the %s coordinates whose "
        r"canvas clearance exceeds $0.1\,\mu$m in both precisions, so no perturbation "
        "can cross a canvas clamp. All forward floating operations run in float64; "
        "stored float32 constants are promoted without regeneration. With $P$ selecting "
        r"these coordinates, the comparison uses $PH_{\mathrm{AD}}Pv$ and differences of "
        r"$Pg_{\mathrm{AD}}$. The frozen composite is also split into its three weighted "
        "primitives, which reproduce the composite value, slope and curvature to $10^{-14}$. "
        r"Table~\ref{tab:ibm01-curvature-followup} reports the gradient jump "
        r"$\lVert P[g(x+\epsilon v)-g(x-\epsilon v)]\rVert_2$ in $\mu\mathrm{m}^{-1}$."
        % ("{:,}".format(guarded[0]["effective_coordinate_count"]).replace(",", "{,}")),
        "",
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{Guarded float64 follow-up at the saved IBM01 endpoint: gradient jump "
        r"across $\pm\epsilon$ (in $\mu\mathrm{m}^{-1}$) and the relative difference between "
        r"the finite-difference and AD Hessian-vector products at $\epsilon=10^{-5}\,\mu$m. "
        r"A smooth term has a jump proportional to $\epsilon$; a jump that does not shrink "
        r"with $\epsilon$ marks a branch boundary.}",
        r"\label{tab:ibm01-curvature-followup}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Dir. & Term & $\epsilon=10^{-7}$ & $\epsilon=10^{-5}$ & $\epsilon=10^{-3}$ & Rel.\ HVP diff. \\",
        r"\midrule",
    ]
    for index, composite in enumerate(guarded, start=1):
        seed = composite["seed"]
        rows = [(names[run["component"]], run) for run in components if run["seed"] == seed]
        rows.sort(key=lambda item: list(names.values()).index(item[0]))
        rows.append(("Composite $J_B$", composite))
        for label, run in rows:
            lines.append(
                "%d & %s & $%s$ & $%s$ & $%s$ & $%s$ \\\\"
                % (
                    index, label,
                    sci(at(run, 1e-7)["gradient_jump_l2_effective"]),
                    sci(at(run, 1e-5)["gradient_jump_l2_effective"]),
                    sci(at(run, 1e-3)["gradient_jump_l2_effective"]),
                    sci(at(run, 1e-5)["hvp_relative_error_effective"]),
                )
            )
        if index == 1:
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "Inside the guarded region the wirelength gradient changes in proportion to "
        r"$\epsilon$, and its finite-difference HVP agrees with the AD product to $%s$ and "
        r"$%s$ at $\epsilon=10^{-5}\,\mu$m. The density and RUDY gradients jump by amounts "
        r"that persist down to $\epsilon=10^{-7}\,\mu$m, where the $\pm\epsilon$ samples "
        "already differ in %s. "
        r"At $\epsilon=10^{-3}\,\mu$m the composite jump norms are $%s$ and $%s$, and the "
        "relative HVP differences are %.2f and %.2f."
        % (
            sci(at(wirelength[0], 1e-5)["hvp_relative_error_effective"]),
            sci(at(wirelength[1], 1e-5)["hvp_relative_error_effective"]),
            branch_counts(
                sum(at(guarded[0], 1e-7)["plus_minus_branch_delta"].values()),
                sum(at(guarded[1], 1e-7)["plus_minus_branch_delta"].values()),
            ),
            sci(at(guarded[0], 1e-3)["gradient_jump_l2_effective"]),
            sci(at(guarded[1], 1e-3)["gradient_jump_l2_effective"]),
            at(guarded[0], 1e-3)["hvp_relative_error_effective"],
            at(guarded[1], 1e-3)["hvp_relative_error_effective"],
        ),
    ])
    return "\n".join(lines)


def render(report: dict[str, Any], followup: dict[str, Any], historical: dict[str, Any]) -> str:
    scope = report["scope"]
    if scope["benchmark"] != "ibm01":
        raise ValueError("expected the saved IBM01 diagnostic")
    directions = report["random_directions"]
    if len(directions) != 2 or any(len(item["scales"]) != 5 for item in directions):
        raise ValueError("expected two directions and five scales per direction")

    baseline = report["baseline"]
    lanczos = report["bounded_lanczos"]
    historical_lambda = historical_ibm01_eigenvalue(historical)
    relative_errors = [
        scale["hvp_vector_relative_error_l2"]
        for direction in directions for scale in direction["scales"]
    ]
    cosines = [
        scale["hvp_vector_cosine"]
        for direction in directions for scale in direction["scales"]
    ]
    first = directions[0]["scales"][0]
    lines = [
        "% Generated by scripts/render_publication_curvature.py from saved JSON; do not edit.",
        "The probe evaluates the frozen legacy curvature surrogate $J_B$ at the saved "
        "IBM01 endpoint of the first curvature stage, in float32 and without any "
        "optimization or legalization. At that point "
        + r"$J_B=%.5f$ and $\lVert\nabla J_B\rVert_2=%s\,\mu\mathrm{m}^{-1}$. "
        % (baseline["smooth_proxy_value"], sci(baseline["movable_gradient_l2_norm"]))
        + "A Lanczos run capped at %d Hessian-vector products (HVPs) of the "
        "automatic-differentiation (AD) operator returned Rayleigh quotient $%.6f$ with "
        "residual norm $%s$. The historical cascade recorded the smallest eigenvalue "
        "$%.6f$ at the same endpoint; the two values differ by $%s$."
        % (lanczos["hvp_count"], lanczos["validated_rayleigh_quotient"],
           sci(lanczos["validated_residual_l2"]), historical_lambda,
           sci(abs(lanczos["validated_rayleigh_quotient"] - historical_lambda))),
        "",
        r"Let $x$ be the saved coordinate vector, $g_{\mathrm{AD}}(x)$ its AD gradient, and $H_{\mathrm{AD}}v$ the AD HVP for a unit vector $v$ over the movable coordinates. The central differences are",
        r"\[d_{\mathrm{FD}}(\epsilon)=\frac{J_B(x+\epsilon v)-J_B(x-\epsilon v)}{2\epsilon},\qquad"
        r" k_{\mathrm{FD}}(\epsilon)=v^\top\frac{g_{\mathrm{AD}}(x+\epsilon v)-g_{\mathrm{AD}}(x-\epsilon v)}{2\epsilon}.\]",
        r"Positions and $\epsilon$ are in micrometers, so slopes are in $\mu\mathrm{m}^{-1}$ and curvatures in $\mu\mathrm{m}^{-2}$; the surrogate itself is dimensionless.",
        "",
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{Float32 central differences along two seeded random unit directions at the "
        r"saved IBM01 endpoint, all movable coordinates included. The AD columns are the slope "
        r"and curvature of the branch active at $x$; the FD columns sample both sides of nearby "
        r"branch boundaries. Slopes are in $\mu\mathrm{m}^{-1}$ and curvatures in "
        r"$\mu\mathrm{m}^{-2}$. The directions are independent of the Lanczos vector.}",
        r"\label{tab:ibm01-curvature-diagnostic}",
        r"\begin{tabular}{@{}rrrrrr@{}}",
        r"\toprule",
        r"Dir. & $\epsilon$ ($\mu$m) & $g_{\mathrm{AD}}^\top v$ & $d_{\mathrm{FD}}$ & $v^\top H_{\mathrm{AD}}v$ & $k_{\mathrm{FD}}$ \\",
        r"\midrule",
    ]
    for direction_index, direction in enumerate(directions, start=1):
        for scale in direction["scales"]:
            lines.append(
                "%d & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\"
                % (
                    direction_index,
                    sci(scale["epsilon_microns_l2"]),
                    sci(direction["autodiff_directional_gradient"]),
                    sci(scale["cost_central_first_derivative"]),
                    sci(direction["autodiff_directional_curvature_vHv"]),
                    sci(scale["directional_curvature_from_gradient_fd"]),
                )
            )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"For the full vectors, with $h_{\mathrm{FD}}=[g_{\mathrm{AD}}(x+\epsilon v)-g_{\mathrm{AD}}(x-\epsilon v)]/(2\epsilon)$, the relative difference $\lVert h_{\mathrm{FD}}-H_{\mathrm{AD}}v\rVert_2/\lVert H_{\mathrm{AD}}v\rVert_2$ ranges from %.2f to %.0f over the ten rows, and the cosine between the two vectors from $%s$ to $%s$. At the endpoint, %d coordinates sit at or beyond a canvas clamp. For direction~1 at $\epsilon=10^{-3}\,\mu$m the forward and backward cost slopes are $%s$ and $%s\,\mu\mathrm{m}^{-1}$, so the two sides already lie on different branches."
        % (min(relative_errors), max(relative_errors), sci(min(cosines)), sci(max(cosines)),
           sum(report["state"]["baseline_clamp_counts"].values()),
           sci((first["cost_plus"] - first["cost_base"]) / first["epsilon_microns_l2"]),
           sci((first["cost_base"] - first["cost_minus"]) / first["epsilon_microns_l2"])),
        "",
        "% Follow-up source: writeup/data/publication/curvature_followup_ibm01.json.",
        render_followup(followup),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--followup-input", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--historical-input", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    followup = json.loads(args.followup_input.read_text(encoding="utf-8"))
    historical = json.loads(args.historical_input.read_text(encoding="utf-8"))
    output = render(report, followup, historical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print("Rendered %s from %s, %s and %s."
          % (args.output, args.input, args.followup_input, args.historical_input))


if __name__ == "__main__":
    main()
