#!/usr/bin/env python3
"""Freeze historical evidence and regenerate the report's tables and figures.

These are audited historical records, not matched new experiment results.
Run from any directory with the repository's Python environment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "writeup/data/publication"
GENERATED = ROOT / "writeup/latex/generated"
FIGURES = ROOT / "writeup/latex/figures"
SNAPSHOT = "2ab92a33d352f0dbc4ba97d7c2bfad41ccf0ecfd"
UPSTREAM = "996193c83eaad151e5ae3fb166cedbd83372d060"
IBM = [f"ibm{i:02d}" for i in range(1, 19) if i != 5]
RECORDS = [
    ("v1", "v1 stacked-periphery", "CDLNSSACascadeStackedPeripheryPlacer_20260517_062436.json"),
    ("v3", "Earlier V3-based v2", "EPYC_thinkorplace_v2_validation_20260520.json"),
    ("gaussian", "E117 Gaussian ancestor", "E111MinimalGaussianPlacer_20260520_135149.json"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, destination: str) -> dict:
    target = DATA / "raw" / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    # A publication checkout may have only the frozen evidence, not local runs.
    payload = path.read_bytes() if path.exists() else target.read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite changed frozen evidence: {target}")
    target.write_bytes(payload)
    return {"source": str(path.relative_to(ROOT)), "frozen": str(target.relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest()}


def latex_table(headers, rows, alignment):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    for directory in [DATA / "raw", GENERATED, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    groups, sources, per_benchmark = {}, [], []
    for key, label, filename in RECORDS:
        path = ROOT / "results" / filename
        source = freeze(path, filename)
        sources.append(source)
        record = json.loads((ROOT / source["frozen"]).read_text())
        rows = record["benchmarks"]
        by_name = {row["name"]: row for row in rows}
        assert len(rows) == len(by_name) == 17 and set(by_name) == set(IBM)
        for row in rows:
            assert all(math.isfinite(float(row[k])) for k in
                       ["proxy_cost", "wirelength", "density", "congestion", "runtime_seconds"])
            reconstructed = row["wirelength"] + .5 * row["density"] + .5 * row["congestion"]
            assert abs(reconstructed - row["proxy_cost"]) < 1e-6
            per_benchmark.append({"variant": key, **{k: row[k] for k in
                ["name", "proxy_cost", "wirelength", "density", "congestion", "overlap_count", "runtime_seconds"]}})
        proxy = np.array([by_name[n]["proxy_cost"] for n in IBM])
        walls = np.array([by_name[n]["runtime_seconds"] for n in IBM])
        components = np.array([[by_name[n]["wirelength"], .5 * by_name[n]["density"],
                                .5 * by_name[n]["congestion"]] for n in IBM])
        groups[key] = {"label": label, "source_file": filename, "coverage": len(rows),
                       "mean_proxy": float(proxy.mean()), "min_proxy": float(proxy.min()),
                       "max_proxy": float(proxy.max()), "total_overlaps": sum(r["overlap_count"] for r in rows),
                       "mean_seconds": float(walls.mean()), "max_seconds": float(walls.max()),
                       "component_means": components.mean(axis=0).tolist(),
                       "weighted_component_shares": (components.mean(axis=0) / proxy.mean()).tolist(),
                       "timestamp": record.get("timestamp"), "recorded_jobs": record.get("jobs"),
                       "recorded_torch_threads": record.get("torch_threads"), "by_name": by_name}
    with (DATA / "per_benchmark.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(per_benchmark[0]))
        writer.writeheader(); writer.writerows(per_benchmark)

    cascades = []
    for name in IBM:
        old = ROOT / f"experiments/E74_hessian_saddle/results/hessian_{name}.json"
        new = ROOT / f"experiments/E84_cascading_saddle/results/cascade_{name}.json"
        old_source = freeze(old, f"cascade/hessian_{name}.json")
        new_source = freeze(new, f"cascade/cascade_{name}.json")
        sources.extend([old_source, new_source])
        before = json.loads((ROOT / old_source["frozen"]).read_text())
        after = json.loads((ROOT / new_source["frozen"]).read_text())
        for stage, prefix in [("E74_hessian_saddle", "hessian"), ("E84_cascading_saddle", "cascade")]:
            sources.append(freeze(ROOT / f"experiments/{stage}/results/{prefix}_{name}.pt",
                                  f"placements/{prefix}_{name}.pt"))
        matched = abs(before["best_proxy"] - after["init_proxy"]) < 1e-7
        cascades.append({"name": name, "matched_endpoint": matched, "before": after["init_proxy"],
                         "after": after["best_proxy"], "gain": after["init_proxy"] - after["best_proxy"],
                         "gain_percent": 100 * (after["init_proxy"]-after["best_proxy"])/after["init_proxy"],
                         "overlaps": after["final_overlap"], "iterations": after["iters_run"],
                         "stage_seconds": after["wall_seconds"], "iter_log": after["iter_log"],
                         "eigenvalues_e74": before.get("eigenvalues", [])})
    matched_cascades = [r for r in cascades if r["matched_endpoint"]]
    a, b = groups["v1"], groups["gaussian"]
    paired = np.array([a["by_name"][n]["proxy_cost"] - b["by_name"][n]["proxy_cost"] for n in IBM])
    summary = {"competition_source_revision": SNAPSHOT, "upstream_revision": UPSTREAM,
               "classification": "historical_unmatched_runs_not_current_ensemble_results",
               "sources": sources, "variants": groups, "cascade_stage_records": cascades,
               "descriptive_v1_vs_gaussian": {"mean_score_reduction_percent": 100*(a["mean_proxy"]-b["mean_proxy"])/a["mean_proxy"],
                 "designs_lower": int((paired > 0).sum()), "designs_higher": int((paired < 0).sum()),
                 "largest_absolute_reduction_design": IBM[int(paired.argmax())],
                 "largest_absolute_reduction": float(paired.max()),
                 "causal_interpretation": False},
               "cascade_summary": {"records": len(cascades), "matched_e74_endpoints": len(matched_cascades),
                 "mean_before": float(np.mean([r["before"] for r in cascades])),
                 "mean_after": float(np.mean([r["after"] for r in cascades])),
                 "designs_improved": sum(r["gain"] > 1e-7 for r in cascades),
                 "total_overlaps": sum(r["overlaps"] for r in cascades)}}
    (DATA / "historical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (GENERATED / "historical_table.tex").write_text(latex_table(
        ["Historical variant", "$n$", "Mean proxy", r"\shortstack{Recorded\\overlaps}",
         r"\shortstack{Mean\\time (s)}", r"\shortstack{Max.\\time (s)}"],
        [[g["label"], g["coverage"], f'{g["mean_proxy"]:.5f}', g["total_overlaps"],
          f'{g["mean_seconds"]:.1f}', f'{g["max_seconds"]:.1f}'] for g in groups.values()], "lrrrrr"))
    (GENERATED / "historical_per_benchmark.tex").write_text(latex_table(
        ["Design", "v1", "Earlier V3 v2", "Gaussian ancestor"],
        [[n] + [f'{g["by_name"][n]["proxy_cost"]:.5f}' for g in groups.values()] for n in IBM], "lrrr"))
    (GENERATED / "component_table.tex").write_text(latex_table(
        ["Historical variant", r"$C_{\mathrm{wl}}$", r"$\tfrac12 C_{\mathrm{den}}$", r"$\tfrac12 C_{\mathrm{cong}}$"],
        [[g["label"]] + [f'{v:.5f}' for v in g["component_means"]] for g in groups.values()], "lrrr"))
    (GENERATED / "cascade_table.tex").write_text(latex_table(
        ["Design", "Start", "End", "Reduction (\\%)", "Stage time (s)"],
        [[r["name"], f'{r["before"]:.5f}', f'{r["after"]:.5f}', f'{r["gain_percent"]:.2f}',
          f'{r["stage_seconds"]:.0f}'] for r in cascades], "lrrrr"))

    colors = ["#246187", "#c57726", "#357b56"]
    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
    xx = np.arange(17)
    for (key, g), color, marker in zip(groups.items(), colors, ["o", "s", "^"]):
        ax.plot(xx, [g["by_name"][n]["proxy_cost"] for n in IBM], marker=marker,
                markersize=3.8, linewidth=.9, color=color, label=g["label"])
    ax.set_xticks(xx, [n.replace("ibm", "") for n in IBM]); ax.set_xlabel("IBM design")
    ax.set_ylabel("Canonical proxy cost (lower is better)")
    ax.legend(frameon=False, fontsize=8, ncol=1); ax.grid(axis="y", alpha=.18)
    fig.savefig(FIGURES / "historical_scores.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 2.65), layout="constrained")
    bottom = np.zeros(3)
    for i, (label, color) in enumerate(zip(["Wirelength", "0.5 × density", "0.5 × congestion"], colors)):
        vals = [g["component_means"][i] for g in groups.values()]
        ax.barh(range(3), vals, left=bottom, color=color, label=label, height=.55)
        bottom += vals
    ax.set_yticks(range(3), ["v1", "Earlier V3 v2", "Gaussian ancestor"])
    ax.set_xlabel("Mean contribution to canonical proxy cost")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5,1.22), ncol=3, fontsize=8)
    ax.invert_yaxis(); fig.savefig(FIGURES / "historical_components.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3), layout="constrained")
    ax.bar(range(len(cascades)), [r["gain_percent"] for r in cascades], color=colors[0])
    ax.set_xticks(range(len(cascades)), [r["name"].replace("ibm", "") for r in cascades])
    ax.set_xlabel("IBM design"); ax.set_ylabel("Recorded cascade-stage reduction (%)")
    ax.grid(axis="y", alpha=.18); fig.savefig(FIGURES / "cascade_gains.pdf"); plt.close(fig)

    geom_path = ROOT / "benchmarks/processed/public/ibm01.pt"
    tensor_paths = [DATA / "raw/placements/hessian_ibm01.pt",
                    DATA / "raw/placements/cascade_ibm01.pt"]
    geom = torch.load(geom_path, map_location="cpu", weights_only=False)
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in tensor_paths]
    fig, axes = plt.subplots(1,2,figsize=(6.5,3.4),layout="constrained",sharey=True)
    nh = int(geom["num_hard_macros"])
    for ax, state, title, color in zip(axes, states, ["Before cascade", "After cascade"], colors):
        pos = state["placement"].numpy(); sizes = geom["macro_sizes"].numpy()
        patches = [Rectangle((pos[i,0]-sizes[i,0]/2,pos[i,1]-sizes[i,1]/2),sizes[i,0],sizes[i,1]) for i in range(nh)]
        ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=color, linewidth=.25, alpha=.45))
        ax.set_xlim(0,geom["canvas_width"]); ax.set_ylim(0,geom["canvas_height"]); ax.set_aspect("equal")
        ax.set_xlabel("x (µm)"); ax.set_title(f'{title}\nProxy {state["stats"]["best_proxy"]:.5f}',fontsize=9)
    axes[0].set_ylabel("y (µm)")
    fig.savefig(FIGURES / "cascade_placement_ibm01.pdf"); plt.close(fig)
    (DATA / "placement_figure_sources.json").write_text(json.dumps(
        [{"path":str(p.relative_to(ROOT)),"sha256":sha256(p)} for p in [geom_path]+tensor_paths],indent=2)+"\n")

    prefixes = ["macro_place", "submissions/thinkorplace", "submissions/thinkorplace-v2", "submissions/common",
                "experiments/E18_dpo_init", "experiments/E39_kmacro_joint_lns", "experiments/E41_dpo_kjoint",
                "experiments/E74_hessian_saddle", "experiments/E84_cascading_saddle",
                "experiments/E97_levy_saddle", "experiments/E100_weight_portfolio_saddle",
                "writeup/archive/submissions/dpo/ablation_v2_steps.py"]
    snapshot_path = DATA / "source_snapshot.json"
    archive_path = DATA / "competition-source.tar"
    if snapshot_path.exists() and archive_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["commit"] == SNAPSHOT
        assert sha256(archive_path) == snapshot["archive_sha256"], "Frozen source archive changed"
        code_paths = snapshot["source_files"]
    else:
        listing = subprocess.check_output(["git","ls-tree","-r","--name-only",SNAPSHOT,"--",*prefixes],cwd=ROOT,text=True).splitlines()
        code_paths = [p for p in listing if p.endswith(".py")]
        archive = subprocess.check_output(["git","archive",SNAPSHOT,"--",*code_paths],cwd=ROOT)
        archive_path.write_bytes(archive)
        snapshot_path.write_text(json.dumps({"commit":SNAPSHOT,"archive":"competition-source.tar",
            "archive_sha256":hashlib.sha256(archive).hexdigest(),"source_files":code_paths,
            "scope":"Python source closure and competition infrastructure; benchmarks/TILOS submodule must be supplied separately"},indent=2)+"\n")
    print(json.dumps({"variants": {k:{q:v[q] for q in ["mean_proxy","weighted_component_shares"]} for k,v in groups.items()},
                      "cascade":summary["cascade_summary"], "historical_comparison":summary["descriptive_v1_vs_gaussian"],
                      "source_files":len(code_paths)},indent=2))


if __name__ == "__main__":
    main()
