#!/usr/bin/env python3
"""Render the report v1.0 figures and generated tables from frozen evidence.

Inputs are the frozen publication records under writeup/data/publication/ and
the official leaderboard in the pinned upstream README (revision 996193c). The
README is first frozen into leaderboard_996193c.json; when the upstream commit
is not in the local Git object store, the frozen copy is used instead.

The IBM01 congestion underlay is recomputed with the canonical evaluator from
the frozen after-cascade tensor. Before doing so the script checks the
evaluator sources and IBM01 inputs against the SHA-256 values recorded in
e74_e84_canonical_rescore_20260907.json and requires the recomputed proxy to
match that record. Nothing in the runtime closure is written.

Run from the repository root:

    .venv/bin/python scripts/render_publication_v1.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True  # Keep the evaluator import from writing into the runtime closure.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "writeup/data/publication"
RAW = DATA / "raw"
FIGURES = ROOT / "writeup/latex/figures"
GENERATED = ROOT / "writeup/latex/generated"
INTERACTIVE = ROOT / "writeup/interactive"
LEADERBOARD = DATA / "leaderboard_996193c.json"
SUMMARY = DATA / "v1_figures_summary.json"
RESCORE = DATA / "e74_e84_canonical_rescore_20260907.json"
REFEREE = DATA / "referee_context_20260907.json"
UPSTREAM = "996193c83eaad151e5ae3fb166cedbd83372d060"
UPSTREAM_REPOSITORY = "partcleda/macro-place-challenge-2026"
IBM = [f"ibm{i:02d}" for i in range(1, 19) if i != 5]
# Display names only; the raw files keep their historical identities.
RECORDS = [
    ("v1", "v1 (17 May)", "CDLNSSACascadeStackedPeripheryPlacer_20260517_062436.json"),
    ("early_v2", "early v2", "EPYC_thinkorplace_v2_validation_20260520.json"),
    ("single", "single pipeline (20 May lineage)", "E111MinimalGaussianPlacer_20260520_135149.json"),
]

INK, INK2, AXIS, GRID = "#1f1f1f", "#555555", "#6b6b6b", "#ececec"
CONTEXT, BASELINE = "#9a9a9a", "#7a7a7a"
# Record colors reuse the main.tex tokens: blue as in "surrogate proposes", orange as in "exact score accepts".
COLOR = {"v1": "#1769AA", "early_v2": "#0E8796", "single": "#C26816"}
ACCENT = COLOR["single"]
MARKER = {"v1": "o", "early_v2": "s", "single": "D"}
COMPONENT_COLORS = ["#c4c4c4", "#8c8c8c", "#5e4fa2"]
CONGESTION_CMAP = LinearSegmentedColormap.from_list(
    "congestion", ["#f8f7fb", "#dcd6ee", "#ada1d6", "#7f6dbd", "#5a4799"])
DISPLACEMENT_CMAP = LinearSegmentedColormap.from_list(
    "displacement", ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0b2547"])
PDF_METADATA = {"CreationDate": None, "ModDate": None}
ARROW_SCALE = 4
DOT_BELOW_UM = 0.05  # Displacements shorter than this are drawn as dots; at 4x they would be bare arrowheads.
LEADERBOARD_DATE = "14 August 2026"  # Date of the upstream revision that carries the final leaderboard.


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source(path: Path) -> dict:
    return {"path": rel(path), "sha256": sha256(path)}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Official leaderboard
# ---------------------------------------------------------------------------

def number_or_none(text: str):
    value = text.replace("*", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def parse_leaderboard(readme: bytes) -> dict:
    lines = readme.decode("utf-8").splitlines()
    field_line = next(i for i, line in enumerate(lines, 1) if "verified eligible submissions" in line)
    field = re.search(r"\*\*(\d+) registrations\*\*, with \*\*(\d+) verified eligible submissions\*\*",
                      lines[field_line - 1])
    header = next(i for i, line in enumerate(lines, 1) if line.startswith("| Rank | Team |"))
    columns = [c.strip() for c in lines[header - 1].strip("|").split("|")]
    rows = []
    for number in range(header + 2, len(lines) + 1):
        line = lines[number - 1]
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(columns), (number, cells)
        raw = dict(zip(columns, cells))
        rank = int(raw["Rank"]) if raw["Rank"].isdigit() else raw["Rank"]
        rows.append({
            "line": number, "rank": rank, "team": raw["Team"].strip('"'),
            "mean_proxy_text": raw["Avg Proxy Cost"].replace("*", ""),
            "mean_proxy": number_or_none(raw["Avg Proxy Cost"]),
            "best": number_or_none(raw["Best"]), "worst": number_or_none(raw["Worst"]),
            "overlaps": int(raw["Overlaps"]) if raw["Overlaps"].isdigit() else None,
            "listed_runtime": None if raw["Runtime"] in ("-", "—", "") else raw["Runtime"],
            "verified": raw["Verified"] == ":white_check_mark:", "notes": raw["Notes"],
        })
    baselines = {}
    for key, team in [("replace", "RePlAce (baseline)"), ("sa", "SA (baseline)")]:
        baselines[key] = next(row for row in rows if row["team"] == team)
    design_rows = []
    pattern = re.compile(r"^\| \*\*(ibm\d\d)\*\* \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| "
                         r"([0-9.]+) \| \*\*([0-9.]+)\*\* ⭐ \|$")
    for number, line in enumerate(lines, 1):
        match = pattern.match(line.strip())
        if match:
            name, macros, nets, canvas, util, sa, replace = match.groups()
            design_rows.append({"line": number, "name": name, "sa": float(sa), "replace": float(replace),
                                "readme_macros": macros.strip(), "readme_nets": nets.strip(),
                                "readme_canvas_um": canvas.strip(), "readme_area_utilization": util.strip()})
    ranked = [row for row in rows if isinstance(row["rank"], int)]
    blob = hashlib.sha1(b"blob %d\0" % len(readme) + readme).hexdigest()
    return {
        "kind": "frozen_parse_of_official_final_leaderboard",
        "evidence_label": "Official",
        "source": {
            "repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM, "path": "README.md",
            "retrieval": "git show 996193c83eaa:README.md",
            "git_blob_sha1": blob, "sha256": sha256_bytes(readme), "line_count": len(lines),
        },
        "field_size": {"line": field_line, "registrations": int(field.group(1)),
                       "verified_eligible_submissions": int(field.group(2))},
        "leaderboard_header_line": header,
        "ranked_row_count": len(ranked),
        "rows": rows,
        "baselines": {key: {"line": row["line"], "mean_proxy": row["mean_proxy"], "best": row["best"],
                            "worst": row["worst"]} for key, row in baselines.items()},
        "baseline_per_design": design_rows,
        "notes": ("Rows are parsed verbatim from the leaderboard table. Rank is an integer for ranked "
                  "rows; baselines and unranked rows keep the README marker ('—', 'DQ'). The README "
                  "benchmark-table columns other than the SA and RePlAce baselines are kept as text and "
                  "are not used; the report's benchmark statistics come from the parsed inputs."),
    }


def freeze_leaderboard() -> dict:
    try:
        readme = subprocess.check_output(["git", "show", f"{UPSTREAM}:README.md"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        readme = None
    if readme is None:
        if not LEADERBOARD.exists():
            raise RuntimeError("Upstream revision unavailable and no frozen leaderboard exists")
        return load_json(LEADERBOARD)
    record = parse_leaderboard(readme)
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if LEADERBOARD.exists() and LEADERBOARD.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite changed frozen evidence: {rel(LEADERBOARD)}")
    write_text(LEADERBOARD, text)
    return record


def check_leaderboard(lb: dict, referee: dict) -> dict:
    official = referee["official_source"]
    assert lb["source"]["revision"] == official["revision"] == UPSTREAM
    assert lb["source"]["sha256"] == official["readme_sha256"]
    assert lb["field_size"]["verified_eligible_submissions"] == official["reported_verified_eligible_submissions"] == 132
    assert lb["ranked_row_count"] == 129
    by_rank = {row["rank"]: row for row in lb["rows"] if isinstance(row["rank"], int)}
    ours = by_rank[9]
    assert ours["team"] == "thinkorplace" and ours["mean_proxy"] == 0.9974
    assert (ours["best"], ours["worst"], ours["overlaps"]) == (0.7829, 1.2602, 0) and ours["verified"]
    assert ours["line"] == 286 and lb["field_size"]["line"] == 15
    for entry in official["leaderboard"]:
        assert by_rank[entry["rank"]]["team"] == entry["team"]
        assert by_rank[entry["rank"]]["mean_proxy"] == entry["mean_proxy"]
    assert lb["baselines"]["sa"]["mean_proxy"] == 2.1251 and lb["baselines"]["replace"]["mean_proxy"] == 1.4578
    per_design = {row["name"]: row for row in lb["baseline_per_design"]}
    assert list(per_design) == IBM
    for name, values in official["baselines_per_design"].items():
        assert per_design[name]["sa"] == values["sa"] and per_design[name]["replace"] == values["replace"]
    for key in ("sa", "replace"):
        mean = float(np.mean([per_design[n][key] for n in IBM]))
        assert round(mean, 4) == lb["baselines"][key]["mean_proxy"]
    return by_rank


# ---------------------------------------------------------------------------
# Frozen records
# ---------------------------------------------------------------------------

def load_records(baselines: dict) -> dict:
    records = {}
    for key, label, filename in RECORDS:
        path = RAW / filename
        payload = load_json(path)
        by_name = {row["name"]: row for row in payload["benchmarks"]}
        assert sorted(by_name) == IBM
        for name, row in by_name.items():
            parts = row["wirelength"] + .5 * row["density"] + .5 * row["congestion"]
            assert abs(parts - row["proxy_cost"]) < 1e-6 and row["overlap_count"] == 0
            assert row["sa_baseline"] == baselines[name]["sa"]
            assert row["replace_baseline"] == baselines[name]["replace"]
        proxy = np.array([by_name[n]["proxy_cost"] for n in IBM])
        seconds = np.array([by_name[n]["runtime_seconds"] for n in IBM])
        records[key] = {"label": label, "source": source(path), "by_name": by_name, "proxy": proxy,
                        "seconds": seconds, "mean_proxy": float(proxy.mean()),
                        "mean_seconds": float(seconds.mean())}
    return records


def load_curvature(rescore: dict) -> dict:
    by_bench = {row["name"]: row for row in rescore["benchmarks"]}
    designs, attempts, sources = [], [], []
    for name in IBM:
        first_path = RAW / f"cascade/hessian_{name}.json"
        cascade_path = RAW / f"cascade/cascade_{name}.json"
        first, cascade = load_json(first_path), load_json(cascade_path)
        sources += [source(first_path), source(cascade_path)]
        assert abs(first["final_proxy"] - cascade["init_proxy"]) < 1e-7
        stages = {p["stage"]: p for p in by_bench[name]["placements"]}
        designs.append({
            "name": name, "plateau": first["start_proxy"], "first_stage": first["final_proxy"],
            "cascade": cascade["best_proxy"], "first_stage_seconds": first["wall_seconds"],
            "cascade_seconds": cascade["wall_seconds"],
            "first_stage_strict_valid": stages["E74"]["geometry_validation"]["full_legal"],
            "cascade_strict_valid": stages["E84"]["geometry_validation"]["full_legal"],
            "first_stage_overlaps": first["final_overlap"], "cascade_overlaps": cascade["final_overlap"],
        })
        for attempt in first["attempts"]:
            success = bool(attempt["feasible"] and attempt.get("polished_overlap") == 0
                           and attempt.get("polished_proxy") is not None
                           and attempt["polished_proxy"] < first["start_proxy"])
            attempts.append({"name": name, "eigenvector": attempt["eig"] + 1, "epsilon": attempt["eps"],
                             "feasible": bool(attempt["feasible"]), "success": success})
    groups = {"eigenvector": {}, "epsilon": {}}
    for axis, values in [("eigenvector", [1, 2]), ("epsilon", [0.3, 1.0, 3.0])]:
        for value in values:
            chosen = [a for a in attempts if a[axis] == value]
            groups[axis][str(value)] = {"successes": sum(a["success"] for a in chosen), "attempts": len(chosen)}
    totals = {"attempts": len(attempts), "feasible": sum(a["feasible"] for a in attempts),
              "successes": sum(a["success"] for a in attempts)}
    # Verified counts quoted in the report; fail loudly if the frozen data disagree.
    assert totals == {"attempts": 204, "feasible": 201, "successes": 190}
    assert groups["eigenvector"] == {"1": {"successes": 94, "attempts": 102},
                                     "2": {"successes": 96, "attempts": 102}}
    assert [groups["epsilon"][k]["successes"] for k in ("0.3", "1.0", "3.0")] == [64, 65, 61]
    means = {stage: float(np.mean([d[stage] for d in designs])) for stage in ("plateau", "first_stage", "cascade")}
    assert round(means["plateau"], 5) == 1.08128 and round(means["first_stage"], 5) == 1.06760
    assert round(means["cascade"], 5) == 1.06119
    assert all(d["first_stage"] < d["plateau"] and d["cascade"] < d["first_stage"] for d in designs)
    return {"designs": designs, "groups": groups, "totals": totals, "means": means, "sources": sources}


def ibm01_state() -> dict:
    """Recompute per-cell congestion for the frozen after-cascade IBM01 tensor."""
    rescore = load_json(RESCORE)
    for relative, record in rescore["canonical_evaluator"]["source_files"].items():
        if sha256(ROOT / relative) != record["sha256"]:
            raise RuntimeError(f"Evaluator source differs from the rescore record: {relative}")
    bench = next(row for row in rescore["benchmarks"] if row["name"] == "ibm01")
    for relative, record in bench["input_files"].items():
        if sha256(ROOT / relative) != record["sha256"]:
            raise RuntimeError(f"IBM01 input differs from the rescore record: {relative}")
    sys.path.insert(0, str(ROOT))
    with contextlib.redirect_stdout(io.StringIO()):
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import _set_placement
        benchmark, plc = load_benchmark_from_dir(str(ROOT / "external/MacroPlacement/Testcases/ICCAD04/ibm01"))
    paths = {stage: RAW / f"placements/{prefix}_ibm01.pt" for stage, prefix in
             [("before", "hessian"), ("after", "cascade")]}
    tensors = {stage: torch.load(path, map_location="cpu", weights_only=False) for stage, path in paths.items()}
    after = tensors["after"]["placement"].detach().cpu().contiguous()
    with contextlib.redirect_stdout(io.StringIO()):
        _set_placement(plc, after, benchmark)
        wirelength = float(plc.get_cost())
        density = float(plc.get_density_cost())
        congestion = float(plc.get_congestion_cost())
        horizontal = np.array(plc.get_horizontal_routing_congestion(), dtype=float)
        vertical = np.array(plc.get_vertical_routing_congestion(), dtype=float)
    proxy = wirelength + .5 * density + .5 * congestion
    recorded = next(p for p in bench["placements"] if p["stage"] == "E84")
    assert recorded["geometry_validation"]["full_legal"]
    if abs(proxy - recorded["fresh_canonical"]["proxy_cost"]) > 1e-9:
        raise RuntimeError("Recomputed IBM01 proxy does not match the canonical rescore record")
    rows, cols = int(plc.grid_row), int(plc.grid_col)
    pooled = np.sort(np.concatenate([vertical, horizontal]))[::-1]
    tail = pooled[:math.floor(len(pooled) * 0.05)].mean()
    assert abs(tail - congestion) < 1e-12
    nh = int(benchmark.num_hard_macros)
    return {
        "canvas": (float(benchmark.canvas_width), float(benchmark.canvas_height)),
        "grid": (rows, cols),
        "congestion": np.maximum(horizontal, vertical).reshape(rows, cols),
        "sizes": benchmark.macro_sizes.double().numpy(),
        "fixed": benchmark.macro_fixed.numpy().astype(bool),
        "num_hard": nh,
        "before": tensors["before"]["placement"].double().numpy(),
        "after": after.double().numpy(),
        "components": {"wirelength": wirelength, "density": density, "congestion": congestion, "proxy": proxy},
        "recorded_proxy": recorded["fresh_canonical"]["proxy_cost"],
        "first_stage_proxy": next(p for p in bench["placements"] if p["stage"] == "E74")["fresh_canonical"]["proxy_cost"],
        "sources": [source(paths["before"]), source(paths["after"]), source(RESCORE)]
                   + [{"path": k, "sha256": v["sha256"]} for k, v in bench["input_files"].items()]
                   + [{"path": k, "sha256": v["sha256"]} for k, v in
                      rescore["canonical_evaluator"]["source_files"].items()],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def latin_modern() -> str:
    """Register Latin Modern Roman from the local TeX Live so figure text matches the LaTeX body face."""
    try:
        regular = subprocess.check_output(["kpsewhich", "lmroman10-regular.otf"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        regular = ""
    if not regular:
        raise RuntimeError("lmroman10-regular.otf not found by kpsewhich; TeX Live's lm fonts are required")
    for variant in ("regular", "bold", "italic", "bolditalic"):
        font_manager.fontManager.addfont(str(Path(regular).with_name(f"lmroman10-{variant}.otf")))
    return font_manager.FontProperties(fname=regular).get_name()


def style() -> None:
    # Latin Modern lacks some symbols (≥, ε, ✓); fail rather than print an empty box.
    warnings.filterwarnings("error", message="Glyph .* missing from font")
    plt.rcParams.update({
        "font.family": latin_modern(), "mathtext.fontset": "cm",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
        "axes.linewidth": .6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.minor.width": .5, "ytick.minor.width": .5, "legend.frameon": False,
        # Type 3: matplotlib would embed the CFF-based Latin Modern OTFs as Type 42 under a TrueType
        # descriptor (FontFile2), which is invalid PDF; Type 3 draws the outlines and is valid.
        "pdf.fonttype": 3, "ps.fonttype": 3, "svg.fonttype": "path", "svg.hashsalt": "thinkorplace-v1",
        "figure.dpi": 100, "savefig.dpi": 300, "path.simplify": False,
    })


def save_pdf(fig, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)
    return path


def result_context_figure(lb: dict, by_rank: dict):
    top = [by_rank[r] for r in range(1, 11)]
    fig = plt.figure(figsize=(6.7, 2.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.1], wspace=.07,
                            left=.075, right=.985, top=.97, bottom=.27)
    left = fig.add_subplot(grid[0])
    right = fig.add_subplot(grid[1], sharey=left)
    for ax in (left, right):
        for rank in range(1, 11):
            ax.axhline(rank, color=GRID, linewidth=.6, zorder=0)
    others = [row for row in top if row["rank"] != 9]
    left.scatter([r["mean_proxy"] for r in others], [r["rank"] for r in others], s=18, color=CONTEXT,
                 linewidth=0, zorder=3)
    ours = by_rank[9]
    left.scatter([ours["mean_proxy"]], [9], s=46, color=ACCENT, edgecolor="white", linewidth=.8, zorder=4)
    for rank in (1, 2):
        row = by_rank[rank]
        left.text(row["mean_proxy"] + .0018, rank, f'{row["team"]}  {row["mean_proxy"]:.4f}',
                  va="center", ha="left", fontsize=8.5)
    left.text(ours["mean_proxy"] - .0022, 9, f'Thinkorplace  {ours["mean_proxy"]:.4f}', va="center",
              ha="right", fontsize=8.5, fontweight="bold")
    left.set_xlim(.93, 1.02)
    left.set_xticks([.94, .96, .98, 1.00])
    left.set_ylim(10.6, .4)
    left.set_yticks(range(1, 11))
    left.set_ylabel("Final rank")
    for key, label, offset, align, row in [("replace", "RePlAce\nbaseline", .03, "left", 2.0),
                                           ("sa", "SA\nbaseline", -.03, "right", 6.0)]:
        value = lb["baselines"][key]["mean_proxy"]
        right.axvline(value, color=BASELINE, linewidth=1.1, zorder=2)
        right.text(value + offset, row, f"{label}\n{value:.4f}", ha=align, va="center", fontsize=8,
                   color=INK2)
    right.set_xlim(1.38, 2.22)
    right.set_xticks([1.5, 2.0])
    right.tick_params(axis="y", left=False, labelleft=False)
    right.spines["left"].set_visible(False)
    marker = [(-1, -1.8), (1, 1.8)]
    for ax, x in ((left, 1), (right, 0)):
        ax.plot([x], [0], transform=ax.transAxes, marker=marker, markersize=6, linestyle="none",
                color=AXIS, mec=AXIS, mew=.7, clip_on=False)
    fig.text(.53, .105, "Mean proxy cost over the 17 IBM designs (lower is better)", ha="center", fontsize=8.5)
    fig.text(.075, .015, f"Official final leaderboard, {LEADERBOARD_DATE}", ha="left", fontsize=8,
             color=INK2)
    return fig


def canvas_axes(state: dict, size: tuple, left: float, bottom: float, side: float):
    """One canvas panel and a colorbar axis at fixed print positions (inches)."""
    width, height = state["canvas"]
    fig_w, fig_h = size
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([left / fig_w, bottom / fig_h, side / fig_w, side * height / width / fig_h])
    cax = fig.add_axes([(left + side + .14) / fig_w, (bottom + .2) / fig_h, .11 / fig_w, (side - .4) / fig_h])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(0, int(width) + 1, 5))
    ax.set_yticks(range(0, int(height) + 1, 5))
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    return fig, ax, cax


def finish_colorbar(fig, mappable, cax, label: str):
    bar = fig.colorbar(mappable, cax=cax)
    bar.set_label(label)
    bar.outline.set_linewidth(.5)
    bar.ax.tick_params(labelsize=8, width=.5)
    return bar


def placement_figure(state: dict):
    width, height = state["canvas"]
    rows, cols = state["grid"]
    nh = state["num_hard"]
    pos, sizes, fixed = state["after"], state["sizes"], state["fixed"]
    fig, ax, cax = canvas_axes(state, (4.35, 3.55), left=.55, bottom=.45, side=2.8)
    vmax = math.ceil(state["congestion"].max() * 5) / 5
    mesh = ax.pcolormesh(np.linspace(0, width, cols + 1), np.linspace(0, height, rows + 1),
                         state["congestion"], cmap=CONGESTION_CMAP, vmin=0, vmax=vmax,
                         shading="flat", edgecolors="face", linewidth=.25, rasterized=False, zorder=1)
    # Light fills keep the congestion map readable beneath the macros; fixed macros get the darker tone.
    tones = {False: ("white", .55, "#2b2b2b"), True: ("#3d3d3d", .6, "#1f1f1f")}
    for is_fixed in (False, True):
        index = [i for i in range(nh) if bool(fixed[i]) == is_fixed]
        if not index:
            continue
        face, alpha, edge = tones[is_fixed]
        patches = [Rectangle((pos[i, 0] - sizes[i, 0] / 2, pos[i, 1] - sizes[i, 1] / 2),
                             sizes[i, 0], sizes[i, 1]) for i in index]
        ax.add_collection(PatchCollection(patches, facecolor=face, alpha=alpha, edgecolor="none", zorder=3))
        ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=edge, linewidth=.4, zorder=4))
    ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2.2, color=INK, alpha=.6, linewidth=0, zorder=5)
    movable = int((~fixed[:nh]).sum())
    handles = [Patch(facecolor="white", edgecolor="#2b2b2b", linewidth=.6,
                     label=f"Hard macro, movable ({movable})")]
    if fixed[:nh].any():
        handles.append(Patch(facecolor="#3d3d3d", edgecolor="#1f1f1f", linewidth=.6,
                             label=f"Hard macro, fixed ({int(fixed[:nh].sum())})"))
    else:
        handles[0].set_label(f"Hard macro ({movable}, all movable)")
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=2.8, color=INK, alpha=.7,
                          label=f"Soft cluster center ({len(pos) - nh})"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=len(handles),
              borderaxespad=0, handlelength=1.3, columnspacing=1.6)
    finish_colorbar(fig, mesh, cax, "Routing congestion per grid cell\n(demand ÷ capacity, larger of H and V)")
    return fig


def per_design_figure(records: dict, baselines: dict):
    fig, ax = plt.subplots(figsize=(6.7, 2.75), layout="constrained")
    x = np.arange(len(IBM))
    replace = np.array([baselines[n]["replace"] for n in IBM])
    sa_ratio = np.array([baselines[n]["sa"] for n in IBM]) / replace
    ax.axhline(1.0, color=BASELINE, linewidth=1.1, zorder=1)
    ax.text(len(IBM) - .55, 1.0, "RePlAce = 1", ha="right", va="bottom", fontsize=8, color=INK2)
    for offset, (key, record) in zip((-.22, 0, .22), records.items()):
        ax.scatter(x + offset, record["proxy"] / replace, s=20 if key != "single" else 24,
                   marker=MARKER[key], color=COLOR[key], edgecolor="white", linewidth=.5, zorder=3,
                   label=record["label"])
    for value in x[1::2]:
        ax.axvspan(value - .5, value + .5, color="#f6f6f6", zorder=0, linewidth=0)
    ax.text(len(IBM) - .55, .955, f"SA baseline is {sa_ratio.min():.2f}–{sa_ratio.max():.2f}× RePlAce "
            "on every design (above this range)", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [n.replace("ibm", "") for n in IBM])
    ax.set_xlim(-.6, len(IBM) - .4)
    ax.set_ylim(.5, 1.05)
    ax.set_xlabel("IBM design")
    ax.set_ylabel("Proxy ÷ published RePlAce\n(lower is better)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=.2, handletextpad=.3,
              columnspacing=1.6)
    return fig, sa_ratio


def component_figure(records: dict):
    a, b = records["v1"]["by_name"], records["single"]["by_name"]
    deltas = np.array([[b[n]["wirelength"] - a[n]["wirelength"],
                        .5 * (b[n]["density"] - a[n]["density"]),
                        .5 * (b[n]["congestion"] - a[n]["congestion"])] for n in IBM])
    mean = deltas.mean(axis=0)
    values = np.vstack([deltas, mean])
    x = np.append(np.arange(len(IBM)), len(IBM) + .6)
    fig, ax = plt.subplots(figsize=(6.7, 2.95), layout="constrained")
    labels = ["Wirelength", "½ · density", "½ · congestion"]
    positive, negative = np.zeros(len(x)), np.zeros(len(x))
    for column, (label, color) in enumerate(zip(labels, COMPONENT_COLORS)):
        part = values[:, column]
        bottom = np.where(part >= 0, positive, negative)
        ax.bar(x, part, bottom=bottom, width=.72, color=color, edgecolor="white", linewidth=.6,
               label=label, zorder=3)
        positive += np.where(part >= 0, part, 0)
        negative += np.where(part < 0, part, 0)
    net = ax.scatter(x, values.sum(axis=1), marker="_", s=70, color=INK, linewidth=1.3, zorder=4,
                     label="Net change")
    ax.axhline(0, color=AXIS, linewidth=.7, zorder=2)
    ax.axvline(len(IBM) - .2, color=GRID, linewidth=.8, zorder=1)
    share = mean[2] / mean.sum()
    ax.text(-.45, values.sum(axis=1).min() * 1.02, f"Mean (right): {minus(mean.sum())}, of which\n"
            f"½ · congestion {minus(mean[2])} ({100 * share:.1f}%)", ha="left", va="bottom", fontsize=8,
            color=INK)
    ax.set_xticks(x, [n.replace("ibm", "") for n in IBM] + ["Mean"])
    ax.set_xlim(-.6, x[-1] + .6)
    ax.set_xlabel("IBM design")
    ax.set_ylabel("Change in proxy cost\n(single pipeline − v1; below 0 is better)")
    ax.grid(axis="y", color=GRID, linewidth=.6, zorder=0)
    handles, names = ax.get_legend_handles_labels()
    order = [names.index(n) for n in labels + ["Net change"]]
    ax.legend([handles[i] for i in order], [names[i] for i in order], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=.2, columnspacing=1.6)
    return fig, deltas, mean


def time_quality_figure(records: dict):
    fig, ax = plt.subplots(figsize=(6.7, 2.75), layout="constrained")
    ax.axvline(3600, color=BASELINE, linewidth=.9, zorder=1)
    ax.text(3800, 1.36, "1 h limit\nper design", ha="left", va="top", fontsize=8, color=INK2)
    placement = {"v1": (-12, 0, "right"), "early_v2": (12, 0, "left"), "single": (-12, 0, "right")}
    for key, record in records.items():
        ax.scatter(record["seconds"], record["proxy"], s=12, marker=MARKER[key], color=COLOR[key],
                   alpha=.35, linewidth=0, zorder=2)
        ax.scatter([record["mean_seconds"]], [record["mean_proxy"]], s=72, marker=MARKER[key],
                   color=COLOR[key], edgecolor="white", linewidth=1, zorder=4, label=record["label"])
        dx, dy, align = placement[key]
        ax.annotate(f'mean {record["mean_proxy"]:.4f}\n{record["mean_seconds"]:,.0f} s',
                    xy=(record["mean_seconds"], record["mean_proxy"]), xytext=(dx, dy),
                    textcoords="offset points", ha=align, va="center", fontsize=8, color=INK,
                    bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": .8})
    ax.set_xscale("log")
    ax.set_xlim(250, 6000)
    ax.set_xticks([300, 600, 1200, 2400, 3600], ["300", "600", "1,200", "2,400", "3,600"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(.64, 1.37)
    ax.set_xlabel("Runtime per design (s, log scale)")
    ax.set_ylabel("Proxy cost\n(lower is better)")
    ax.grid(axis="y", color=GRID, linewidth=.6, zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=.2, handletextpad=.4,
              columnspacing=1.8)
    ax.text(.26, .025, "Faint markers: single designs; large markers: means over 17 designs.\n"
            "Historical records; hosts and timing boundaries differ.", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color=INK2)
    return fig


def curvature_figure(curvature: dict):
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.7, 2.9), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.0, 1.15]})
    stages = ["plateau", "first_stage", "cascade"]
    x = np.arange(3)
    for design in curvature["designs"]:
        values = [100 * (design[s] / design["plateau"] - 1) for s in stages]
        left.plot(x, values, color=CONTEXT, linewidth=.7, alpha=.7, zorder=2)
    means = curvature["means"]
    mean_change = [100 * (means[s] / means["plateau"] - 1) for s in stages]
    left.plot(x, mean_change, color=COLOR["v1"], linewidth=2, marker="o", markersize=4.5, zorder=4)
    for xi, stage, change in zip(x, stages, mean_change):
        # The plateau label sits up and to the right so it clears the y-axis spine.
        left.annotate(f"{means[stage]:.5f}", xy=(xi, change), xytext=(0, -9) if xi else (6, 7),
                      textcoords="offset points", ha="center" if xi else "left", va="top" if xi else "bottom",
                      fontsize=8, color=INK, zorder=5,
                      bbox={"boxstyle": "square,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 1})
    left.set_xticks(x, ["Plateau", "After first\ncurvature stage", "After\ncascade"])
    left.set_xlim(-.35, 2.25)
    left.set_ylim(-9.3, 1.0)
    left.set_ylabel("Change from plateau (%)\n(lower is better)")
    left.grid(axis="y", color=GRID, linewidth=.6, zorder=0)
    left.legend([Line2D([], [], color=CONTEXT, linewidth=.9), Line2D([], [], color=COLOR["v1"], linewidth=2,
                                                                     marker="o", markersize=4)],
                ["One design", "Mean (labels: mean proxy)"], loc="lower left", borderaxespad=.2)
    groups = curvature["groups"]
    entries = [("1", groups["eigenvector"]["1"]), ("2", groups["eigenvector"]["2"]),
               ("0.3", groups["epsilon"]["0.3"]), ("1.0", groups["epsilon"]["1.0"]),
               ("3.0", groups["epsilon"]["3.0"])]
    positions = np.array([0, 1, 2.7, 3.7, 4.7])
    fractions = [100 * e["successes"] / e["attempts"] for _, e in entries]
    right.bar(positions, fractions, width=.6, color=COLOR["v1"], zorder=3)
    for position, fraction, (_, entry) in zip(positions, fractions, entries):
        right.text(position, fraction + 2, f'{entry["successes"]}/{entry["attempts"]}', ha="center",
                   va="bottom", fontsize=8, color=INK)
    right.set_xticks(positions, [label for label, _ in entries])
    # The step is the norm of the whole displacement over all blocks; ε is the acceptance error in the text.
    for center, text in [(.5, "Eigenvector"), (3.7, "Total step norm (µm)")]:
        right.annotate(text, xy=(center, 0), xycoords=("data", "axes fraction"), xytext=(0, -20),
                       textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK)
    totals = curvature["totals"]
    overall = 100 * totals["successes"] / totals["attempts"]
    right.set_ylim(0, 114)
    right.set_yticks([0, 25, 50, 75, 100])
    right.set_xlim(-.6, 5.3)
    right.set_ylabel("Attempts below the plateau\nwith zero overlap (%)")
    right.grid(axis="y", color=GRID, linewidth=.6, zorder=0)
    right.text(5.3, 109, f"All: {totals['successes']}/{totals['attempts']} ({overall:.1f}%); "
               f"{totals['feasible']} of {totals['attempts']} feasible", ha="right", va="center", fontsize=8,
               color=INK2)
    # Panel letters only; the caption describes the panels.
    for ax, letter in ((left, "a"), (right, "b")):
        ax.set_title(letter, loc="left", fontsize=8.5, fontweight="bold", pad=3)
    return fig, mean_change


def displacement_figure(state: dict):
    nh = state["num_hard"]
    before, after, sizes = state["before"][:nh], state["after"][:nh], state["sizes"][:nh]
    delta = after - before
    magnitude = np.hypot(delta[:, 0], delta[:, 1])
    moved = magnitude > 1e-6
    # About 0.6 of the text width and nearly square, for a side caption.
    fig, ax, cax = canvas_axes(state, (4.0, 3.55), left=.5, bottom=.42, side=2.72)
    width, height = state["canvas"]
    patches = [Rectangle((after[i, 0] - sizes[i, 0] / 2, after[i, 1] - sizes[i, 1] / 2), sizes[i, 0],
                         sizes[i, 1]) for i in range(nh)]
    ax.add_collection(PatchCollection(patches, facecolor="#ececec", edgecolor="#c4c4c4", linewidth=.4,
                                      zorder=2))
    # Only the left and bottom spines are drawn, so outline the canvas to show where arrows leave it.
    ax.add_patch(Rectangle((0, 0), width, height, facecolor="none", edgecolor=AXIS, linewidth=.6, zorder=3))
    norm = Normalize(vmin=0, vmax=math.ceil(magnitude.max() * 10) / 10)
    order = np.argsort(magnitude[moved])
    start, step, size = before[moved][order], delta[moved][order], magnitude[moved][order]
    short = size < DOT_BELOW_UM
    ax.scatter(start[short, 0], start[short, 1], c=size[short], cmap=DISPLACEMENT_CMAP, norm=norm, s=5,
               edgecolor=DISPLACEMENT_CMAP(1.0), linewidth=.3, zorder=4)
    # Moves are at most about 1 µm on a 23 µm canvas, so arrows are lengthened by a stated factor. A few
    # arrows near the edge end outside the canvas; they are not clipped, so every head stays visible.
    arrows = ax.quiver(start[~short, 0], start[~short, 1], ARROW_SCALE * step[~short, 0],
                       ARROW_SCALE * step[~short, 1], size[~short], cmap=DISPLACEMENT_CMAP, norm=norm,
                       angles="xy", scale_units="xy", scale=1, width=.006, headwidth=3.4, headlength=3.6,
                       headaxislength=3.2, minlength=0, zorder=4, clip_on=False)
    finish_colorbar(fig, arrows, cax, "Hard-macro displacement (µm)")
    handles = [Line2D([], [], marker=r"$\rightarrow$", linestyle="none", markersize=9, color=DISPLACEMENT_CMAP(.6),
                      label=f"Arrows: moves drawn {ARROW_SCALE}× actual length"),
               Line2D([], [], marker="o", linestyle="none", markersize=2.6, color=DISPLACEMENT_CMAP(0.0),
                      markeredgecolor=DISPLACEMENT_CMAP(1.0), markeredgewidth=.3,
                      label=f"Moves < {DOT_BELOW_UM:g} µm shown as dots")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.02, 1.03), ncol=1, borderaxespad=0,
              handlelength=1.3, labelspacing=.3)
    return fig, magnitude, moved


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

DASH = "---"


def table(caption: str, label: str, alignment: str, header: list, rows: list, note: str = "",
          placement: str = "htbp") -> str:
    lines = [r"\begin{table}[%s]" % placement, r"\centering", r"\footnotesize",
             r"\caption{%s}" % caption, r"\label{%s}" % label, r"\begin{tabular}{%s}" % alignment,
             r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        if row == "midrule":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\smallskip", r"\begin{minipage}{0.92\linewidth}\footnotesize %s\end{minipage}" % note]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def minus(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace("-", "\u2212")


def grouped(value: float) -> str:
    """Whole number with the report's thousands separator (10,944); braces keep it safe in math mode."""
    return f"{value:,.0f}".replace(",", "{,}")


def percent(value: float) -> str:
    return r"$%s%.1f\%%$" % ("-" if value < 0 else "+", abs(value))


def runtime_text(listed):
    if listed is None:
        return DASH
    text = listed.replace("min/bench", r"\,min per design").replace("~", r"$\sim$")
    return text


def competition_table(lb: dict, by_rank: dict) -> tuple:
    replace = lb["baselines"]["replace"]["mean_proxy"]
    rows, shown = [], []
    for rank, name in [(1, "Archgen"), (2, "Carrotato"), (9, "Thinkorplace")]:
        row = by_rank[rank]
        delta = 100 * (row["mean_proxy"] / replace - 1)
        cells = [str(rank), name, f'{row["mean_proxy"]:.4f}', runtime_text(row["listed_runtime"]), percent(delta)]
        if rank == 9:
            cells = [r"\textbf{\boldmath %s}" % c for c in cells]
        rows.append(cells)
        shown.append({"rank": rank, "entry": name, "mean_proxy": row["mean_proxy"],
                      "listed_runtime": row["listed_runtime"], "vs_replace_percent": round(delta, 1),
                      "readme_line": row["line"]})
    rows.append([DASH, "RePlAce baseline", f"{replace:.4f}", DASH, DASH])
    sa = lb["baselines"]["sa"]["mean_proxy"]
    rows.append([DASH, "SA baseline", f"{sa:.4f}", DASH, percent(100 * (sa / replace - 1))])
    shown += [{"rank": None, "entry": "RePlAce baseline", "mean_proxy": replace,
               "readme_line": lb["baselines"]["replace"]["line"]},
              {"rank": None, "entry": "SA baseline", "mean_proxy": sa,
               "vs_replace_percent": round(100 * (sa / replace - 1), 1),
               "readme_line": lb["baselines"]["sa"]["line"]}]
    caption = ("Official, final leaderboard (%s). Mean proxy cost over the 17 IBM designs; "
               "lower is better. The organizers report %d verified eligible submissions. Runtimes are as "
               "listed by each entrant on its own hardware; the last column is the change relative to the "
               "RePlAce mean." % (LEADERBOARD_DATE, lb["field_size"]["verified_eligible_submissions"]))
    text = table(caption, "tab:competition", "clrrr",
                 ["Rank", "Entry", "Mean proxy", "Listed runtime", "vs RePlAce"], rows)
    return text, shown


def historical_table(records: dict, baselines: dict) -> tuple:
    rows = []
    for name in IBM:
        rows.append([name.upper(), f'{baselines[name]["sa"]:.4f}', f'{baselines[name]["replace"]:.4f}']
                    + [f'{record["by_name"][name]["proxy_cost"]:.5f}' for record in records.values()])
    sa_mean = float(np.mean([baselines[n]["sa"] for n in IBM]))
    replace_mean = float(np.mean([baselines[n]["replace"] for n in IBM]))
    rows += ["midrule", ["Mean", f"{sa_mean:.4f}", f"{replace_mean:.4f}"]
             + [f'{record["mean_proxy"]:.5f}' for record in records.values()]]
    header = ["Design", "SA", "RePlAce", records["v1"]["label"], r"Early v2",
              r"\shortstack{Single pipeline\\(20 May lineage)}"]
    caption = ("Historical records; hardware, budgets and concurrency differ. Proxy cost per design "
               "(lower is better) for three development records and the two published baselines listed "
               "by the challenge. Each record is below both baselines on all 17 designs.")
    return table(caption, "tab:historical-baselines", "lrrrrr", header, rows), {
        "sa_mean": sa_mean, "replace_mean": replace_mean}


def benchmark_table(rescore: dict, readme_rows: dict) -> tuple:
    stats = {row["name"]: row["parsed_benchmark_statistics"] for row in rescore["benchmarks"]}
    # The challenge README also tabulates macros and nets; record where it agrees with the parsed inputs.
    readme_agrees = [n for n in IBM if readme_rows[n]["readme_macros"] == str(stats[n]["num_hard_macros"])
                     and readme_rows[n]["readme_nets"].replace(",", "") == str(stats[n]["plc_net_count"])]
    # The table drops the fixed-macro column and says so in its note; fail if a design ever has one.
    assert all(stats[n]["num_fixed_hard_macros"] == 0 for n in IBM)
    rows, shown = [], {}
    for name in IBM:
        s = stats[name]
        movable = s["num_hard_macros"] - s["num_fixed_hard_macros"]
        rows.append([name.upper(), grouped(movable),
                     grouped(s["num_soft_macros"]), grouped(s["plc_net_count"]),
                     r'%.2f $\times$ %.2f' % (s["canvas_width"], s["canvas_height"]),
                     r'%d $\times$ %d' % (s["grid_cols"], s["grid_rows"])])
        shown[name] = {"movable_hard_macros": movable, "fixed_hard_macros": s["num_fixed_hard_macros"],
                       "soft_clusters": s["num_soft_macros"], "nets": s["plc_net_count"],
                       "canvas_width_um": s["canvas_width"], "canvas_height_um": s["canvas_height"],
                       "grid_cols": s["grid_cols"], "grid_rows": s["grid_rows"]}
    header = ["Design", r"\shortstack{Movable\\hard macros}", r"\shortstack{Soft\\clusters}", "Nets",
              r"\shortstack{Canvas $W \times H$\\($\mu$m)}", r"\shortstack{Grid\\columns $\times$ rows}"]
    caption = ("Benchmark statistics of the 17 IBM designs, parsed from the challenge inputs with the "
               "canonical evaluator's loader. Nets is the evaluator's net count; the grid is the "
               "density and congestion grid.")
    agree = ", ".join(n.upper() for n in readme_agrees) or "no design"
    note = ("No design has fixed hard macros. "
            "IBM05 is excluded because it has no macros; the benchmark set used by the challenge omits it. "
            "The macro and net counts summarized in the challenge README agree with these parsed values "
            "only for %s." % agree)
    return table(caption, "tab:benchmarks", "lrrrcc", header, rows, note), {
        "per_design": shown, "readme_macro_and_net_counts_agree_for": readme_agrees}


def curvature_table(curvature: dict, rescore: dict) -> tuple:
    mark = {True: r"$\checkmark$", False: r"$\times$"}
    rows = []
    for d in curvature["designs"]:
        rows.append([d["name"].upper(), f'{d["plateau"]:.5f}', f'{d["first_stage"]:.5f}', f'{d["cascade"]:.5f}',
                     f'{mark[d["first_stage_strict_valid"]]} / {mark[d["cascade_strict_valid"]]}',
                     grouped(d["cascade_seconds"])])
    means = curvature["means"]
    valid_first = sum(d["first_stage_strict_valid"] for d in curvature["designs"])
    valid_cascade = sum(d["cascade_strict_valid"] for d in curvature["designs"])
    mean_seconds = float(np.mean([d["cascade_seconds"] for d in curvature["designs"]]))
    rows += ["midrule", ["Mean", f'{means["plateau"]:.5f}', f'{means["first_stage"]:.5f}',
                         f'{means["cascade"]:.5f}', f"{valid_first} / {valid_cascade} of 17",
                         grouped(mean_seconds)]]
    header = ["Design", "Plateau", r"\shortstack{After first\\curvature stage}", r"\shortstack{After\\cascade}",
              r"\shortstack{Strict validity\\(first / cascade)}", r"\shortstack{Cascade stage\\time (s)}"]
    endpoints = [p["geometry_validation"] for row in rescore["benchmarks"] for p in row["placements"]]
    assert all(g["hard_macro_overlap"]["overlap_count"] == 0 for g in endpoints)
    excess = max(g["bounds"]["maximum_boundary_excess"] for g in endpoints)
    mantissa, exponent = f"{excess:.2e}".split("e")
    caption = ("Historical records from uncapped research runs (no one-hour limit). Proxy cost after each "
               "curvature stage (lower is better); the plateau is the better of the two v1 branches. "
               "Validity is a strict canvas check of the saved endpoints (Measurement); every failure "
               r"exceeds a canvas edge by at most $%s\times10^{%d}\,\mu$m and no endpoint has a "
               "hard-macro overlap." % (mantissa, int(exponent)))
    return table(caption, "tab:curvature-stages", "lrrrcr", header, rows), {
        "valid_first_stage": valid_first, "valid_cascade": valid_cascade, "mean_cascade_seconds": mean_seconds,
        "maximum_boundary_excess_um": excess}


# ---------------------------------------------------------------------------

def main() -> None:
    for directory in (FIGURES, GENERATED, INTERACTIVE):
        directory.mkdir(parents=True, exist_ok=True)
    referee = load_json(REFEREE)
    rescore = load_json(RESCORE)
    lb = freeze_leaderboard()
    by_rank = check_leaderboard(lb, referee)
    baselines = {row["name"]: row for row in lb["baseline_per_design"]}
    records = load_records(baselines)
    curvature = load_curvature(rescore)
    state = ibm01_state()
    style()
    outputs = []

    fig = result_context_figure(lb, by_rank)
    outputs.append(save_pdf(fig, "result_context.pdf"))
    for suffix, kwargs in [("svg", {"metadata": {"Date": None}}), ("png", {"dpi": 200, "metadata": {"Software": None}})]:
        fig = result_context_figure(lb, by_rank)
        path = INTERACTIVE / f"result-context.{suffix}"
        fig.savefig(path, facecolor="white", **kwargs)
        plt.close(fig)
        outputs.append(path)
    outputs.append(save_pdf(placement_figure(state), "placement_ibm01.pdf"))
    fig, sa_ratio = per_design_figure(records, baselines)
    outputs.append(save_pdf(fig, "per_design_vs_replace.pdf"))
    fig, deltas, mean_delta = component_figure(records)
    outputs.append(save_pdf(fig, "component_deltas.pdf"))
    outputs.append(save_pdf(time_quality_figure(records), "time_vs_quality.pdf"))
    fig, mean_change = curvature_figure(curvature)
    outputs.append(save_pdf(fig, "curvature_attempts.pdf"))
    fig, magnitude, moved = displacement_figure(state)
    outputs.append(save_pdf(fig, "ibm01_displacement.pdf"))

    text, competition_rows = competition_table(lb, by_rank)
    write_text(GENERATED / "competition_context.tex", text)
    text, historical_means = historical_table(records, baselines)
    write_text(GENERATED / "historical_vs_baselines.tex", text)
    text, benchmark_rows = benchmark_table(rescore, baselines)
    write_text(GENERATED / "benchmark_stats.tex", text)
    text, curvature_extra = curvature_table(curvature, rescore)
    write_text(GENERATED / "curvature_stages.tex", text)
    outputs += [GENERATED / f for f in ("competition_context.tex", "historical_vs_baselines.tex",
                                        "benchmark_stats.tex", "curvature_stages.tex")]

    replace = np.array([baselines[n]["replace"] for n in IBM])
    ratios = {key: {n: float(r) for n, r in zip(IBM, record["proxy"] / replace)} for key, record in records.items()}
    nh = state["num_hard"]
    soft_shift = np.hypot(*(state["after"][nh:] - state["before"][nh:]).T)
    leaderboard_source = {"path": rel(LEADERBOARD), "sha256": sha256(LEADERBOARD),
                          "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM,
                          "upstream_revision_date": LEADERBOARD_DATE,
                          "upstream_readme_sha256": lb["source"]["sha256"],
                          "upstream_readme_git_blob_sha1": lb["source"]["git_blob_sha1"]}
    summary = {
        "kind": "report_v1_figure_and_table_values",
        "producer": source(Path(__file__)),
        "evidence_labels": {"Official": "organizers' final leaderboard",
                            "Historical": "May 2026 development records; unmatched hardware, budgets and concurrency",
                            "Measurement": "post-competition rescoring and parsing of frozen inputs"},
        "display_names": {key: label for key, label, _ in RECORDS},
        "display_name_note": ("The v1 record is the 17 May full-suite run of the v1 family "
                              "(raw file timestamp 2026-05-17T10:24:36Z). It is not the 13 May leaderboard "
                              "entry, which the leaderboard lists only as 'Was 1.0771'."),
        "font": "Latin Modern Roman (TeX Live lm package, lmroman10-*.otf); mathtext Computer Modern",
        "palette": {"context_grey": CONTEXT, "baseline_grey": BASELINE, "records": COLOR,
                    "components": dict(zip(["wirelength", "half_density", "half_congestion"], COMPONENT_COLORS))},
        "outputs": [{"path": rel(p), "sha256": sha256(p)} for p in outputs],
        "figures": {
            "result_context": {
                "files": ["writeup/latex/figures/result_context.pdf", "writeup/interactive/result-context.svg",
                          "writeup/interactive/result-context.png"],
                "evidence_label": "Official",
                "top_ten": [{"rank": by_rank[r]["rank"], "team": by_rank[r]["team"],
                             "mean_proxy": by_rank[r]["mean_proxy"], "readme_line": by_rank[r]["line"]}
                            for r in range(1, 11)],
                "labelled": {"Archgen": 0.9507, "Carrotato": 0.9522, "Thinkorplace": 0.9974},
                "baselines": {"RePlAce": lb["baselines"]["replace"]["mean_proxy"],
                              "SA": lb["baselines"]["sa"]["mean_proxy"]},
                "verified_eligible_submissions": lb["field_size"]["verified_eligible_submissions"],
                "ranked_rows": lb["ranked_row_count"],
                "thinkorplace_vs_replace_percent": 100 * (0.9974 / lb["baselines"]["replace"]["mean_proxy"] - 1),
                "thinkorplace_vs_winner_percent": 100 * (0.9974 / 0.9507 - 1),
                "sources": [leaderboard_source],
            },
            "placement_ibm01": {
                "file": "writeup/latex/figures/placement_ibm01.pdf",
                "evidence_label": "Measurement (canonical evaluator on the frozen historical endpoint)",
                "placement": "after-cascade IBM01 endpoint (raw/placements/cascade_ibm01.pt)",
                "underlay": "routing_congestion",
                "underlay_definition": ("per grid cell, max(H, V) of the canonical evaluator's smoothed routing "
                                        "congestion (routing demand / capacity, macro blockage included) after "
                                        "get_congestion_cost(); the congestion cost is the mean of the top 5% of "
                                        "the pooled H and V cell values"),
                "grid_rows": state["grid"][0], "grid_cols": state["grid"][1],
                "canvas_um": list(state["canvas"]),
                "congestion_cell_max": float(state["congestion"].max()),
                "congestion_cell_mean": float(state["congestion"].mean()),
                "colorbar_max": math.ceil(state["congestion"].max() * 5) / 5,
                "recomputed_components": state["components"],
                "recorded_rescore_proxy": state["recorded_proxy"],
                "hard_macros": nh, "fixed_hard_macros": int(state["fixed"][:nh].sum()),
                "soft_clusters": int(len(state["after"]) - nh),
                "strict_validation": True,
                "sources": state["sources"],
            },
            "per_design_vs_replace": {
                "file": "writeup/latex/figures/per_design_vs_replace.pdf",
                "evidence_label": "Historical (records) and Official (published baselines)",
                "ratios": ratios,
                "ratio_ranges": {key: [min(v.values()), max(v.values())] for key, v in ratios.items()},
                "sa_over_replace_range": [float(sa_ratio.min()), float(sa_ratio.max())],
                "sources": [records[k]["source"] for k in records] + [leaderboard_source],
            },
            "component_deltas": {
                "file": "writeup/latex/figures/component_deltas.pdf",
                "evidence_label": "Historical",
                "definition": "single pipeline minus v1 (17 May), per design; negative is an improvement",
                "per_design": {n: {"wirelength": float(d[0]), "half_density": float(d[1]),
                                   "half_congestion": float(d[2]), "net": float(d.sum())}
                               for n, d in zip(IBM, deltas)},
                "mean": {"wirelength": float(mean_delta[0]), "half_density": float(mean_delta[1]),
                         "half_congestion": float(mean_delta[2]), "net": float(mean_delta.sum())},
                "congestion_share_of_mean_decrease": float(mean_delta[2] / mean_delta.sum()),
                "designs_lower": int((deltas.sum(axis=1) < 0).sum()),
                "designs_higher": [n for n, d in zip(IBM, deltas) if d.sum() > 0],
                "mean_change_percent": 100 * (records["single"]["mean_proxy"] / records["v1"]["mean_proxy"] - 1),
                "sources": [records["v1"]["source"], records["single"]["source"]],
            },
            "time_vs_quality": {
                "file": "writeup/latex/figures/time_vs_quality.pdf",
                "evidence_label": "Historical; hosts and timing boundaries differ",
                "records": {key: {"mean_proxy": r["mean_proxy"], "mean_seconds": r["mean_seconds"],
                                  "min_seconds": float(r["seconds"].min()), "max_seconds": float(r["seconds"].max()),
                                  "per_design_seconds": dict(zip(IBM, map(float, r["seconds"]))),
                                  "per_design_proxy": dict(zip(IBM, map(float, r["proxy"]))),
                                  "proxy_range": [float(r["proxy"].min()), float(r["proxy"].max())]}
                            for key, r in records.items()},
                "reference_line_seconds": 3600,
                "sources": [records[k]["source"] for k in records],
            },
            "curvature_attempts": {
                "file": "writeup/latex/figures/curvature_attempts.pdf",
                "evidence_label": "Historical (uncapped research runs)",
                "stage_means": curvature["means"],
                "mean_change_from_plateau_percent": dict(zip(["plateau", "first_stage", "cascade"], mean_change)),
                "per_design": curvature["designs"],
                "attempt_totals": curvature["totals"],
                "success_definition": "feasible, polished overlap 0, polished proxy below the plateau",
                "step_definition": ("groups['epsilon'] keys are the total step norm in µm: the Euclidean norm of "
                                    "the whole displacement over all movable coordinates, not a per-block step"),
                "groups": curvature["groups"],
                "sources": curvature["sources"],
            },
            "ibm01_displacement": {
                "file": "writeup/latex/figures/ibm01_displacement.pdf",
                "evidence_label": "Historical (frozen endpoints of an uncapped research run)",
                "before_proxy": state["first_stage_proxy"], "after_proxy": state["recorded_proxy"],
                "hard_macros_moved": int(moved.sum()), "hard_macros": nh,
                "hard_displacement_um": {"max": float(magnitude.max()),
                                         "median_of_moved": float(np.median(magnitude[moved])),
                                         "mean_all": float(magnitude.mean())},
                "colorbar_max_um": math.ceil(magnitude.max() * 10) / 10,
                "arrows": ("from the before position along the displacement, lengthened by arrow_scale and not "
                           "clipped at the canvas edge; moves shorter than dots_below_um are drawn as dots"),
                "arrow_scale": ARROW_SCALE,
                "dots_below_um": DOT_BELOW_UM,
                "hard_macros_drawn_as_dots": int((moved & (magnitude < DOT_BELOW_UM)).sum()),
                "soft_clusters_moved_not_drawn": int((soft_shift > 1e-6).sum()),
                "soft_displacement_max_um": float(soft_shift.max()),
                "sources": state["sources"][:2],
            },
        },
        "tables": {
            "competition_context": {"file": "writeup/latex/generated/competition_context.tex",
                                    "label": "tab:competition", "rows": competition_rows,
                                    "sources": [leaderboard_source]},
            "historical_vs_baselines": {"file": "writeup/latex/generated/historical_vs_baselines.tex",
                                        "label": "tab:historical-baselines",
                                        "record_means": {k: r["mean_proxy"] for k, r in records.items()},
                                        "baseline_means": historical_means,
                                        "per_design": {n: {"sa": baselines[n]["sa"], "replace": baselines[n]["replace"],
                                                           **{k: r["by_name"][n]["proxy_cost"] for k, r in records.items()}}
                                                       for n in IBM},
                                        "sources": [records[k]["source"] for k in records] + [leaderboard_source]},
            "benchmark_stats": {"file": "writeup/latex/generated/benchmark_stats.tex", "label": "tab:benchmarks",
                                **benchmark_rows, "net_count_field": "plc_net_count",
                                "ibm05_exclusion_source": ("external/MacroPlacement/Testcases/ICCAD04/README.md: "
                                                           "'We do not include the ibm05 testcase as it does not "
                                                           "have any macros.'"),
                                "sources": [source(RESCORE), leaderboard_source]},
            "curvature_stages": {"file": "writeup/latex/generated/curvature_stages.tex",
                                 "label": "tab:curvature-stages", **curvature_extra,
                                 "means": curvature["means"],
                                 "sources": curvature["sources"] + [source(RESCORE)]},
        },
    }
    write_text(SUMMARY, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"outputs": [rel(p) for p in outputs] + [rel(SUMMARY), rel(LEADERBOARD)],
                      "ratio_ranges": summary["figures"]["per_design_vs_replace"]["ratio_ranges"],
                      "congestion_share": summary["figures"]["component_deltas"]["congestion_share_of_mean_decrease"]},
                     indent=2))


if __name__ == "__main__":
    main()
