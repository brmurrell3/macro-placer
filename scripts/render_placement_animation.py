#!/usr/bin/env python3
"""Animate one Thinkorplace run on ibm01 for the README and the website article.

With --raw, build the run's frame record from the local recorder's pickle. It keeps the
accepted descent attempt (every tenth Adam step), the legalized placement and a subset of
coordinate-descent frames, scores every kept frame with the reference evaluator, and
writes animation_frames.npz and animation_frames.json into the run directory. Without
--raw, draw writeup/interactive/thinkorplace-ibm01.gif from that record alone.

The positions come from recorder hooks applied locally to the runtime for this one run and
removed afterwards. They are not part of the frozen runtime or this repository, and they
copied positions without changing any solver decision.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_publication_v1 import ACCENT, GRID, INK2, INTERACTIVE, ROOT, style  # noqa: E402

BLUE = "#1769AA"
OUTPUT = INTERACTIVE / "thinkorplace-ibm01.gif"
DESCENT_EVERY = 10   # keep every tenth Adam step
CD_FRAMES = 60       # coordinate-descent frames, spaced evenly in accepted moves
STAGE_TITLE = {
    "init": "Starting placement",
    "descent": "Gradient descent on the smooth score",
    "legalized": "Legalized, no overlaps",
    "cd": "Coordinate descent on the exact score",
    "final": "Final placement",
}


def load_ibm01():
    from macro_place.loader import load_benchmark_from_dir
    from thinkorplace.common.bench_paths import find_benchmark_dir
    return load_benchmark_from_dir(str(find_benchmark_dir("ibm01")))


def select(frames: list) -> list:
    """The accepted attempt's frames, thinned for the animation."""
    cd_start = max(i for i, f in enumerate(frames) if f["stage"] == "cd_start")
    first = max(i for i, f in enumerate(frames[:cd_start]) if f["stage"] == "init")
    attempt = frames[first:]
    kept = [attempt[0]]
    descent = [f for f in attempt if f["stage"] == "descent"]
    kept += [f for f in descent if f["step"] % DESCENT_EVERY == 0 or f is descent[-1]]
    kept.append(next(f for f in attempt if f["stage"] == "cd_start"))
    cd = [f for f in attempt if f["stage"] == "cd"]
    final = next(f for f in attempt if f["stage"] == "final")
    counts = np.array([f["count"] for f in cd])
    picked = []
    for k in range(1, CD_FRAMES + 1):
        index = int(np.abs(counts - counts[-1] * k / CD_FRAMES).argmin())
        if index not in picked:
            picked.append(index)
    kept += [cd[i] for i in sorted(picked)]
    kept.append(final)
    return kept


def build_record(raw: Path, run: Path) -> None:
    from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
    with raw.open("rb") as fh:
        frames = select(pickle.load(fh))
    benchmark, plc = load_ibm01()
    t_start = frames[0]["t"]
    rows = []
    for number, frame in enumerate(frames):
        pos = torch.from_numpy(frame["pos"]).to(torch.float32)
        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        overlaps = int(compute_overlap_metrics(pos, benchmark)["overlap_count"])
        stage = "legalized" if frame["stage"] == "cd_start" else frame["stage"]
        rows.append({"frame": number, "stage": stage, "elapsed_s": round(frame["t"] - t_start, 3),
                     "adam_step": frame.get("step"), "accepted_moves": frame["count"] if stage == "cd" else None,
                     "proxy": proxy, "hard_macro_overlaps": overlaps})
        print(f"frame {number:3d} {stage:9s} t={rows[-1]['elapsed_s']:7.1f}s proxy={proxy:.5f} ovl={overlaps}")
    np.savez_compressed(run / "animation_frames.npz", positions=np.stack([f["pos"] for f in frames]).astype(np.float32))
    record = {
        "kind": "placement_animation_frames",
        "evidence_label": "Measurement",
        "benchmark": "ibm01",
        "run_directory": str(run.relative_to(ROOT)),
        "positions": "animation_frames.npz (float32, frames x macros x 2, macro centers in um)",
        "scoring": "every frame rescored with macro_place.objective.compute_proxy_cost (reference evaluator)",
        "selection": (f"the accepted descent attempt from its initializer, every {DESCENT_EVERY}th Adam step "
                      f"and the last, the placement handed to coordinate descent, up to {CD_FRAMES} coordinate-descent "
                      "samples spaced evenly in accepted moves, and the final placement"),
        "instrumentation": ("recorder hooks applied locally to the runtime for this run only and removed afterwards; "
                            "they are not part of the frozen runtime or this repository. They copied positions "
                            "and changed no solver decision. "
                            "Coordinate-descent positions were sampled at most once per second."),
        "frames": rows,
    }
    (run / "animation_frames.json").write_text(json.dumps(record, indent=1) + "\n")


def render(run: Path) -> None:
    record = json.loads((run / "animation_frames.json").read_text())
    positions = np.load(run / "animation_frames.npz")["positions"]
    rows = record["frames"]
    benchmark, _ = load_ibm01()
    hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.numpy()
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    t = np.array([r["elapsed_s"] for r in rows])
    proxy = np.array([r["proxy"] for r in rows])
    cd_from = next(r["elapsed_s"] for r in rows if r["stage"] == "legalized")
    style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})

    images, durations = [], []
    for number, row in enumerate(rows):
        fig = plt.figure(figsize=(12, 5.4), dpi=100)
        canvas = fig.add_axes([0.03, 0.12, 0.42, 0.76])
        curve = fig.add_axes([0.56, 0.2, 0.41, 0.68])
        exact_stage = row["stage"] in ("legalized", "cd", "final")
        color = ACCENT if exact_stage else BLUE
        pos = positions[number]
        moved = np.zeros(hard, bool)
        if row["stage"] == "cd" and number > 0:
            moved = np.any(np.abs(pos[:hard] - positions[number - 1][:hard]) > 1e-9, axis=1)
        canvas.scatter(pos[hard:, 0], pos[hard:, 1], s=1.5, color="#b5b5b5", linewidth=0, zorder=1)
        rects = [Rectangle((x - w / 2, y - h / 2), w, h) for (x, y), (w, h) in zip(pos[:hard], sizes[:hard])]
        canvas.add_collection(PatchCollection(rects, facecolor=[color] * hard, alpha=.42, edgecolor=color,
                                              linewidth=.5, zorder=2))
        if moved.any():
            canvas.add_collection(PatchCollection([r for r, m in zip(rects, moved) if m], facecolor=ACCENT,
                                                  alpha=.95, edgecolor="#7a3f0c", linewidth=.6, zorder=3))
        canvas.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="#333333", linewidth=1, zorder=4))
        canvas.set_xlim(-.3, cw + .3)
        canvas.set_ylim(-.3, ch + .3)
        canvas.set_aspect("equal")
        canvas.axis("off")
        canvas.set_title(STAGE_TITLE[row["stage"]], fontsize=14, color=color, pad=8)
        detail = (f"{row['elapsed_s']:.0f} s    exact proxy {row['proxy']:.4f}    "
                  + (f"{row['hard_macro_overlaps']} overlapping pairs" if row["hard_macro_overlaps"]
                     else "no overlaps"))
        fig.text(0.24, 0.06, detail, ha="center", fontsize=12, color=INK2)

        curve.axvspan(0, cd_from, color=BLUE, alpha=.06, linewidth=0)
        curve.axvspan(cd_from, t[-1], color=ACCENT, alpha=.06, linewidth=0)
        split = int(np.searchsorted(t, cd_from))
        # The starting placement is a lone point: no step runs while the smooth score is set up.
        curve.scatter([t[0]], [proxy[0]], s=14, color=BLUE, linewidth=0, zorder=3)
        curve.plot(t[1:min(number + 1, split + 1)], proxy[1:min(number + 1, split + 1)], color=BLUE, linewidth=1.6)
        if number >= split:
            curve.plot(t[split:number + 1], proxy[split:number + 1], color=ACCENT, linewidth=1.6)
        curve.scatter([t[number]], [proxy[number]], s=36, color=color, edgecolor="white", linewidth=.8, zorder=3)
        # Descent takes seconds and coordinate descent minutes, so time is linear below 1 s, log above.
        curve.set_xscale("symlog", linthresh=1, linscale=.4)
        curve.set_xlim(0, t[-1] * 1.08)
        curve.set_xticks([0, 1, 3, 10, 30, 100, 300])
        curve.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        curve.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for start, stop, label, tone in ((0, cd_from, "gradient descent", BLUE),
                                         (cd_from, t[-1], "coordinate descent", ACCENT)):
            middle = curve.transData.inverted().transform(
                (curve.transData.transform([(start, 0), (stop, 0)])[:, 0].mean(), 0))[0]
            curve.text(middle, 1.0, label, transform=curve.get_xaxis_transform(), ha="center", va="bottom",
                       fontsize=11, color=tone)
        lo, hi = proxy.min(), proxy.max()
        curve.set_ylim(lo - .04 * (hi - lo), hi + .04 * (hi - lo))
        curve.grid(axis="y", color=GRID, linewidth=.6)
        curve.set_xlabel("Elapsed seconds, log scale above 1 s")
        curve.set_ylabel("Exact proxy cost, lower is better")
        fig.text(0.5, 0.012, "Thinkorplace on ibm01. A post-competition laptop run of the configuration most "
                 "likely ranked. Each frame is scored with the reference evaluator.", ha="center", fontsize=10,
                 color=INK2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor="white", dpi=100)
        plt.close(fig)
        images.append(Image.open(buffer).convert("RGB"))
        durations.append(2500 if row["stage"] == "final" else 700 if row["stage"] == "legalized" else 100)

    sample = Image.new("RGB", (images[0].width, images[0].height * 3))
    for slot, index in enumerate((0, len(images) // 3, len(images) - 1)):
        sample.paste(images[index], (0, slot * images[0].height))
    palette = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(frames)} frames, {OUTPUT.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "writeup/runs/animation-ibm01-20261117")
    parser.add_argument("--raw", type=Path, help="recorder pickle; rebuilds the frame record before rendering")
    args = parser.parse_args()
    if args.raw:
        build_record(args.raw, args.run.resolve())
    render(args.run.resolve())


if __name__ == "__main__":
    main()
