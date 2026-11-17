#!/usr/bin/env python3
"""Render the website chart of every ranked entry on the official final leaderboard.

Reads only the frozen parse writeup/data/publication/leaderboard_996193c.json and writes
writeup/interactive/leaderboard-all.svg. The report's Figure 1 (top ten) is unchanged; this
chart is for the website article.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_publication_v1 import (ACCENT, AXIS, BASELINE, CONTEXT, GRID, INK2, INTERACTIVE,  # noqa: E402
                                   LEADERBOARD, LEADERBOARD_DATE, load_json, style)

OUTPUT = INTERACTIVE / "leaderboard-all.svg"
Y_MAX = 2.3  # Two entries lie far above the SA baseline; they are marked at the top edge instead.


def figure(lb: dict):
    ranked = [row for row in lb["rows"] if isinstance(row["rank"], int)]
    assert len(ranked) == lb["ranked_row_count"] == 129
    ours = next(row for row in ranked if row["team"] == "thinkorplace")
    assert (ours["rank"], ours["mean_proxy"]) == (9, 0.9974)
    unplotted = [row for row in ranked if row["mean_proxy"] is None]
    assert [row["rank"] for row in unplotted] == [30]  # Listed as "~1.16", an approximate score.
    plotted = [row for row in ranked if row["mean_proxy"] is not None and row is not ours]
    inside = [row for row in plotted if row["mean_proxy"] <= Y_MAX]
    above = [row for row in plotted if row["mean_proxy"] > Y_MAX]

    fig, ax = plt.subplots(figsize=(6.7, 3.1))
    fig.subplots_adjust(left=.085, right=.80, top=.9, bottom=.2)
    ax.grid(axis="y", color=GRID, linewidth=.6, zorder=0)
    for verified, face in ((True, CONTEXT), (False, "white")):
        rows = [row for row in inside if row["verified"] == verified]
        ax.scatter([r["rank"] for r in rows], [r["mean_proxy"] for r in rows], s=15, facecolor=face,
                   edgecolor=CONTEXT if not verified else "white", linewidth=.7, zorder=3)
    for row in above:
        ax.scatter([row["rank"]], [Y_MAX], s=22, marker="^", facecolor="white", edgecolor=CONTEXT,
                   linewidth=.7, zorder=3, clip_on=False)
    ax.text(129, Y_MAX + .04, "Off scale, ranks 128 and 129 at %s and %s" % tuple(r["mean_proxy_text"] for r in above),
            ha="right", va="bottom", fontsize=7.5, color=INK2, clip_on=False)
    ax.scatter([ours["rank"]], [ours["mean_proxy"]], s=48, color=ACCENT, edgecolor="white", linewidth=.8,
               zorder=4)
    ax.annotate(f'Thinkorplace, ninth, {ours["mean_proxy_text"]}', (ours["rank"], ours["mean_proxy"]),
                xytext=(14, .93), ha="left", va="center", fontsize=8.5, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": .7, "shrinkA": 1, "shrinkB": 4})
    for key, label in (("replace", "RePlAce baseline"), ("sa", "SA baseline")):
        value = lb["baselines"][key]["mean_proxy"]
        ax.axhline(value, color=BASELINE, linewidth=.9, linestyle=(0, (4, 2)), zorder=2)
        ax.text(131, value, f"{label}\n{value:.4f}", ha="left", va="center", fontsize=8, color=INK2,
                clip_on=False)
    ax.set_xlim(-1, 130)
    ax.set_ylim(.88, Y_MAX)
    ax.set_xticks([1, 20, 40, 60, 80, 100, 129])
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
    ax.set_xlabel("Final rank")
    ax.set_ylabel("Mean proxy cost, lower is better")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=CONTEXT,
               markeredgecolor=CONTEXT, label="Verified by the organizers"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor="white",
               markeredgecolor=CONTEXT, label="Not verified"),
    ], loc="lower right", handletextpad=.3, borderaxespad=.2)
    fig.text(.085, .025, f"Official final leaderboard, {LEADERBOARD_DATE}. All {len(ranked)} ranked entries. "
             "Rank 30 lists an approximate score and is not plotted.", ha="left", fontsize=7.5, color=INK2)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    return fig


def main() -> None:
    style()
    fig = figure(load_json(LEADERBOARD))
    fig.savefig(OUTPUT, facecolor="white", metadata={"Date": None})
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(INTERACTIVE.parents[1])}")


if __name__ == "__main__":
    main()
