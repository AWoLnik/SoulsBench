"""Shareable win-rate column chart for a finished bench (900x900 PNG, dark-fantasy styled).

    uv run --with matplotlib python scripts/win_rate_chart.py --fonts <dir with Cinzel.ttf, Inter.ttf>

Reads runs/bench_full50-*/<adapter>/scorecard.json, so every number is the benchmark's own.
Fonts: Cinzel and Inter (SIL Open Font License), from github.com/google/fonts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BG = "#14110E"
GOLD = "#D4A24C"
TITLE = "#E6C98F"
TEXT = "#EDE3D1"
TEXT_2 = "#B8AC98"
MUTED = "#8C8172"
GRID = "#2A251F"
RULE = "#9C8B6E"

MODELS = [  # bench dir, adapter, one-line label
    ("full50-hosted", "openai", "OpenAI GPT-6 Luna"),
    ("full50-hosted", "jev", "TypeSafe Jev"),
    ("full50-hosted", "pplx", "Perplexity Decider"),
    ("full50-kev", "kev", "Kev 0.8B"),
    # Laya's typed-decisions checkpoint (~210M params) is its entry: built for typed choices.
    ("full50-laya-typed", "laya-typed", "Laya 210M"),
]


def load(runs: Path):
    rows = []
    for bench, adapter, label in MODELS:
        card = json.loads((runs / f"bench_{bench}" / adapter / "scorecard.json").read_text())
        rows.append(
            {
                "label": label,
                "rate": card["win_rate"],
                "p50": card["latency_ms_p50"],
                "p95": card["latency_ms_p95"],
                "cost": card["cost_per_episode_usd"],
            }
        )
    return sorted(rows, key=lambda r: -r["rate"])


def column(ax, x, height, width, color, radius_px):
    """Column with a rounded top (data end) and a square baseline."""
    if height <= 0:
        return
    y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.get_window_extent().height
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.get_window_extent().width
    ry = min(radius_px * y_per_px, height / 2)
    rx = radius_px * x_per_px
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, 0),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rx}",
            mutation_aspect=ry / rx,
            linewidth=0,
            facecolor=color,
            zorder=3,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (x - width / 2, 0), width, min(ry, height), linewidth=0, facecolor=color, zorder=3
        )
    )  # square at the baseline


def cost_text(cost: float | None) -> str:
    if cost is None:
        return "n/a"
    if cost == 0:
        return "run locally"
    return f"${cost:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/full50_win_rates.png"))
    args = ap.parse_args()
    for f in ("Cinzel.ttf", "Inter.ttf"):
        font_manager.fontManager.addfont(str(args.fonts / f))
    title_font = font_manager.FontProperties(fname=str(args.fonts / "Cinzel.ttf"))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(args.fonts / "Inter.ttf")
    ).get_name()

    rows = load(args.runs)
    fig = plt.figure(figsize=(4.5, 4.5), dpi=200, facecolor=BG)  # 900x900

    # Backdrop: ember glow rising from the bottom and a soft vignette (a bonfire in the dark).
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.set_axis_off()
    yy, xx = np.mgrid[0:1:450j, 0:1:450j]
    glow = np.exp(-(((xx - 0.5) / 0.6) ** 2 + ((yy - 1.08) / 0.4) ** 2))
    vignette = 1 - 0.55 * np.clip(((xx - 0.5) / 0.7) ** 2 + ((yy - 0.5) / 0.7) ** 2, 0, 1)
    base = np.array([0x14, 0x11, 0x0E]) / 255
    ember = np.array([0x5A, 0x2A, 0x0C]) / 255
    img = (base + glow[..., None] * (ember - base) * 0.75) * vignette[..., None]
    bg.imshow(img, extent=[0, 1, 0, 1], aspect="auto", origin="upper")

    # Title block
    fig.text(
        0.5,
        0.905,
        "DECISION MODELS CAN PLAY",
        fontproperties=title_font,
        fontsize=15,
        color=TITLE,
        ha="center",
    )
    fig.text(
        0.5,
        0.845,
        "DARK SOULS III",
        fontproperties=title_font,
        fontsize=15,
        color=TITLE,
        ha="center",
    )
    fig.text(
        0.5,
        0.79,
        "Win rate over 50 realtime boss fights against Iudex Gundyr",
        fontsize=6.2,
        color=TEXT_2,
        ha="center",
    )
    rule = fig.add_axes([0.07, 0.755, 0.86, 0.02], zorder=1)
    rule.set_axis_off()
    rule.set_xlim(0, 1)
    rule.set_ylim(-1, 1)
    rule.plot([0, 0.47], [0, 0], color=RULE, lw=0.5, alpha=0.7)
    rule.plot([0.53, 1], [0, 0], color=RULE, lw=0.5, alpha=0.7)
    rule.plot([0.5], [0], marker="D", ms=2.6, color=GOLD)

    # Columns
    ax = fig.add_axes([0.18, 0.29, 0.75, 0.42], zorder=2)  # starts just after the row labels
    ax.set_facecolor("none")
    n = len(rows)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 0.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for t in (0.1, 0.2, 0.3, 0.4):
        ax.axhline(t, color=GRID, lw=0.5, zorder=1)
        ax.text(
            -0.5, t + 0.006, f"{int(t * 100)}%", fontsize=4.4, color=MUTED, ha="left", va="bottom"
        )
    fig.canvas.draw()  # the window extent sizes the rounded ends in pixels
    for i, r in enumerate(rows):
        column(ax, i, r["rate"], 0.42, GOLD, radius_px=6)
        ax.text(
            i,
            r["rate"] + 0.012,
            f"{r['rate']:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )
        fig_x = ax.transData.transform((i, 0))[0] / fig.bbox.width
        fig.text(
            fig_x,
            0.268,
            r["label"].replace(" ", "\n", 1),
            ha="center",
            va="top",
            fontsize=6,
            color=TEXT,
            linespacing=1.25,
        )
        fig.text(fig_x, 0.155, f"{r['p50']:.0f} ms", ha="center", fontsize=5.6, color=TEXT_2)
        fig.text(fig_x, 0.105, cost_text(r["cost"]), ha="center", fontsize=5.6, color=TEXT_2)
    # One row label per row instead of repeating it under every column
    for y, label in ((0.155, "p50 latency"), (0.105, "Cost per fight")):
        fig.text(0.07, y, label, ha="left", fontsize=5.6, color=MUTED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=BG)
    print(args.out)


if __name__ == "__main__":
    main()
