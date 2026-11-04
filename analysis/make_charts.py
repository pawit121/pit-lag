#!/usr/bin/env python3
"""Draw the article's two charts from the results tables.

Chart 1, filing_curves: the share of reports filed by day N after the period ends,
the US against Korea, for quarterly and annual reports.
Chart 2, leaks: the share of reports each lag rule uses before they were public,
the US against Korea.

Each chart is written in a light and a dark version, as SVG and PNG, to
results/charts/. The folder also gets leak_comparison.csv, the numbers behind
Chart 2 for the article's table (Korea with its next-day and same-day bounds).

Run it after us_filing_lags.py and kr_filing_lags.py:
    python make_charts.py
Needs pandas and matplotlib.

Korea's filings carry a date but no time. Where that matters, Chart 2 shows an
estimate: the same-day share plus the after-close share of deadline-day filings,
measured on KRX's KIND (results/kr_kind_timing_sample.csv), times the reports
filed on the attach date.
"""

from __future__ import annotations

import argparse
import math
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Patch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

HERE = Path(__file__).resolve().parent

# Validated two-slot palette (slot 1 blue, slot 2 orange) and chart chrome, per mode.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", us="#2a78d6", kr="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", us="#3987e5", kr="#d95926"),
}
SANS = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
LINE_PT = 1.5          # 2px lines
BAR_PT = 9.0           # bar thickness, well under the 24px cap
RADIUS_PT = 3.0        # 4px rounded data end
GAP_PT = 1.5           # 2px gap between touching bars
SOURCES = "Sources: SEC Financial Statement Data Sets; OpenDART; KRX KIND. Periods ending 2015–2025."

LEAK_ROWS = [  # label, US convention, Korea convention
    ("US large-company deadlines\n(40 days quarterly, 60 annual)",
     "US large-accelerated deadlines for all (10-Q 40, 10-K 60)",
     "US large-accelerated deadlines (quarterly and half-year 40, annual 60)"),
    ("Period end + 45 days", "period end + 45 days", "period end + 45 days"),
    ("Period end + 90 days", "period end + 90 days", "period end + 90 days"),
    ("Each market's legal deadline", "legal deadline for the filer's status",
     "Korean legal deadline (quarterly and half-year 45, annual 90)"),
]


def setup_fonts() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = [name for name in SANS if name in installed] or ["DejaVu Sans"]
    plt.rcParams.update({"font.family": family, "svg.fonttype": "none",
                         "axes.unicode_minus": False})


def style_axes(ax, theme: dict, grid_axis: str) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(colors=theme["muted"], labelcolor=theme["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)


def header(fig, theme: dict, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.955, title, color=theme["ink"], fontsize=15, fontweight="semibold", va="top")
    fig.text(0.06, 0.895, subtitle, color=theme["ink2"], fontsize=10, va="top")


def legend(fig, theme: dict, handles, y: float) -> None:
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, y), ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=theme["ink2"], handlelength=1.6,
               handletextpad=0.6, columnspacing=1.8, borderaxespad=0)


def save(fig, out_dir: Path, name: str, mode: str) -> list[Path]:
    paths = []
    for ext in ("svg", "png"):
        path = out_dir / f"{name}_{mode}.{ext}"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Chart 1: filing curves
# ---------------------------------------------------------------------------

def filing_curves(results: Path) -> list[tuple[str, pd.Series, pd.Series, list[int]]]:
    us = pd.read_csv(results / "us_lag_cdf.csv").set_index("day")
    kr = pd.read_csv(results / "kr_lag_cdf.csv").set_index("day")
    counts = (pd.read_csv(results / "kr_lag_summary.csv")
              .query("market == 'all'").set_index("report_type")["filings"])
    n_q, n_h = counts["quarterly"], counts["half-year"]
    kr_quarterly = (kr["quarterly | all"] * n_q + kr["half-year | all"] * n_h) / (n_q + n_h)
    return [
        ("Quarterly reports", us["10-Q | all"], kr_quarterly, [40, 45]),
        ("Annual reports", us["10-K | all"], kr["annual | all"], [60, 75, 90]),
    ]


def direct_label(ax, theme: dict, text: str, curves: pd.DataFrame, key: str, side: int,
                 deadlines: list[int]) -> None:
    """Name a curve beside it, at the first height where the label touches no line.

    side is -1 for a label left of the curve, +1 for right. Heights between the
    gridlines come first; the last resort sits on the 50% gridline, masked.
    """
    renderer = ax.figure.canvas.get_renderer()
    levels = (0.40, 0.60, 0.35, 0.65, 0.30, 0.70)
    for level in levels:
        x = int((curves[key] >= level).idxmax()) + 3 * side
        label = ax.text(x, level, text, color=theme["ink"], fontsize=9.5, va="center",
                        ha="left" if side > 0 else "right", zorder=4)
        box = label.get_window_extent(renderer).transformed(ax.transData.inverted())
        span = curves.loc[max(0, int(box.x0) - 1):int(box.x1) + 1]
        hits_curve = ((span >= box.y0 - 0.02) & (span <= box.y1 + 0.02)).to_numpy().any()
        hits_deadline = any(box.x0 - 1.5 <= day <= box.x1 + 1.5 for day in deadlines)
        if box.x0 >= 0 and box.x1 <= 120 and not hits_curve and not hits_deadline:
            return
        label.remove()
    x = int((curves[key] >= 0.5).idxmax()) + 3 * side
    ax.text(x, 0.5, text, color=theme["ink"], fontsize=9.5, va="center", ha="left" if side > 0 else "right",
            zorder=4, bbox=dict(boxstyle="square,pad=0.15", facecolor=theme["surface"], edgecolor="none"))


def draw_filing_curves(results: Path, out_dir: Path, mode: str) -> list[Path]:
    theme = THEMES[mode]
    fig = plt.figure(figsize=(10, 5.6), facecolor=theme["surface"])
    header(fig, theme, "How long companies take to file",
           "Share of reports filed by day N after the period ends. Grey lines mark legal deadlines.")
    legend(fig, theme, [Line2D([], [], color=theme["us"], lw=LINE_PT * 1.4, label="United States (10-Q, 10-K)"),
                        Line2D([], [], color=theme["kr"], lw=LINE_PT * 1.4, label="Korea (KOSPI, KOSDAQ)")], 0.835)
    panels = filing_curves(results)
    for i, (title, us, kr, deadlines) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.475, 0.16, 0.42, 0.55])
        style_axes(ax, theme, "y")
        days = us.index[us.index <= 120]
        for j, day in enumerate(deadlines):
            # Neighbouring deadlines a few days apart get their labels pushed to either side.
            close_left = j > 0 and day - deadlines[j - 1] < 10
            close_right = j + 1 < len(deadlines) and deadlines[j + 1] - day < 10
            align = "right" if close_right else "left" if close_left else "center"
            ax.axvline(day, color=theme["axis"], linewidth=0.75, zorder=1)
            ax.text(day, 1.03, f" {day} " if align != "center" else f"{day}", color=theme["muted"],
                    fontsize=8, ha=align, va="bottom", transform=ax.get_xaxis_transform())
        for series, key in ((us, "us"), (kr, "kr")):
            ax.plot(days, series.loc[days].to_numpy(), color=theme[key], lw=LINE_PT,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.0)
        # Direct labels beside each curve, on the side away from the other curve.
        curves = pd.DataFrame({"us": us.loc[days], "kr": kr.loc[days]})
        direct_label(ax, theme, "US", curves, "us", -1, deadlines)
        direct_label(ax, theme, "Korea", curves, "kr", +1, deadlines)
        ax.set_xticks([0, 30, 60, 90, 120])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"] if i == 0 else [])
        ax.set_xlabel("Days after the period ends", color=theme["ink2"], fontsize=9)
        ax.set_title(title, color=theme["ink"], fontsize=11, loc="left", pad=16)
    fig.text(0.06, 0.03, SOURCES, color=theme["muted"], fontsize=8)
    return save(fig, out_dir, "filing_curves", mode)


# ---------------------------------------------------------------------------
# Chart 2: leaks
# ---------------------------------------------------------------------------

def after_close_shares(results: Path) -> dict:
    sample = pd.read_csv(results / "kr_kind_timing_sample.csv")
    kind = sample["report"].str.split(" ").str[0]
    totals = sample.groupby(kind)[["reports_on_kind", "at_or_after_1530"]].sum()
    return (totals["at_or_after_1530"] / totals["reports_on_kind"]).to_dict()


def leak_comparison(results: Path) -> pd.DataFrame:
    us = pd.read_csv(results / "us_leak_table.csv").set_index(["form", "convention"])
    kr = pd.read_csv(results / "kr_leak_table.csv").set_index(["report_type", "convention"])
    share = after_close_shares(results)
    records = []
    for report, us_form, kr_types in (("quarterly", "10-Q", ["quarterly", "half-year"]),
                                      ("annual", "10-K", ["annual"])):
        for label, us_conv, kr_conv in LEAK_ROWS:
            parts = [kr.loc[(kind, kr_conv)] for kind in kr_types]
            weights = [row["filings"] for row in parts]
            next_day = sum(r["leaked_pct"] * w for r, w in zip(parts, weights)) / sum(weights)
            same_day = sum(r["leaked_pct_if_usable_same_day"] * w for r, w in zip(parts, weights)) / sum(weights)
            estimate = sum((r["leaked_pct_if_usable_same_day"] + share[kind] *
                            (r["leaked_pct"] - r["leaked_pct_if_usable_same_day"])) * w
                           for r, w, kind in zip(parts, weights, kr_types)) / sum(weights)
            records.append({
                "report": report, "rule": label.replace("\n", " "),
                "us_pct": round(float(us.loc[(us_form, us_conv), "leaked_pct"]), 1),
                "korea_estimate_pct": round(estimate, 1),
                "korea_if_next_day_pct": round(next_day, 1),
                "korea_if_same_day_pct": round(same_day, 1),
            })
    return pd.DataFrame(records)


def rounded_bar(ax, y: float, width: float, height: float, rx: float, ry: float, color: str) -> None:
    """A horizontal bar from x=0, square at the baseline and rounded at the data end."""
    if width <= 0:
        return
    rx, ry = min(rx, width), min(ry, height / 2)
    k = 0.5523  # Bezier circle constant
    x0, x1, y0, y1 = 0.0, width, y - height / 2, y + height / 2
    verts = [(x0, y0), (x1 - rx, y0),
             (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
             (x1, y1 - ry),
             (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
             (x0, y1), (x0, y0)]
    codes = ([MplPath.MOVETO, MplPath.LINETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO]
             + [MplPath.CURVE4] * 3 + [MplPath.LINETO, MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def draw_leaks(table: pd.DataFrame, shares: dict, out_dir: Path, mode: str) -> list[Path]:
    theme = THEMES[mode]
    fig_w, fig_h = 10.0, 5.2
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=theme["surface"])
    header(fig, theme, "How often a lag rule uses a report before it was public",
           "Share of reports a backtest would use too early, by rule. ≈ marks a Korean estimate.")
    legend(fig, theme, [Patch(color=theme["us"], label="United States"),
                        Patch(color=theme["kr"], label="Korea")], 0.835)
    rows = len(LEAK_ROWS)
    rect_w, rect_h, x_max = 0.30, 0.56, 118.0
    for i, report in enumerate(("quarterly", "annual")):
        left = 0.31 + i * 0.355
        ax = fig.add_axes([left, 0.16, rect_w, rect_h])
        style_axes(ax, theme, "x")
        ax.set_xlim(0, x_max)
        ax.set_ylim(rows - 0.5, -0.5)
        # Convert point sizes to data units for this axes (fixed figure and axes size).
        x_per_pt = x_max / (fig_w * 72 * rect_w)
        y_per_pt = rows / (fig_h * 72 * rect_h)
        h, gap = BAR_PT * y_per_pt, GAP_PT * y_per_pt
        rx, ry = RADIUS_PT * x_per_pt, RADIUS_PT * y_per_pt
        part = table[table["report"] == report].reset_index(drop=True)
        for row, record in part.iterrows():
            for value, key, offset, estimated in (
                    (record["us_pct"], "us", -(h + gap) / 2, False),
                    (record["korea_estimate_pct"], "kr", (h + gap) / 2,
                     abs(record["korea_if_next_day_pct"] - record["korea_if_same_day_pct"]) >= 1)):
                y = row + offset
                rounded_bar(ax, y, value, h, rx, ry, theme[key])
                shown = f"{value:.1f}" if 99.5 <= value < 100 else f"{math.floor(value + 0.5)}"  # half up: 60.5 -> 61
                label = f"{'≈' if estimated else ''}{shown}%"
                ax.text(value + 4 * x_per_pt, y, label, color=theme["ink2"], fontsize=8.5,
                        va="center", ha="left", zorder=4)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(rows))
        ax.set_yticklabels([label for label, _, _ in LEAK_ROWS] if i == 0 else [])
        if i == 0:
            ax.tick_params(axis="y", labelcolor=theme["ink"], labelsize=9.5)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(f"{report.capitalize()} reports", color=theme["ink"], fontsize=11, loc="left", pad=10)
    low, high = min(shares.values()) * 100, max(shares.values()) * 100
    share_text = f"{low:.0f}%" if round(low) == round(high) else f"{low:.0f}–{high:.0f}%"
    fig.text(0.06, 0.06, "Korean filings carry a date, not a time. Estimates apply the share of deadline-day "
             f"filings KIND shows arriving after the 15:30 close ({share_text}); bounds in leak_comparison.csv.",
             color=theme["muted"], fontsize=8)
    fig.text(0.06, 0.03, SOURCES, color=theme["muted"], fontsize=8)
    return save(fig, out_dir, "leaks", mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the article's charts from results/.")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=None, help="default: <results>/charts")
    args = parser.parse_args()
    out_dir = args.out or args.results / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    table = leak_comparison(args.results)
    table.to_csv(out_dir / "leak_comparison.csv", index=False)
    shares = after_close_shares(args.results)
    written = [out_dir / "leak_comparison.csv"]
    for mode in THEMES:
        written += draw_filing_curves(args.results, out_dir, mode)
        written += draw_leaks(table, shares, out_dir, mode)
    print("\n".join(str(path) for path in written))


if __name__ == "__main__":
    main()
