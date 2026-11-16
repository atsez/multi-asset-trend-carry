"""Tables and charts. Every number in the write-up should trace to a file written here."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .stats import drawdown  # noqa: E402

# Validated palette (categorical slots 1-2, ordinal blue ramp), light surface
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.5, "legend.frameon": False, "figure.dpi": 110,
})


def _ts(idx):
    return idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK2)
    ax.plot([x], [y], "o", ms=3.5, color=color)


def _shade_test(ax, test_start, end):
    if test_start is not None and pd.Timestamp(test_start) < end:
        ax.axvspan(pd.Timestamp(test_start), end, color="#f0efec", zorder=0, lw=0)
        ax.text(pd.Timestamp(test_start), ax.get_ylim()[1], " locked test period", va="top",
                fontsize=8, color=INK2)


def cumulative(series: dict[str, pd.Series], path: Path, title: str, test_start: str | None = None,
               log: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for (name, x), col in zip(series.items(), [S1, S2]):
        nav = (1 + x.fillna(0)).cumprod()
        ax.plot(_ts(nav.index), nav.values, color=col, label=name)
        _end_label(ax, _ts(nav.index)[-1], nav.values[-1], name, col)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 (log scale)" if log else "Growth of 1")
    if len(series) > 1:
        ax.legend(loc="upper left")
    _shade_test(ax, test_start, max(_ts(x.index)[-1] for x in series.values()))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def drawdowns(series: dict[str, pd.Series], path: Path, title: str, colors=None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    colors = colors or [S1, S2]
    for (name, x), col in zip(series.items(), colors):
        dd = drawdown(x) * 100
        ax.plot(_ts(dd.index), dd.values, color=col, label=name)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    if len(series) > 1:
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rolling_sharpe(x: pd.Series, path: Path, window: int = 36) -> None:
    r = x.rolling(window).mean() / x.rolling(window).std() * np.sqrt(12)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.plot(_ts(r.index), r.values, color=S1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(f"Rolling {window}-month Sharpe ratio, net of costs")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bars(values: pd.Series, path: Path, title: str, fmt: str = "{:.1%}", xlabel: str = "") -> None:
    v = values.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(v) + 1.0))
    ax.barh(v.index.astype(str), v.values, color=S1, height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)
    span = np.nanmax(np.abs(v.values)) or 1
    for i, val in enumerate(v.values):
        ax.text(val + (0.02 * span if val >= 0 else -0.02 * span), i, fmt.format(val),
                va="center", ha="left" if val >= 0 else "right", fontsize=8.5, color=INK2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def grouped_bars(df: pd.DataFrame, path: Path, title: str) -> None:
    """Two series side by side (e.g. 60/40 versus trend in each crisis)."""
    labels = df.index.astype(str)
    y = np.arange(len(df))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(df) + 1.2))
    for i, (col, color) in enumerate(zip(df.columns, [S2, S1])):
        ax.barh(y + (i - 0.5) * (h + 0.03), df[col].values * 100, height=h, color=color, label=col)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Return over the episode (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_table(df: pd.DataFrame, path_stem: Path, floatfmt: str = ".3f") -> None:
    df.to_csv(path_stem.with_suffix(".csv"))
    try:
        md = df.to_markdown(floatfmt=floatfmt)
    except ImportError:  # tabulate not installed
        md = _plain_markdown(df, floatfmt)
    path_stem.with_suffix(".md").write_text(md + "\n")


def _plain_markdown(df: pd.DataFrame, floatfmt: str) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_json(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.Period, pd.Timestamp)):
            return str(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv))
