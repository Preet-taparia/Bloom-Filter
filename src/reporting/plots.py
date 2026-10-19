from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def line_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: dict[str, tuple[list[float], list[float], str]],
    log_y: bool = False,
) -> None:
    """Draw several series, each given as xs, ys and a style of line or points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4.5))

    for label, (xs, ys, style) in series.items():
        if style == "points":
            axes.plot(xs, ys, marker="o", linestyle="none", label=label)
        else:
            axes.plot(xs, ys, linestyle="-", label=label)

    if log_y:
        axes.set_yscale("log")

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def bar_plot(
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    values: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4.5))
    bars = axes.bar(labels, values)
    axes.bar_label(bars, fmt="%.0f")
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
