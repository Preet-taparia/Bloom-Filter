from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from FalsePositiveTrials import TrialResult
from ParameterStudies import OptimalKSummary


def _finish(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def plot_sweep(
    results: Sequence[TrialResult],
    attribute: str,
    xlabel: str,
    title: str,
    path: Path,
    logy: bool = True,
) -> Path:
    xs = [getattr(result, attribute) for result in results]
    figure, axis = plt.subplots(figsize=(7, 4.5))

    axis.errorbar(
        xs,
        [result.empirical_rate for result in results],
        yerr=[result.empirical_std for result in results],
        marker="o",
        capsize=3,
        label="empirical",
    )
    axis.plot(
        xs,
        [result.theoretical_rate for result in results],
        linestyle="--",
        label="theory",
    )
    axis.plot(
        xs,
        [result.observed_rate for result in results],
        linestyle=":",
        marker="x",
        label="fill ratio estimate",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("false positive rate")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()

    if logy:
        axis.set_yscale("log")

    return _finish(figure, path)


def plot_optimal_k(
    rows: Sequence[TrialResult],
    summaries: Sequence[OptimalKSummary],
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))

    for summary in summaries:
        group = [row for row in rows if row.size == summary.size]
        line = axis.plot(
            [row.hash_count for row in group],
            [row.empirical_rate for row in group],
            marker="o",
            markersize=3,
            label=f"m/n = {summary.bits_per_item:g}",
        )[0]
        axis.plot(
            [row.hash_count for row in group],
            [row.theoretical_rate for row in group],
            linestyle="--",
            color=line.get_color(),
            alpha=0.6,
        )
        axis.axvline(
            summary.predicted_k_exact,
            color=line.get_color(),
            linestyle=":",
            alpha=0.8,
        )

    axis.set_yscale("log")
    axis.set_xlabel("hash count k")
    axis.set_ylabel("false positive rate")
    axis.set_title("Empirical (solid), theory (dashed), predicted optimum (dotted)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    return _finish(figure, path)
