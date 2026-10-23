from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from Experiments.CountingStudies import (
    ChurnResult,
    CounterWidthResult,
    CountingComparison,
)
from Experiments.CrossCheckStudies import CrossCheckResult


def _finish(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def plot_counting_comparison(
    rows: Sequence[CountingComparison],
    path: Path,
) -> Path:
    figure, (rates, memory) = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = [row.bits_per_item for row in rows]
    positions = list(range(len(rows)))
    width = 0.4

    rates.plot(xs, [row.standard_rate for row in rows], marker="o", label="standard")
    rates.plot(
        xs,
        [row.counting_rate for row in rows],
        marker="s",
        linestyle="--",
        label="counting",
    )
    rates.plot(
        xs,
        [row.theoretical_rate for row in rows],
        linestyle=":",
        label="theory",
    )
    rates.set_yscale("log")
    rates.set_xlabel("slots per item m/n")
    rates.set_ylabel("false positive rate")
    rates.set_title("Equal slot count and hash count")
    rates.grid(True, alpha=0.3)
    rates.legend()

    memory.bar(
        [p - width / 2 for p in positions],
        [row.standard_memory_bytes for row in rows],
        width,
        label="standard",
    )
    memory.bar(
        [p + width / 2 for p in positions],
        [row.counting_memory_bytes for row in rows],
        width,
        label=f"counting ({rows[0].counter_bits} bit)",
    )
    memory.set_xticks(positions)
    memory.set_xticklabels([f"{x:g}" for x in xs])
    memory.set_xlabel("slots per item m/n")
    memory.set_ylabel("theoretical memory in bytes")
    memory.set_title("Memory cost of removal support")
    memory.grid(True, axis="y", alpha=0.3)
    memory.legend()

    return _finish(figure, path)


def plot_counter_widths(
    results: Sequence[CounterWidthResult],
    path: Path,
) -> Path:
    figure, (counts, fractions) = plt.subplots(1, 2, figsize=(11, 4.5))
    positions = list(range(len(results)))
    labels = [str(result.counter_bits) for result in results]
    width = 0.4

    counts.bar(
        [p - width / 2 for p in positions],
        [result.mean_saturated_counters for result in results],
        width,
        label="saturated counters",
    )
    counts.bar(
        [p + width / 2 for p in positions],
        [result.mean_overflow_events for result in results],
        width,
        label="dropped increments",
    )
    counts.set_yscale("symlog")
    counts.set_xticks(positions)
    counts.set_xticklabels(labels)
    counts.set_xlabel("counter width in bits")
    counts.set_ylabel("mean count per filter")
    counts.set_title("Saturation by counter width")
    counts.grid(True, axis="y", alpha=0.3)
    counts.legend()

    fractions.plot(
        positions,
        [result.overflow_trial_fraction for result in results],
        marker="o",
        label="observed fraction of trials",
    )
    fractions.plot(
        positions,
        [result.overflow_bound for result in results],
        linestyle="--",
        marker="x",
        label="Chernoff union bound",
    )
    fractions.set_xticks(positions)
    fractions.set_xticklabels(labels)
    fractions.set_ylim(-0.05, 1.05)
    fractions.set_xlabel("counter width in bits")
    fractions.set_ylabel("probability of any overflow")
    fractions.set_title("Observed overflow against the bound")
    fractions.grid(True, alpha=0.3)
    fractions.legend()

    return _finish(figure, path)


def plot_churn(results: Sequence[ChurnResult], path: Path) -> Path:
    figure, (fill, rate) = plt.subplots(1, 2, figsize=(11, 4.5))
    rounds = [result.round_number for result in results]
    misses = sum(result.counting_false_negatives for result in results)

    fill.plot(
        rounds,
        [result.counting_fill_ratio for result in results],
        marker="o",
        label="counting with removal",
    )
    fill.plot(
        rounds,
        [result.standard_fill_ratio for result in results],
        marker="s",
        linestyle="--",
        label="standard, never removes",
    )
    fill.set_xlabel("churn round")
    fill.set_ylabel("fill ratio")
    fill.set_title("Fill ratio under add and remove churn")
    fill.grid(True, alpha=0.3)
    fill.legend()

    rate.plot(
        rounds,
        [result.counting_false_positive_rate for result in results],
        marker="o",
        label="counting",
    )
    rate.plot(
        rounds,
        [result.standard_false_positive_rate for result in results],
        marker="s",
        linestyle="--",
        label="standard, never removes",
    )
    rate.plot(
        rounds,
        [result.stale_positive_rate for result in results],
        linestyle=":",
        label="removed items still present",
    )
    rate.set_yscale("log")
    rate.set_xlabel("churn round")
    rate.set_ylabel("false positive rate")
    rate.set_title(f"False positives (false negatives observed: {misses})")
    rate.grid(True, alpha=0.3)
    rate.legend()

    return _finish(figure, path)


def plot_cross_check(
    rows: Sequence[CrossCheckResult],
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))

    for error_rate in sorted({row.error_rate for row in rows}, reverse=True):
        group = [row for row in rows if row.error_rate == error_rate]
        counts = [row.filter_count for row in group]
        container = axis.errorbar(
            counts,
            [row.combined_rate for row in group],
            yerr=[row.combined_std for row in group],
            marker="o",
            capsize=3,
            label=f"cross checked, target {error_rate:g}",
        )
        color = container.lines[0].get_color()

        axis.plot(
            counts,
            [row.theoretical_rate for row in group],
            linestyle="--",
            color=color,
            alpha=0.7,
        )
        axis.plot(
            counts,
            [row.single_filter_rate for row in group],
            linestyle=":",
            marker="x",
            color=color,
            alpha=0.9,
        )

    axis.set_yscale("log")
    axis.set_xticks(sorted({row.filter_count for row in rows}))
    axis.set_xlabel("number of member filters")
    axis.set_ylabel("false positive rate")
    axis.set_title(
        "Cross checked (solid), theory (dashed), one filter same memory (dotted)"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    return _finish(figure, path)
