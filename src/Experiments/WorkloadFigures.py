from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .CacheSimulation import CacheResult, CacheTimelinePoint
from .UrlFilterSimulation import UrlFilterResult


def _finish(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def plot_cache_results(
    results: Sequence[CacheResult],
    timeline: Sequence[CacheTimelinePoint],
    timeline_ratio: float,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    saved, wasted, memory, drift = axes.flatten()
    strategies = list(dict.fromkeys(r.strategy for r in results if r.strategy != "none"))

    for strategy in strategies:
        group = [r for r in results if r.strategy == strategy]
        xs = [r.bits_per_item for r in group]

        saved.plot(xs, [r.time_saved_fraction for r in group], marker="o", label=strategy)
        wasted.plot(xs, [r.wasted_per_miss for r in group], marker="o", label=strategy)
        memory.plot(xs, [r.filter_memory_bytes for r in group], marker="o", label=strategy)

    saved.set_xlabel("slots per cached item m/n")
    saved.set_ylabel("fraction of total latency saved")
    saved.set_title("Latency saved against no filter")
    saved.grid(True, alpha=0.3)
    saved.legend()

    wasted.set_yscale("log")
    wasted.set_xlabel("slots per cached item m/n")
    wasted.set_ylabel("wasted disk lookups per cache miss")
    wasted.set_title("Filter false positives in the cache")
    wasted.grid(True, alpha=0.3)
    wasted.legend()

    memory.set_xlabel("slots per cached item m/n")
    memory.set_ylabel("filter memory in bytes")
    memory.set_title("Memory cost of each strategy")
    memory.grid(True, alpha=0.3)
    memory.legend()

    for strategy in strategies:
        points = [
            p
            for p in timeline
            if p.strategy == strategy and p.bits_per_item == timeline_ratio
        ]

        drift.plot(
            [p.requests_end for p in points],
            [p.wasted_per_miss for p in points],
            marker="o",
            label=strategy,
        )

    drift.set_xlabel("requests processed")
    drift.set_ylabel("wasted disk lookups per cache miss")
    drift.set_title(f"Drift over time at m/n = {timeline_ratio:g}")
    drift.grid(True, alpha=0.3)
    drift.legend()

    return _finish(figure, path)


def plot_url_filter_results(
    results: Sequence[UrlFilterResult],
    path: Path,
) -> Path:
    figure, (rates, savings, memory) = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = [r.bits_per_item for r in results]
    positions = list(range(len(results)))
    width = 0.4

    rates.plot(xs, [r.false_positive_rate for r in results], marker="o", label="per request")
    rates.plot(
        xs,
        [r.distinct_false_positive_rate for r in results],
        marker="s",
        label="per distinct URL",
    )
    rates.plot(xs, [r.theoretical_rate for r in results], linestyle="--", label="theory")
    rates.set_yscale("log")
    rates.set_xlabel("slots per blocklist URL m/n")
    rates.set_ylabel("false positive rate")
    rates.set_title("False positives on benign traffic")
    rates.grid(True, alpha=0.3)
    rates.legend()

    savings.plot(
        xs,
        [r.avoided_fraction for r in results],
        marker="o",
        label="remote queries avoided",
    )
    savings.plot(
        xs,
        [r.time_saved_fraction for r in results],
        marker="s",
        linestyle="--",
        label="latency saved",
    )
    savings.set_xlabel("slots per blocklist URL m/n")
    savings.set_ylabel("fraction")
    savings.set_title("Savings against querying the service every time")
    savings.grid(True, alpha=0.3)
    savings.legend()

    memory.bar(
        [p - width / 2 for p in positions],
        [r.filter_bytes for r in results],
        width,
        label="Bloom filter",
    )
    memory.bar(
        [p + width / 2 for p in positions],
        [r.set_bytes for r in results],
        width,
        label="Python set",
    )
    memory.set_yscale("log")
    memory.set_xticks(positions)
    memory.set_xticklabels([f"{x:g}" for x in xs])
    memory.set_xlabel("slots per blocklist URL m/n")
    memory.set_ylabel("memory in bytes")
    memory.set_title("Space against an exact set")
    memory.grid(True, axis="y", alpha=0.3)
    memory.legend()

    return _finish(figure, path)