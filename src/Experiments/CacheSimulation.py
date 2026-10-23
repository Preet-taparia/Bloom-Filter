from collections.abc import Sequence
from dataclasses import dataclass, replace

from Bloom.BaseBloomFilter import BloomBase
from Bloom.CountingBloomFilter import CountingBloomFilter
from Bloom.Standard import BloomFilter
from Bloom.Theory import optimal_hash_count
from Bloom.Validation import require_choice, require_int
from .Workload import (
    CostModel,
    LruDiskCache,
    RemoteService,
    ZipfSampler,
    average_rows,
    make_urls,
)

ACTIVE_STRATEGIES = ("standard", "rebuild", "counting")
STRATEGIES = ("none",) + ACTIVE_STRATEGIES


@dataclass(frozen=True)
class CacheResult:
    strategy: str
    bits_per_item: float
    size: int
    hash_count: int
    requests: int
    cache_capacity: int
    hit_rate: float
    disk_lookups: int
    disk_lookups_avoided: int
    wasted_disk_lookups: int
    wasted_per_miss: float
    false_negatives: int
    origin_fetches: int
    total_ms: float
    time_saved_fraction: float
    filter_memory_bytes: int
    final_fill_ratio: float


@dataclass(frozen=True)
class CacheTimelinePoint:
    strategy: str
    bits_per_item: float
    window: int
    requests_end: int
    wasted_per_miss: float
    fill_ratio: float


def _build_filter(
    strategy: str,
    size: int,
    hash_count: int,
    seed: int,
    counter_bits: int,
) -> BloomBase:
    if strategy == "counting":
        return CountingBloomFilter(
            size, hash_count, seed=seed, counter_bits=counter_bits
        )

    return BloomFilter(size, hash_count, seed=seed)


def run_cache(
    strategy: str,
    traffic: Sequence[str],
    capacity: int,
    bits_per_item: float,
    cost: CostModel,
    counter_bits: int = 4,
    rebuild_interval: int = 0,
    window: int = 5000,
    seed: int = 0,
) -> tuple[CacheResult, list[CacheTimelinePoint]]:
    require_choice("strategy", strategy, STRATEGIES)
    require_int("capacity", capacity, minimum=1)
    require_int("window", window, minimum=1)
    require_int("rebuild_interval", rebuild_interval, minimum=0)

    size = max(1, round(bits_per_item * capacity))
    hash_count = optimal_hash_count(size, capacity)
    tracker = (
        None
        if strategy == "none"
        else _build_filter(strategy, size, hash_count, seed, counter_bits)
    )
    disk = LruDiskCache(capacity, cost.disk_ms)
    origin = RemoteService((), cost.network_ms)
    filter_ms = 0.0
    hits = avoided = wasted = misses = false_negatives = 0
    window_misses = window_wasted = 0
    timeline: list[CacheTimelinePoint] = []

    for position, url in enumerate(traffic, start=1):
        resident = disk.contains(url)
        probe = True

        if tracker is not None:
            filter_ms += cost.memory_ms
            probe = url in tracker

        served_from_disk = False

        if probe:
            disk.lookup(url)
            served_from_disk = resident

            if not resident and tracker is not None:
                wasted += 1
                window_wasted += 1
        elif resident:
            false_negatives += 1
        else:
            avoided += 1

        if not resident:
            misses += 1
            window_misses += 1

        if served_from_disk:
            hits += 1
        else:
            origin.fetch(url)

            if not resident:
                evicted = disk.insert(url)

                if tracker is not None:
                    tracker.add(url)

                    if strategy == "counting" and evicted is not None:
                        tracker.remove(evicted)

        if (
            strategy == "rebuild"
            and rebuild_interval
            and position % rebuild_interval == 0
        ):
            tracker.clear()
            tracker.update(disk.keys())

        if position % window == 0:
            timeline.append(
                CacheTimelinePoint(
                    strategy=strategy,
                    bits_per_item=bits_per_item,
                    window=position // window,
                    requests_end=position,
                    wasted_per_miss=window_wasted / window_misses
                    if window_misses
                    else 0.0,
                    fill_ratio=tracker.fill_ratio if tracker is not None else 0.0,
                )
            )
            window_misses = window_wasted = 0

    result = CacheResult(
        strategy=strategy,
        bits_per_item=bits_per_item,
        size=size if tracker is not None else 0,
        hash_count=hash_count if tracker is not None else 0,
        requests=len(traffic),
        cache_capacity=capacity,
        hit_rate=hits / len(traffic),
        disk_lookups=disk.lookups,
        disk_lookups_avoided=avoided,
        wasted_disk_lookups=wasted,
        wasted_per_miss=wasted / misses if misses else 0.0,
        false_negatives=false_negatives,
        origin_fetches=origin.queries,
        total_ms=disk.elapsed_ms + origin.elapsed_ms + filter_ms,
        time_saved_fraction=0.0,
        filter_memory_bytes=tracker.memory_bytes if tracker is not None else 0,
        final_fill_ratio=tracker.fill_ratio if tracker is not None else 0.0,
    )

    return result, timeline


def simulate_cache(
    ratios: Sequence[float],
    strategies: Sequence[str],
    universe: int,
    capacity: int,
    requests: int,
    exponent: float = 1.0,
    trials: int = 3,
    cost: CostModel = CostModel(),
    counter_bits: int = 4,
    rebuild_interval: int = 5000,
    window: int = 5000,
    base_seed: int = 0,
) -> tuple[list[CacheResult], list[CacheTimelinePoint]]:
    for strategy in strategies:
        require_choice("strategy", strategy, ACTIVE_STRATEGIES)

    urls = make_urls("page", universe)
    baselines: list[CacheResult] = []
    collected: dict[tuple[str, float], list[CacheResult]] = {}
    timeline: list[CacheTimelinePoint] = []

    for trial in range(trials):
        seed = base_seed + trial
        traffic = ZipfSampler(urls, exponent, seed).draw(requests)
        baseline, _ = run_cache("none", traffic, capacity, 0.0, cost, window=window)

        baselines.append(baseline)

        for ratio in ratios:
            for strategy in strategies:
                result, points = run_cache(
                    strategy,
                    traffic,
                    capacity,
                    ratio,
                    cost,
                    counter_bits,
                    rebuild_interval,
                    window,
                    seed,
                )
                saved = 1.0 - result.total_ms / baseline.total_ms

                collected.setdefault((strategy, ratio), []).append(
                    replace(result, time_saved_fraction=saved)
                )

                if trial == 0:
                    timeline.extend(points)

    results: list[CacheResult] = [average_rows(baselines)]

    for ratio in ratios:
        for strategy in strategies:
            results.append(average_rows(collected[(strategy, ratio)]))

    return results, timeline