import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean

from Bloom.CountingBloomFilter import CountingBloomFilter
from Bloom.Standard import BloomFilter
from Bloom.Theory import optimal_hash_count
from Experiments.FalsePositiveTrials import measure, member_items, probe_items


@dataclass(frozen=True)
class CountingComparison:
    bits_per_item: float
    size: int
    hash_count: int
    item_count: int
    counter_bits: int
    standard_rate: float
    counting_rate: float
    theoretical_rate: float
    standard_memory_bytes: int
    counting_memory_bytes: int
    memory_ratio: float
    false_negatives: int


@dataclass(frozen=True)
class CounterWidthResult:
    counter_bits: int
    size: int
    hash_count: int
    item_count: int
    trials: int
    memory_bytes: int
    mean_overflow_events: float
    mean_saturated_counters: float
    mean_max_counter: float
    overflow_trial_fraction: float
    overflow_bound: float
    false_negatives: int


@dataclass(frozen=True)
class ChurnResult:
    round_number: int
    live_count: int
    counting_false_negatives: int
    stale_positive_rate: float
    counting_false_positive_rate: float
    counting_fill_ratio: float
    standard_false_positive_rate: float
    standard_fill_ratio: float
    saturated_counters: int


def compare_standard_counting(
    ratios: Sequence[float],
    item_count: int,
    counter_bits: int = 4,
    query_count: int = 20000,
    trials: int = 5,
    base_seed: int = 0,
) -> list[CountingComparison]:
    rows: list[CountingComparison] = []

    for ratio in ratios:
        size = round(ratio * item_count)
        hash_count = optimal_hash_count(size, item_count)
        standard = measure(
            size, hash_count, item_count, query_count, trials, base_seed, BloomFilter
        )
        counting = measure(
            size,
            hash_count,
            item_count,
            query_count,
            trials,
            base_seed,
            CountingBloomFilter,
            counter_bits=counter_bits,
        )
        standard_bytes = BloomFilter(size, hash_count).memory_bytes
        counting_bytes = CountingBloomFilter(
            size, hash_count, counter_bits=counter_bits
        ).memory_bytes

        rows.append(
            CountingComparison(
                bits_per_item=size / item_count,
                size=size,
                hash_count=hash_count,
                item_count=item_count,
                counter_bits=counter_bits,
                standard_rate=standard.empirical_rate,
                counting_rate=counting.empirical_rate,
                theoretical_rate=standard.theoretical_rate,
                standard_memory_bytes=standard_bytes,
                counting_memory_bytes=counting_bytes,
                memory_ratio=counting_bytes / standard_bytes,
                false_negatives=standard.false_negatives + counting.false_negatives,
            )
        )

    return rows


def counter_width_study(
    widths: Iterable[int],
    size: int,
    hash_count: int,
    item_count: int,
    trials: int = 5,
    base_seed: int = 0,
) -> list[CounterWidthResult]:
    results: list[CounterWidthResult] = []

    for bits in widths:
        events: list[int] = []
        saturated: list[int] = []
        maxima: list[int] = []
        overflowed = 0
        misses = 0
        bound = 0.0
        memory = 0

        for trial in range(trials):
            seed = base_seed + trial
            bloom = CountingBloomFilter(size, hash_count, seed=seed, counter_bits=bits)
            members = member_items(seed, item_count)

            bloom.update(members)

            events.append(bloom.overflow_events)
            saturated.append(bloom.saturated_counters)
            maxima.append(bloom.max_counter)
            overflowed += 1 if bloom.overflow_events > 0 else 0
            misses += sum(1 for item in members if item not in bloom)
            bound = bloom.overflow_probability_bound()
            memory = bloom.memory_bytes

        results.append(
            CounterWidthResult(
                counter_bits=bits,
                size=size,
                hash_count=hash_count,
                item_count=item_count,
                trials=trials,
                memory_bytes=memory,
                mean_overflow_events=mean(events),
                mean_saturated_counters=mean(saturated),
                mean_max_counter=mean(maxima),
                overflow_trial_fraction=overflowed / trials,
                overflow_bound=bound,
                false_negatives=misses,
            )
        )

    return results


def churn_study(
    size: int,
    hash_count: int,
    live_count: int,
    rounds: int,
    churn_fraction: float,
    probe_count: int,
    counter_bits: int = 4,
    seed: int = 0,
) -> list[ChurnResult]:
    rng = random.Random(seed)
    counting = CountingBloomFilter(
        size, hash_count, seed=seed, counter_bits=counter_bits
    )
    standard = BloomFilter(size, hash_count, seed=seed)
    live = [f"churn-{seed}-{i}" for i in range(live_count)]
    serial = live_count
    removed: list[str] = []
    probes = probe_items(seed, probe_count)
    swaps = max(1, round(churn_fraction * live_count))
    results: list[ChurnResult] = []

    counting.update(live)
    standard.update(live)

    for round_number in range(1, rounds + 1):
        for index in rng.sample(range(live_count), swaps):
            leaving = live[index]
            arriving = f"churn-{seed}-{serial}"
            serial += 1

            counting.remove(leaving)
            counting.add(arriving)
            standard.add(arriving)
            removed.append(leaving)
            live[index] = arriving

        recent = removed[-probe_count:]

        results.append(
            ChurnResult(
                round_number=round_number,
                live_count=live_count,
                counting_false_negatives=sum(
                    1 for item in live if item not in counting
                ),
                stale_positive_rate=sum(1 for item in recent if item in counting)
                / len(recent),
                counting_false_positive_rate=sum(
                    1 for item in probes if item in counting
                )
                / probe_count,
                counting_fill_ratio=counting.fill_ratio,
                standard_false_positive_rate=sum(
                    1 for item in probes if item in standard
                )
                / probe_count,
                standard_fill_ratio=standard.fill_ratio,
                saturated_counters=counting.saturated_counters,
            )
        )

    return results
