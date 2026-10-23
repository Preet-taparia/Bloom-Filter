import math
from dataclasses import dataclass
from statistics import mean, pstdev

from src.Bloom.BaseBloomFilter import BloomBase
from src.Bloom.Standard import BloomFilter
from src.Bloom.Theory import expected_false_positive_rate


@dataclass(frozen=True)
class TrialResult:
    size: int
    hash_count: int
    item_count: int
    query_count: int
    trials: int
    bits_per_item: float
    empirical_rate: float
    empirical_std: float
    observed_rate: float
    theoretical_rate: float
    false_negatives: int


def member_items(trial_seed: int, count: int) -> list[str]:
    return [f"member-{trial_seed}-{i}" for i in range(count)]


def probe_items(trial_seed: int, count: int) -> list[str]:
    return [f"probe-{trial_seed}-{i}" for i in range(count)]


def run_trial(
    size: int,
    hash_count: int,
    item_count: int,
    query_count: int,
    seed: int,
    filter_type: type[BloomBase] = BloomFilter,
    **options: object,
) -> tuple[float, float, int]:
    bloom = filter_type(size, hash_count, seed=seed, **options)
    members = member_items(seed, item_count)

    bloom.update(members)

    false_negatives = sum(1 for item in members if item not in bloom)
    false_positives = sum(1 for item in probe_items(seed, query_count) if item in bloom)

    return (
        false_positives / query_count,
        bloom.observed_false_positive_rate(),
        false_negatives,
    )


def measure(
    size: int,
    hash_count: int,
    item_count: int,
    query_count: int = 20000,
    trials: int = 5,
    base_seed: int = 0,
    filter_type: type[BloomBase] = BloomFilter,
    **options: object,
) -> TrialResult:
    empirical: list[float] = []
    observed: list[float] = []
    false_negatives = 0

    for trial in range(trials):
        rate, fill_rate, misses = run_trial(
            size,
            hash_count,
            item_count,
            query_count,
            base_seed + trial,
            filter_type,
            **options,
        )
        empirical.append(rate)
        observed.append(fill_rate)
        false_negatives += misses

    return TrialResult(
        size=size,
        hash_count=hash_count,
        item_count=item_count,
        query_count=query_count,
        trials=trials,
        bits_per_item=size / item_count,
        empirical_rate=mean(empirical),
        empirical_std=pstdev(empirical),
        observed_rate=mean(observed),
        theoretical_rate=expected_false_positive_rate(size, hash_count, item_count),
        false_negatives=false_negatives,
    )


def relative_error(result: TrialResult) -> float:
    if result.theoretical_rate == 0.0:
        return math.nan

    return (result.empirical_rate - result.theoretical_rate) / result.theoretical_rate
