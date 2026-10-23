import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.Bloom.Theory import LN2, expected_false_positive_rate

from .FalsePositiveTrials import TrialResult, measure


@dataclass(frozen=True)
class OptimalKSummary:
    bits_per_item: float
    size: int
    item_count: int
    predicted_k_exact: float
    predicted_k_rounded: int
    theoretical_best_k: int
    empirical_best_k: int
    empirical_best_rate: float
    rate_at_predicted_k: float
    theoretical_best_rate: float


def sweep_size(
    sizes: Iterable[int],
    item_count: int,
    hash_count: int,
    query_count: int = 20000,
    trials: int = 5,
    base_seed: int = 0,
) -> list[TrialResult]:
    return [
        measure(size, hash_count, item_count, query_count, trials, base_seed)
        for size in sizes
    ]


def sweep_items(
    item_counts: Iterable[int],
    size: int,
    hash_count: int,
    query_count: int = 20000,
    trials: int = 5,
    base_seed: int = 0,
) -> list[TrialResult]:
    return [
        measure(size, hash_count, count, query_count, trials, base_seed)
        for count in item_counts
    ]


def sweep_hash_count(
    hash_counts: Iterable[int],
    size: int,
    item_count: int,
    query_count: int = 20000,
    trials: int = 5,
    base_seed: int = 0,
) -> list[TrialResult]:
    return [
        measure(size, count, item_count, query_count, trials, base_seed)
        for count in hash_counts
    ]


def theoretical_best_k(size: int, item_count: int, limit: int) -> int:
    return min(
        range(1, limit + 1),
        key=lambda k: expected_false_positive_rate(size, k, item_count),
    )


def optimal_k_study(
    ratios: Sequence[float],
    item_count: int,
    k_span: float = 3.0,
    query_count: int = 20000,
    trials: int = 5,
    base_seed: int = 0,
) -> tuple[list[TrialResult], list[OptimalKSummary]]:
    rows: list[TrialResult] = []
    summaries: list[OptimalKSummary] = []

    for ratio in ratios:
        size = round(ratio * item_count)
        predicted = LN2 * size / item_count
        limit = max(3, math.ceil(k_span * predicted))
        group = [
            measure(size, k, item_count, query_count, trials, base_seed)
            for k in range(1, limit + 1)
        ]
        best = min(group, key=lambda result: result.empirical_rate)
        rounded = max(1, round(predicted))
        at_predicted = next(r for r in group if r.hash_count == rounded)
        best_theory_k = theoretical_best_k(size, item_count, limit)

        rows.extend(group)
        summaries.append(
            OptimalKSummary(
                bits_per_item=size / item_count,
                size=size,
                item_count=item_count,
                predicted_k_exact=predicted,
                predicted_k_rounded=rounded,
                theoretical_best_k=best_theory_k,
                empirical_best_k=best.hash_count,
                empirical_best_rate=best.empirical_rate,
                rate_at_predicted_k=at_predicted.empirical_rate,
                theoretical_best_rate=expected_false_positive_rate(
                    size, best_theory_k, item_count
                ),
            )
        )

    return rows, summaries
