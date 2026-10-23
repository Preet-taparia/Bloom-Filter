from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, stdev

from Bloom.BaseBloomFilter import BloomBase
from Bloom.CountingBloomFilter import CountingBloomFilter
from Bloom.CrossCheckBloomFilter import CrossCheckedBloomFilter
from Bloom.Standard import BloomFilter
from Bloom.Theory import expected_false_positive_rate, optimal_hash_count
from Bloom.Validation import require_choice
from Experiments.FalsePositiveTrials import member_items, probe_items, run_trial

MEMBER_KINDS = ("standard", "counting", "mixed")
SEED_STRIDE = 100


@dataclass(frozen=True)
class CrossCheckResult:
    kind: str
    filter_count: int
    error_rate: float
    item_count: int
    query_count: int
    trials: int
    memory_bytes: int
    combined_rate: float
    combined_std: float
    theoretical_rate: float
    mean_member_rate: float
    disagreement_rate: float
    catch_rate: float
    single_filter_rate: float
    single_filter_theory: float
    false_negatives: int


def member_types_for(kind: str, count: int) -> tuple[type[BloomBase], ...]:
    require_choice("kind", kind, MEMBER_KINDS)

    if kind == "standard":
        return (BloomFilter,) * count

    if kind == "counting":
        return (CountingBloomFilter,) * count

    return tuple(BloomFilter if i % 2 == 0 else CountingBloomFilter for i in range(count))


def _spread(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def cross_check_point(
    kind: str,
    filter_count: int,
    error_rate: float,
    item_count: int,
    query_count: int = 50000,
    trials: int = 5,
    base_seed: int = 0,
) -> CrossCheckResult:
    combined: list[float] = []
    member_rates: list[float] = []
    single: list[float] = []
    accepted_total = 0
    disagreement_total = 0
    misses = 0
    memory_bits = 0
    memory_bytes = 0
    theory = 0.0
    single_theory = 0.0

    for trial in range(trials):
        seed = base_seed + trial * SEED_STRIDE
        checker = CrossCheckedBloomFilter.from_capacity(
            item_count,
            error_rate,
            seed=seed,
            filter_count=filter_count,
            member_types=member_types_for(kind, filter_count),
        )
        members = member_items(seed, item_count)

        checker.update(members)

        misses += sum(1 for item in members if item not in checker)

        accepted = 0
        member_hits = [0] * filter_count

        for item in probe_items(seed, query_count):
            verdicts = checker.check(item)
            accepted += 1 if all(verdicts) else 0

            for index, verdict in enumerate(verdicts):
                member_hits[index] += 1 if verdict else 0

        accepted_total += accepted
        disagreement_total += checker.caught_false_positives
        combined.append(accepted / query_count)
        member_rates.append(mean(hits / query_count for hits in member_hits))
        memory_bits = checker.memory_bits
        memory_bytes = checker.memory_bytes
        theory = checker.expected_false_positive_rate()

        single_size = memory_bits
        single_hashes = optimal_hash_count(single_size, item_count)
        rate, _, single_misses = run_trial(
            single_size, single_hashes, item_count, query_count, seed
        )
        single.append(rate)
        misses += single_misses
        single_theory = expected_false_positive_rate(
            single_size, single_hashes, item_count
        )

    exposed = disagreement_total + accepted_total
    total_queries = query_count * trials

    return CrossCheckResult(
        kind=kind,
        filter_count=filter_count,
        error_rate=error_rate,
        item_count=item_count,
        query_count=query_count,
        trials=trials,
        memory_bytes=memory_bytes,
        combined_rate=mean(combined),
        combined_std=_spread(combined),
        theoretical_rate=theory,
        mean_member_rate=mean(member_rates),
        disagreement_rate=disagreement_total / total_queries,
        catch_rate=disagreement_total / exposed if exposed else 1.0,
        single_filter_rate=mean(single),
        single_filter_theory=single_theory,
        false_negatives=misses,
    )


def cross_check_study(
    kind: str,
    filter_counts: Sequence[int],
    error_rates: Sequence[float],
    item_count: int,
    query_count: int = 50000,
    trials: int = 5,
    base_seed: int = 0,
) -> list[CrossCheckResult]:
    return [
        cross_check_point(
            kind, count, rate, item_count, query_count, trials, base_seed
        )
        for rate in error_rates
        for count in filter_counts
    ]