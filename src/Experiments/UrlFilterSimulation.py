import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean

from ..bloom.standard import BloomFilter
from ..bloom.theory import expected_false_positive_rate, optimal_hash_count
from ..bloom.validation import require_int

from .Workload import (
    CostModel,
    average_rows,
    make_urls,
    mixed_traffic,
)

SEED_STRIDE = 1000


@dataclass(frozen=True)
class UrlFilterResult:
    bits_per_item: float
    size: int
    hash_count: int
    requests: int
    malicious_requests: int
    blocked: int
    missed: int
    remote_queries: int
    remote_queries_avoided: int
    avoided_fraction: float
    false_positive_rate: float
    distinct_false_positive_rate: float
    theoretical_rate: float
    total_ms: float
    baseline_ms: float
    time_saved_fraction: float
    filter_bytes: int
    set_bytes: int
    memory_ratio: float


@dataclass(frozen=True)
class FilterOutcome:
    flagged_requests: int
    flagged_distinct: int
    missed: int
    memory_bytes: int


def python_set_bytes(items: Iterable[str]) -> int:
    members = set(items)

    return sys.getsizeof(members) + sum(sys.getsizeof(item) for item in members)


def evaluate_filter(
    size: int,
    hash_count: int,
    filter_seed: int,
    blocklist: Sequence[str],
    benign_counts: Counter[str],
    malicious_counts: Counter[str],
) -> FilterOutcome:
    bloom = BloomFilter(size, hash_count, seed=filter_seed)

    bloom.update(blocklist)

    flagged_requests = 0
    flagged_distinct = 0

    for url, count in benign_counts.items():
        if url in bloom:
            flagged_requests += count
            flagged_distinct += 1

    missed = sum(count for url, count in malicious_counts.items() if url not in bloom)

    return FilterOutcome(
        flagged_requests=flagged_requests,
        flagged_distinct=flagged_distinct,
        missed=missed,
        memory_bytes=bloom.memory_bytes,
    )


def build_result(
    bits_per_item: float,
    size: int,
    hash_count: int,
    outcomes: Sequence[FilterOutcome],
    requests: int,
    benign_requests: int,
    malicious_requests: int,
    distinct_benign: int,
    blocklist_size: int,
    set_bytes: int,
    cost: CostModel,
) -> UrlFilterResult:
    flagged_requests = mean(o.flagged_requests for o in outcomes)
    flagged_distinct = mean(o.flagged_distinct for o in outcomes)
    missed = mean(o.missed for o in outcomes)
    filter_bytes = outcomes[0].memory_bytes
    blocked = malicious_requests - missed
    remote_queries = blocked + flagged_requests
    baseline_ms = requests * cost.network_ms
    total_ms = remote_queries * cost.network_ms + requests * cost.memory_ms

    return UrlFilterResult(
        bits_per_item=bits_per_item,
        size=size,
        hash_count=hash_count,
        requests=requests,
        malicious_requests=malicious_requests,
        blocked=blocked,
        missed=missed,
        remote_queries=remote_queries,
        remote_queries_avoided=requests - remote_queries,
        avoided_fraction=1.0 - remote_queries / requests,
        false_positive_rate=flagged_requests / benign_requests
        if benign_requests
        else 0.0,
        distinct_false_positive_rate=flagged_distinct / distinct_benign
        if distinct_benign
        else 0.0,
        theoretical_rate=expected_false_positive_rate(size, hash_count, blocklist_size),
        total_ms=total_ms,
        baseline_ms=baseline_ms,
        time_saved_fraction=1.0 - total_ms / baseline_ms,
        filter_bytes=filter_bytes,
        set_bytes=set_bytes,
        memory_ratio=set_bytes / filter_bytes,
    )


def simulate_url_filter(
    ratios: Sequence[float],
    blocklist_size: int,
    benign_count: int,
    requests: int,
    malicious_fraction: float = 0.02,
    exponent: float = 1.0,
    trials: int = 3,
    cost: CostModel = CostModel(),
    base_seed: int = 0,
    filter_seeds: int = 10,
) -> list[UrlFilterResult]:
    require_int("filter_seeds", filter_seeds, minimum=1)

    blocklist = make_urls("evil", blocklist_size)
    benign = make_urls("page", benign_count)
    set_bytes = python_set_bytes(blocklist)
    shapes = {
        ratio: (
            max(1, round(ratio * len(blocklist))),
            optimal_hash_count(max(1, round(ratio * len(blocklist))), len(blocklist)),
        )
        for ratio in ratios
    }
    collected: dict[float, list[UrlFilterResult]] = {ratio: [] for ratio in ratios}

    for trial in range(trials):
        seed = base_seed + trial
        traffic = mixed_traffic(
            benign, blocklist, requests, malicious_fraction, exponent, seed
        )
        benign_counts = Counter(
            url for url, is_malicious in traffic if not is_malicious
        )
        malicious_counts = Counter(url for url, is_malicious in traffic if is_malicious)
        benign_requests = sum(benign_counts.values())
        malicious_requests = sum(malicious_counts.values())
        outcomes: dict[float, list[FilterOutcome]] = {ratio: [] for ratio in ratios}

        for offset in range(filter_seeds):
            filter_seed = seed + offset * SEED_STRIDE

            for ratio in ratios:
                size, hash_count = shapes[ratio]

                outcomes[ratio].append(
                    evaluate_filter(
                        size,
                        hash_count,
                        filter_seed,
                        blocklist,
                        benign_counts,
                        malicious_counts,
                    )
                )

        for ratio in ratios:
            size, hash_count = shapes[ratio]

            collected[ratio].append(
                build_result(
                    ratio,
                    size,
                    hash_count,
                    outcomes[ratio],
                    len(traffic),
                    benign_requests,
                    malicious_requests,
                    len(benign_counts),
                    len(blocklist),
                    set_bytes,
                    cost,
                )
            )

    return [average_rows(collected[ratio]) for ratio in ratios]
