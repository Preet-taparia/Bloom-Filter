import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from Bloom.Standard import BloomFilter
from Bloom.Theory import expected_false_positive_rate, optimal_hash_count
from .Workload import (
    CostModel,
    RemoteService,
    average_rows,
    make_urls,
    mixed_traffic,
)


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


def python_set_bytes(items: Iterable[str]) -> int:
    members = set(items)

    return sys.getsizeof(members) + sum(sys.getsizeof(item) for item in members)


def run_url_filter(
    bits_per_item: float,
    traffic: Sequence[tuple[str, bool]],
    blocklist: Sequence[str],
    cost: CostModel,
    seed: int = 0,
) -> UrlFilterResult:
    size = max(1, round(bits_per_item * len(blocklist)))
    hash_count = optimal_hash_count(size, len(blocklist))
    bloom = BloomFilter(size, hash_count, seed=seed)
    remote = RemoteService(blocklist, cost.network_ms)
    filter_ms = 0.0
    blocked = missed = flagged_benign = benign_requests = malicious_requests = 0

    bloom.update(blocklist)

    for url, is_malicious in traffic:
        filter_ms += cost.memory_ms

        if is_malicious:
            malicious_requests += 1
        else:
            benign_requests += 1

        if url in bloom:
            if remote.lookup(url):
                blocked += 1
            elif not is_malicious:
                flagged_benign += 1
        elif is_malicious:
            missed += 1

    distinct_benign = {url for url, is_malicious in traffic if not is_malicious}
    distinct_flagged = sum(1 for url in distinct_benign if url in bloom)
    baseline_ms = len(traffic) * cost.network_ms
    total_ms = remote.elapsed_ms + filter_ms
    set_bytes = python_set_bytes(blocklist)

    return UrlFilterResult(
        bits_per_item=bits_per_item,
        size=size,
        hash_count=hash_count,
        requests=len(traffic),
        malicious_requests=malicious_requests,
        blocked=blocked,
        missed=missed,
        remote_queries=remote.queries,
        remote_queries_avoided=len(traffic) - remote.queries,
        avoided_fraction=1.0 - remote.queries / len(traffic),
        false_positive_rate=flagged_benign / benign_requests if benign_requests else 0.0,
        distinct_false_positive_rate=distinct_flagged / len(distinct_benign)
        if distinct_benign
        else 0.0,
        theoretical_rate=expected_false_positive_rate(
            size, hash_count, len(blocklist)
        ),
        total_ms=total_ms,
        baseline_ms=baseline_ms,
        time_saved_fraction=1.0 - total_ms / baseline_ms,
        filter_bytes=bloom.memory_bytes,
        set_bytes=set_bytes,
        memory_ratio=set_bytes / bloom.memory_bytes,
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
) -> list[UrlFilterResult]:
    blocklist = make_urls("evil", blocklist_size)
    benign = make_urls("page", benign_count)
    collected: dict[float, list[UrlFilterResult]] = {ratio: [] for ratio in ratios}

    for trial in range(trials):
        seed = base_seed + trial
        traffic = mixed_traffic(
            benign, blocklist, requests, malicious_fraction, exponent, seed
        )

        for ratio in ratios:
            collected[ratio].append(
                run_url_filter(ratio, traffic, blocklist, cost, seed)
            )

    return [average_rows(collected[ratio]) for ratio in ratios]