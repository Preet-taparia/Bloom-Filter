import random
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from itertools import accumulate
from statistics import mean

from Bloom.Validation import require_int, require_non_negative_float


@dataclass(frozen=True)
class CostModel:
    memory_ms: float = 0.0005
    disk_ms: float = 5.0
    network_ms: float = 80.0

    def __post_init__(self) -> None:
        require_non_negative_float("memory_ms", self.memory_ms)
        require_non_negative_float("disk_ms", self.disk_ms)
        require_non_negative_float("network_ms", self.network_ms)


class ZipfSampler:
    def __init__(self, population: Sequence[str], exponent: float, seed: int) -> None:
        if not population:
            raise ValueError("population must not be empty")

        exponent = require_non_negative_float("exponent", exponent)
        self._population = list(population)
        self._rng = random.Random(seed)
        self._cumulative = list(
            accumulate(1.0 / (rank + 1) ** exponent for rank in range(len(population)))
        )

    def draw(self, count: int) -> list[str]:
        require_int("count", count, minimum=0)

        return self._rng.choices(
            self._population, cum_weights=self._cumulative, k=count
        )


class RemoteService:
    def __init__(self, contents: Iterable[str], latency_ms: float) -> None:
        self._contents = frozenset(contents)
        self._latency_ms = latency_ms
        self.queries = 0
        self.elapsed_ms = 0.0

    def lookup(self, url: str) -> bool:
        self.queries += 1
        self.elapsed_ms += self._latency_ms

        return url in self._contents

    def fetch(self, url: str) -> None:
        self.queries += 1
        self.elapsed_ms += self._latency_ms


class LruDiskCache:
    def __init__(self, capacity: int, latency_ms: float) -> None:
        self._capacity = require_int("capacity", capacity, minimum=1)
        self._latency_ms = latency_ms
        self._entries: OrderedDict[str, None] = OrderedDict()
        self.lookups = 0
        self.elapsed_ms = 0.0

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, url: str) -> bool:
        return url in self._entries

    def keys(self) -> list[str]:
        return list(self._entries)

    def lookup(self, url: str) -> bool:
        self.lookups += 1
        self.elapsed_ms += self._latency_ms

        if url in self._entries:
            self._entries.move_to_end(url)

            return True

        return False

    def insert(self, url: str) -> str | None:
        self._entries[url] = None
        self._entries.move_to_end(url)

        if len(self._entries) > self._capacity:
            evicted, _ = self._entries.popitem(last=False)

            return evicted

        return None


def make_urls(prefix: str, count: int) -> list[str]:
    require_int("count", count, minimum=1)

    return [f"https://{prefix}{i % 997}.example.com/{prefix}/{i}" for i in range(count)]


def mixed_traffic(
    benign: Sequence[str],
    malicious: Sequence[str],
    requests: int,
    malicious_fraction: float,
    exponent: float,
    seed: int,
) -> list[tuple[str, bool]]:
    rng = random.Random(seed)
    benign_draws = ZipfSampler(benign, exponent, seed + 1).draw(requests)
    malicious_draws = ZipfSampler(malicious, exponent, seed + 2).draw(requests)

    return [
        (malicious_draws[i], True)
        if rng.random() < malicious_fraction
        else (benign_draws[i], False)
        for i in range(requests)
    ]


def average_rows(rows: Sequence[object]) -> object:
    first = rows[0]
    merged: dict[str, object] = {}

    for field in fields(first):
        column = [getattr(row, field.name) for row in rows]

        if isinstance(column[0], (int, float)) and not isinstance(column[0], bool):
            merged[field.name] = mean(column)
        else:
            merged[field.name] = column[0]

    return type(first)(**merged)