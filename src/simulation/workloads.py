import random
from itertools import accumulate


def member_keys(count: int, namespace: int = 0) -> list[str]:
    return [f"http://site{namespace}.example.com/member/{index}" for index in range(count)]


def probe_keys(count: int, namespace: int = 0) -> list[str]:
    return [f"http://site{namespace}.example.com/probe/{index}" for index in range(count)]


def url_universe(count: int) -> list[str]:
    return [f"http://cache.example.com/page/{index}" for index in range(count)]


class ZipfSampler:
    """Draw items with probability proportional to 1 / rank ** exponent."""

    def __init__(self, items: list[str], exponent: float, rng: random.Random) -> None:
        self._items = items
        self._rng = rng
        weights = [1.0 / (rank**exponent) for rank in range(1, len(items) + 1)]
        self._cumulative = list(accumulate(weights))

    def sample(self, count: int) -> list[str]:
        return self._rng.choices(self._items, cum_weights=self._cumulative, k=count)
