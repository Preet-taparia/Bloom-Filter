from __future__ import annotations

from collections.abc import Iterable

from bloom.hashing import MAX_SEED
from bloom.standard import BloomFilter
from bloom.theory import (
    combined_false_positive_rate,
    optimal_hash_count,
    optimal_size,
    split_error_budget,
)
from bloom.validation import require_int


class LayeredBloomFilter:
    """Several independently seeded Bloom filters that must all agree."""

    def __init__(
        self,
        size: int,
        hash_count: int,
        layer_count: int,
        seed: int = 0,
    ) -> None:
        require_int("layer_count", layer_count, minimum=1)
        require_int("seed", seed, minimum=0, maximum=MAX_SEED - layer_count)

        self._layers = [
            BloomFilter(size, hash_count, seed + offset)
            for offset in range(layer_count)
        ]

    @classmethod
    def from_capacity(
        cls,
        capacity: int,
        error_rate: float,
        layer_count: int,
        seed: int = 0,
    ) -> LayeredBloomFilter:
        """Split the error budget equally so the product meets error_rate."""
        layer_rate = split_error_budget(error_rate, layer_count)
        size = optimal_size(capacity, layer_rate)
        hash_count = optimal_hash_count(size, capacity)

        return cls(size, hash_count, layer_count, seed)

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    @property
    def layer_size(self) -> int:
        return self._layers[0].size

    @property
    def hash_count(self) -> int:
        return self._layers[0].hash_count

    @property
    def item_count(self) -> int:
        return self._layers[0].item_count

    @property
    def memory_bits(self) -> int:
        return sum(layer.memory_bits for layer in self._layers)

    @property
    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes for layer in self._layers)

    def add(self, item: object) -> None:
        for layer in self._layers:
            layer.add(item)

    def update(self, items: Iterable[object]) -> None:
        for item in items:
            self.add(item)

    def __contains__(self, item: object) -> bool:
        return all(item in layer for layer in self._layers)

    def __len__(self) -> int:
        return self.item_count

    def expected_false_positive_rate(self) -> float:
        return combined_false_positive_rate(
            layer.expected_false_positive_rate() for layer in self._layers
        )

    def __repr__(self) -> str:
        return (
            f"LayeredBloomFilter(layers={self.layer_count}, "
            f"layer_size={self.layer_size}, hash_count={self.hash_count})"
        )
