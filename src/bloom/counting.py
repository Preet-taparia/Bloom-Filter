from __future__ import annotations

import struct
from collections.abc import Iterable

from bloom.exceptions import SerializationError
from bloom.hashing import MAX_SEED, hash_indexes
from bloom.theory import (
    counter_overflow_bound,
    expected_false_positive_rate,
    optimal_hash_count,
    optimal_size,
)
from bloom.validation import require_int

MAGIC = b"BLMC"
HEADER = struct.Struct(">4sQIQQBQ")


class CountingBloomFilter:
    """Bloom filter with saturating counters that supports deletion."""

    def __init__(
        self,
        size: int,
        hash_count: int,
        counter_bits: int = 4,
        seed: int = 0,
    ) -> None:
        self._size = require_int("size", size, minimum=1)
        self._hash_count = require_int("hash_count", hash_count, minimum=1)
        self._counter_bits = require_int(
            "counter_bits", counter_bits, minimum=1, maximum=8
        )
        self._seed = require_int("seed", seed, minimum=0, maximum=MAX_SEED)
        self._max_value = (1 << counter_bits) - 1
        self._counters = bytearray(size)
        self._nonzero = 0
        self._item_count = 0
        self._overflow_events = 0

    @classmethod
    def from_capacity(
        cls,
        capacity: int,
        error_rate: float,
        counter_bits: int = 4,
        seed: int = 0,
    ) -> CountingBloomFilter:
        size = optimal_size(capacity, error_rate)
        hash_count = optimal_hash_count(size, capacity)

        return cls(size, hash_count, counter_bits, seed)

    @property
    def size(self) -> int:
        return self._size

    @property
    def hash_count(self) -> int:
        return self._hash_count

    @property
    def counter_bits(self) -> int:
        return self._counter_bits

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def item_count(self) -> int:
        return self._item_count

    @property
    def overflow_events(self) -> int:
        return self._overflow_events

    @property
    def saturated_counters(self) -> int:
        return sum(1 for value in self._counters if value == self._max_value)

    @property
    def max_counter(self) -> int:
        return max(self._counters)

    @property
    def fill_ratio(self) -> float:
        return self._nonzero / self._size

    @property
    def memory_bits(self) -> int:
        return self._size * self._counter_bits

    @property
    def memory_bytes(self) -> int:
        return (self.memory_bits + 7) // 8

    def add(self, item: object) -> None:
        for index in hash_indexes(item, self._size, self._hash_count, self._seed):
            value = self._counters[index]

            if value == self._max_value:
                self._overflow_events += 1
                continue

            if value == 0:
                self._nonzero += 1

            self._counters[index] = value + 1

        self._item_count += 1

    def update(self, items: Iterable[object]) -> None:
        for item in items:
            self.add(item)

    def remove(self, item: object) -> bool:
        """Remove an item if it appears present and return whether it was removed.

        Saturated counters are never decremented, which preserves the no false
        negative guarantee at the cost of a few permanently stuck counters.
        """
        if item not in self:
            return False

        for index in hash_indexes(item, self._size, self._hash_count, self._seed):
            value = self._counters[index]

            if value == self._max_value or value == 0:
                continue

            self._counters[index] = value - 1

            if value == 1:
                self._nonzero -= 1

        self._item_count -= 1

        return True

    def __contains__(self, item: object) -> bool:
        for index in hash_indexes(item, self._size, self._hash_count, self._seed):
            if self._counters[index] == 0:
                return False

        return True

    def __len__(self) -> int:
        return self._item_count

    def clear(self) -> None:
        self._counters = bytearray(self._size)
        self._nonzero = 0
        self._item_count = 0
        self._overflow_events = 0

    def expected_false_positive_rate(self) -> float:
        return expected_false_positive_rate(
            self._size, self._hash_count, max(self._item_count, 0)
        )

    def overflow_probability_bound(self) -> float:
        return counter_overflow_bound(
            self._size,
            self._hash_count,
            max(self._item_count, 0),
            self._counter_bits,
        )

    def to_bytes(self) -> bytes:
        header = HEADER.pack(
            MAGIC,
            self._size,
            self._hash_count,
            self._seed,
            max(self._item_count, 0),
            self._counter_bits,
            self._overflow_events,
        )

        return header + bytes(self._counters)

    @classmethod
    def from_bytes(cls, data: bytes) -> CountingBloomFilter:
        if len(data) < HEADER.size:
            raise SerializationError("data is too short")

        magic, size, hash_count, seed, item_count, bits, overflow = HEADER.unpack_from(
            data
        )

        if magic != MAGIC:
            raise SerializationError("data is not a counting Bloom filter")

        payload = data[HEADER.size :]

        if len(payload) != size:
            raise SerializationError("payload length does not match size")

        try:
            result = cls(size, hash_count, bits, seed)
        except ValueError as exc:
            raise SerializationError(str(exc)) from exc

        if payload and max(payload) > result._max_value:
            raise SerializationError("counter exceeds the configured width")

        result._counters = bytearray(payload)
        result._nonzero = sum(1 for value in payload if value)
        result._item_count = item_count
        result._overflow_events = overflow

        return result

    def __repr__(self) -> str:
        return (
            f"CountingBloomFilter(size={self._size}, hash_count={self._hash_count}, "
            f"counter_bits={self._counter_bits}, items={self._item_count})"
        )
