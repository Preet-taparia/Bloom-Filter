from __future__ import annotations

import struct
from collections.abc import Iterable

from bloom.exceptions import (
    IncompatibleFilterError,
    SerializationError,
    UnsupportedOperationError,
)
from bloom.hashing import MAX_SEED, hash_indexes
from bloom.theory import (
    estimate_item_count,
    expected_false_positive_rate,
    optimal_hash_count,
    optimal_size,
)
from bloom.validation import require_fraction, require_int

MAGIC = b"BLMS"
HEADER = struct.Struct(">4sQIQQ")


class BloomFilter:
    """Standard Bloom filter backed by a packed bit array."""

    def __init__(self, size: int, hash_count: int, seed: int = 0) -> None:
        self._size = require_int("size", size, minimum=1)
        self._hash_count = require_int("hash_count", hash_count, minimum=1)
        self._seed = require_int("seed", seed, minimum=0, maximum=MAX_SEED)
        self._bits = bytearray((size + 7) // 8)
        self._set_bits = 0
        self._item_count = 0

    @classmethod
    def from_capacity(
        cls,
        capacity: int,
        error_rate: float,
        seed: int = 0,
    ) -> BloomFilter:
        """Build a filter sized for a capacity and target false positive rate."""
        size = optimal_size(capacity, error_rate)
        hash_count = optimal_hash_count(size, capacity)

        return cls(size, hash_count, seed)

    @property
    def size(self) -> int:
        return self._size

    @property
    def hash_count(self) -> int:
        return self._hash_count

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def item_count(self) -> int:
        return self._item_count

    @property
    def set_bit_count(self) -> int:
        return self._set_bits

    @property
    def fill_ratio(self) -> float:
        return self._set_bits / self._size

    @property
    def memory_bits(self) -> int:
        return self._size

    @property
    def memory_bytes(self) -> int:
        return len(self._bits)

    def add(self, item: object) -> bool:
        """Insert an item and return True if any bit changed.

        Every call counts as one insertion toward item_count.
        """
        changed = False

        for index in hash_indexes(item, self._size, self._hash_count, self._seed):
            byte, mask = index >> 3, 1 << (index & 7)

            if not self._bits[byte] & mask:
                self._bits[byte] |= mask
                self._set_bits += 1
                changed = True

        self._item_count += 1

        return changed

    def update(self, items: Iterable[object]) -> None:
        for item in items:
            self.add(item)

    def remove(self, item: object) -> bool:
        raise UnsupportedOperationError(
            "a standard Bloom filter cannot remove items, use CountingBloomFilter"
        )

    def __contains__(self, item: object) -> bool:
        for index in hash_indexes(item, self._size, self._hash_count, self._seed):
            if not self._bits[index >> 3] & (1 << (index & 7)):
                return False

        return True

    def __len__(self) -> int:
        return self._item_count

    def clear(self) -> None:
        self._bits = bytearray(len(self._bits))
        self._set_bits = 0
        self._item_count = 0

    def copy(self) -> BloomFilter:
        clone = BloomFilter(self._size, self._hash_count, self._seed)
        clone._bits = bytearray(self._bits)
        clone._set_bits = self._set_bits
        clone._item_count = self._item_count

        return clone

    def expected_false_positive_rate(self) -> float:
        """Theoretical false positive rate for the current item count."""
        return expected_false_positive_rate(
            self._size, self._hash_count, self._item_count
        )

    def observed_false_positive_rate(self) -> float:
        """False positive rate implied by the measured fill ratio."""
        return self.fill_ratio**self._hash_count

    def estimated_item_count(self) -> float:
        """Estimate distinct inserted items from the measured fill ratio."""
        return estimate_item_count(
            self._size,
            self._hash_count,
            require_fraction("fill_ratio", self.fill_ratio),
        )

    def _require_compatible(self, other: BloomFilter) -> None:
        if not isinstance(other, BloomFilter):
            raise IncompatibleFilterError("other is not a BloomFilter")

        if (
            other._size != self._size
            or other._hash_count != self._hash_count
            or other._seed != self._seed
        ):
            raise IncompatibleFilterError(
                "filters must share size, hash count and seed"
            )

    def union(self, other: BloomFilter) -> BloomFilter:
        """Return a filter representing the union of two compatible filters."""
        self._require_compatible(other)
        merged = self.copy()

        for position, value in enumerate(other._bits):
            merged._bits[position] |= value

        merged._set_bits = sum(byte.bit_count() for byte in merged._bits)
        merged._item_count = round(merged.estimated_item_count())

        return merged

    def to_bytes(self) -> bytes:
        header = HEADER.pack(
            MAGIC, self._size, self._hash_count, self._seed, self._item_count
        )

        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomFilter:
        if len(data) < HEADER.size:
            raise SerializationError("data is too short")

        magic, size, hash_count, seed, item_count = HEADER.unpack_from(data)

        if magic != MAGIC:
            raise SerializationError("data is not a standard Bloom filter")

        payload = data[HEADER.size :]

        if len(payload) != (size + 7) // 8:
            raise SerializationError("payload length does not match size")

        try:
            result = cls(size, hash_count, seed)
        except ValueError as exc:
            raise SerializationError(str(exc)) from exc

        result._bits = bytearray(payload)
        result._set_bits = sum(byte.bit_count() for byte in payload)
        result._item_count = item_count

        return result

    def __repr__(self) -> str:
        return (
            f"BloomFilter(size={self._size}, hash_count={self._hash_count}, "
            f"items={self._item_count}, fill={self.fill_ratio:.3f})"
        )
