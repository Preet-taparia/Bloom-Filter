import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Self

from .Exceptions import AllocationError
from .Hashing import MAX_SEED, hash_indexes
from .Theory import (
    estimate_item_count,
    expected_false_positive_rate,
    optimal_hash_count,
    optimal_size,
)
from .Validation import require_int


class BloomBase(ABC):
    """Shared implementation for filters built on a fixed array of slots.

    Answers "possibly present" or "definitely absent". Items that were added
    and not removed are never reported absent, but absent items may be
    reported present at a predictable rate.

    Subclasses decide how a slot is stored by implementing four hooks:
    slot_bits, _new_storage, _is_occupied and _occupy. Hashing, sizing,
    statistics, copying and clearing live here once.

    Args:
        size: Number of slots in the array.
        hash_count: Number of positions touched per item.
        seed: Value from 0 to 2 to the power of 64 minus 1 selecting the
            hash family. Filters with different seeds are independent.

    Raises:
        ParameterTypeError: If an argument is not an integer.
        ParameterValueError: If an argument is out of range.
        AllocationError: If the storage cannot be allocated.
    """

    def __init__(self, size: int, hash_count: int, seed: int = 0) -> None:
        self._size = require_int("size", size, minimum=1)
        self._hash_count = require_int("hash_count", hash_count, minimum=1)
        self._seed = require_int("seed", seed, minimum=0, maximum=MAX_SEED)
        self._occupied = 0
        self._item_count = 0
        self._storage = self._allocate()

    @classmethod
    def from_capacity(
        cls,
        capacity: int,
        error_rate: float,
        seed: int = 0,
        **options: Any,
    ) -> Self:
        """Build a filter sized for a capacity and a target error rate.

        Args:
            capacity: Number of distinct items the filter should hold.
            error_rate: Target false positive probability, strictly
                between 0 and 1.
            seed: Hash family selector.
            **options: Extra constructor arguments for subclasses, such as
                counter_bits for a counting filter.

        Raises:
            InvalidParameterError: If capacity, error_rate or seed is invalid.
            AllocationError: If the required storage cannot be allocated.
        """
        size = optimal_size(capacity, error_rate)
        hash_count = optimal_hash_count(size, capacity)

        return cls(size, hash_count, seed=seed, **options)

    @property
    @abstractmethod
    def slot_bits(self) -> int:
        """Number of bits one slot logically occupies."""

    @property
    def size(self) -> int:
        """Number of slots in the array."""
        return self._size

    @property
    def hash_count(self) -> int:
        """Number of positions touched per item."""
        return self._hash_count

    @property
    def seed(self) -> int:
        """Hash family selector."""
        return self._seed

    @property
    def item_count(self) -> int:
        """Number of insertions, counting repeated items each time."""
        return self._item_count

    @property
    def occupied_count(self) -> int:
        """Number of slots currently occupied."""
        return self._occupied

    @property
    def fill_ratio(self) -> float:
        """Fraction of slots currently occupied."""
        return self._occupied / self._size

    @property
    def memory_bits(self) -> int:
        """Theoretical storage requirement in bits."""
        return self._size * self.slot_bits

    @property
    def memory_bytes(self) -> int:
        """Theoretical storage requirement in whole bytes."""
        return (self.memory_bits + 7) // 8

    def add(self, item: object) -> bool:
        """Insert an item.

        Returns:
            True if at least one slot became occupied. False means the item
            was already reported as present.
        """
        changed = False

        for index in self._indexes(item):
            if self._occupy(index):
                self._occupied += 1
                changed = True

        self._item_count += 1

        return changed

    def update(self, items: Iterable[object]) -> None:
        """Insert every item from an iterable."""
        for item in items:
            self.add(item)

    def clear(self) -> None:
        """Remove every item and reset all statistics."""
        self._storage = self._allocate()
        self._occupied = 0
        self._item_count = 0

    def copy(self) -> Self:
        """Return an independent copy that shares no mutable state."""
        clone = copy.copy(self)
        clone._storage = bytearray(self._storage)

        return clone

    def expected_false_positive_rate(self) -> float:
        """Theoretical false positive rate for the recorded insertion count.

        Repeated insertions of the same item count separately, so this is an
        upper estimate when duplicates were added.
        """
        return expected_false_positive_rate(
            self._size, self._hash_count, self._item_count
        )

    def observed_false_positive_rate(self) -> float:
        """False positive rate implied by the measured fill ratio."""
        return self.fill_ratio**self._hash_count

    def estimated_item_count(self) -> float:
        """Estimate the number of distinct inserted items from the fill ratio."""
        return estimate_item_count(self._size, self._hash_count, self.fill_ratio)

    def __contains__(self, item: object) -> bool:
        return all(self._is_occupied(index) for index in self._indexes(item))

    def __len__(self) -> int:
        return self._item_count

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._describe()})"

    @abstractmethod
    def _new_storage(self) -> bytearray:
        """Return freshly zeroed backing storage for size slots."""

    @abstractmethod
    def _is_occupied(self, index: int) -> bool:
        """Return True if the slot at index is occupied."""

    @abstractmethod
    def _occupy(self, index: int) -> bool:
        """Mark the slot at index and return True if it was previously empty."""

    def _allocate(self) -> bytearray:
        try:
            return self._new_storage()
        except (MemoryError, OverflowError) as exc:
            raise AllocationError(
                f"cannot allocate storage for {self._size} slots"
            ) from exc

    def _indexes(self, item: object) -> list[int]:
        return hash_indexes(item, self._size, self._hash_count, self._seed)

    def _describe(self) -> str:
        return (
            f"size={self._size}, hash_count={self._hash_count}, "
            f"items={self._item_count}, fill={self.fill_ratio:.3f}"
        )
