from Bloom.BaseBloomFilter import BloomBase
from Bloom.Theory import counter_overflow_bound
from Bloom.Validation import require_int


class CountingBloomFilter(BloomBase):
    """Bloom filter with saturating counters that supports removal.

    Each slot is a small counter instead of a single bit. Counters that reach
    their maximum are frozen and never decremented again, which protects
    against false negatives at the cost of a few permanently stuck slots.

    Only remove items that were previously added. Removing an item that was
    never inserted but happens to look present decrements counters owned by
    other items and can create false negatives for them.

    Args:
        size: Number of counters.
        hash_count: Number of counters touched per item.
        seed: Hash family selector.
        counter_bits: Width of each counter in bits, from 1 to 8.

    Raises:
        InvalidParameterError: If an argument is invalid.
        AllocationError: If the counters cannot be allocated.
    """

    def __init__(
        self,
        size: int,
        hash_count: int,
        seed: int = 0,
        counter_bits: int = 4,
    ) -> None:
        self._counter_bits = require_int(
            "counter_bits", counter_bits, minimum=1, maximum=8
        )
        self._max_value = (1 << self._counter_bits) - 1
        self._overflow_events = 0
        super().__init__(size, hash_count, seed)

    @property
    def slot_bits(self) -> int:
        return self._counter_bits

    @property
    def counter_bits(self) -> int:
        """Width of each counter in bits."""
        return self._counter_bits

    @property
    def overflow_events(self) -> int:
        """Number of increments dropped because a counter was saturated."""
        return self._overflow_events

    @property
    def saturated_counters(self) -> int:
        """Number of counters frozen at their maximum value."""
        return sum(1 for value in self._storage if value == self._max_value)

    @property
    def max_counter(self) -> int:
        """Largest value currently held by any counter."""
        return max(self._storage)

    def remove(self, item: object) -> bool:
        """Remove one occurrence of an item if it appears to be present.

        Saturated counters are skipped so earlier overflows cannot cause
        false negatives.

        Args:
            item: The item to remove.

        Returns:
            True if the item looked present and its counters were
            decremented, False if it was reported absent and nothing changed.

        Raises:
            InvalidItemError: If the item type is unsupported.
        """
        indexes = self._indexes(item)

        if not all(self._is_occupied(index) for index in indexes):
            return False

        for index in indexes:
            value = self._storage[index]

            if value == 0 or value == self._max_value:
                continue

            self._storage[index] = value - 1

            if value == 1:
                self._occupied -= 1

        self._item_count = max(self._item_count - 1, 0)

        return True

    def clear(self) -> None:
        super().clear()
        self._overflow_events = 0

    def overflow_probability_bound(self) -> float:
        """Upper bound on the chance that any counter overflows at this load."""
        return counter_overflow_bound(
            self._size,
            self._hash_count,
            self._item_count,
            self._counter_bits,
        )

    def _new_storage(self) -> bytearray:
        return bytearray(self._size)

    def _is_occupied(self, index: int) -> bool:
        return self._storage[index] != 0

    def _occupy(self, index: int) -> bool:
        value = self._storage[index]

        if value == self._max_value:
            self._overflow_events += 1
            return False

        self._storage[index] = value + 1

        return value == 0

    def _describe(self) -> str:
        return f"{super()._describe()}, counter_bits={self._counter_bits}"