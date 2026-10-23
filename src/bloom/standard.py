from .BaseBloomFilter import BloomBase


class BloomFilter(BloomBase):
    """Standard Bloom filter backed by a packed bit array.

    Items can be added but never removed. Membership tests may return false
    positives, never false negatives.

    Example:
        >>> filter = BloomFilter.from_capacity(1000, 0.01)
        >>> filter.add("apple")
        True
        >>> "apple" in filter
        True
    """

    @property
    def slot_bits(self) -> int:
        return 1

    @property
    def set_bit_count(self) -> int:
        """Number of bits currently set to one."""
        return self._occupied

    def _new_storage(self) -> bytearray:
        return bytearray((self._size + 7) // 8)

    def _is_occupied(self, index: int) -> bool:
        return bool(self._storage[index >> 3] & (1 << (index & 7)))

    def _occupy(self, index: int) -> bool:
        byte, mask = index >> 3, 1 << (index & 7)

        if self._storage[byte] & mask:
            return False

        self._storage[byte] |= mask

        return True
