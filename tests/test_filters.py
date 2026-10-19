import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bloom.counting import CountingBloomFilter
from bloom.exceptions import (
    IncompatibleFilterError,
    InvalidParameterError,
    SerializationError,
    UnsupportedOperationError,
)
from bloom.layered import LayeredBloomFilter
from bloom.standard import BloomFilter
from bloom.theory import optimal_hash_count, optimal_size, split_error_budget


class StandardFilterTests(unittest.TestCase):
    def test_no_false_negatives(self):
        flt = BloomFilter.from_capacity(500, 0.01)
        items = [f"item{index}" for index in range(500)]
        flt.update(items)
        self.assertTrue(all(item in flt for item in items))

    def test_false_positive_rate_near_target(self):
        flt = BloomFilter.from_capacity(2000, 0.02)
        flt.update(f"in{index}" for index in range(2000))
        hits = sum(1 for index in range(20000) if f"out{index}" in flt)
        self.assertLess(hits / 20000, 0.04)

    def test_remove_is_unsupported(self):
        with self.assertRaises(UnsupportedOperationError):
            BloomFilter(100, 3).remove("a")

    def test_invalid_parameters(self):
        with self.assertRaises(InvalidParameterError):
            BloomFilter(0, 3)
        with self.assertRaises(InvalidParameterError):
            BloomFilter(10, True)
        with self.assertRaises(InvalidParameterError):
            BloomFilter(10, 3).add(1.5)

    def test_union(self):
        left, right = BloomFilter(2000, 4), BloomFilter(2000, 4)
        left.update(["a", "b"])
        right.update(["c"])
        merged = left.union(right)
        self.assertTrue(all(key in merged for key in "abc"))
        with self.assertRaises(IncompatibleFilterError):
            left.union(BloomFilter(2001, 4))

    def test_serialization_round_trip(self):
        flt = BloomFilter(1000, 3, seed=5)
        flt.update(range(50))
        clone = BloomFilter.from_bytes(flt.to_bytes())
        self.assertEqual(clone.to_bytes(), flt.to_bytes())
        self.assertTrue(all(index in clone for index in range(50)))
        with self.assertRaises(SerializationError):
            BloomFilter.from_bytes(b"junk")


class CountingFilterTests(unittest.TestCase):
    def test_remove_and_membership(self):
        flt = CountingBloomFilter.from_capacity(200, 0.01)
        flt.update(range(200))
        self.assertTrue(flt.remove(7))
        self.assertNotIn(7, flt)
        self.assertFalse(flt.remove(7))
        self.assertTrue(all(index in flt for index in range(200) if index != 7))

    def test_no_false_negatives_after_removals(self):
        flt = CountingBloomFilter.from_capacity(1000, 0.01)
        flt.update(range(1000))
        for index in range(0, 1000, 2):
            flt.remove(index)
        self.assertTrue(all(index in flt for index in range(1, 1000, 2)))

    def test_saturation_is_sticky(self):
        flt = CountingBloomFilter(4, 1, counter_bits=1)
        for _ in range(3):
            flt.add("x")
        self.assertGreater(flt.overflow_events, 0)
        flt.remove("x")
        self.assertIn("x", flt)

    def test_memory_overhead(self):
        flt = CountingBloomFilter(1000, 3, counter_bits=4)
        self.assertEqual(flt.memory_bits, 4000)

    def test_serialization_round_trip(self):
        flt = CountingBloomFilter(500, 3, 4, seed=2)
        flt.update(range(40))
        clone = CountingBloomFilter.from_bytes(flt.to_bytes())
        self.assertEqual(clone.to_bytes(), flt.to_bytes())


class LayeredFilterTests(unittest.TestCase):
    def test_membership(self):
        flt = LayeredBloomFilter.from_capacity(300, 0.01, 2)
        flt.update(range(300))
        self.assertTrue(all(index in flt for index in range(300)))

    def test_expected_rate_matches_budget(self):
        flt = LayeredBloomFilter.from_capacity(1000, 0.01, 3)
        flt.update(range(1000))
        self.assertAlmostEqual(flt.expected_false_positive_rate(), 0.01, delta=0.003)


class TheoryTests(unittest.TestCase):
    def test_optimal_values(self):
        size = optimal_size(1000, 0.01)
        self.assertEqual(optimal_hash_count(size, 1000), 7)
        self.assertAlmostEqual(split_error_budget(0.01, 2), 0.1)


if __name__ == "__main__":
    unittest.main()
