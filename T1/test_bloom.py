import unittest

from bloom import (
    BloomFilter,
    CountingBloomFilter,
    HashFamily,
    optimal_k,
    optimal_m,
    theoretical_fpr,
)


def urls(n, prefix="http://a.example.com/p/"):
    return [f"{prefix}{i}" for i in range(n)]


class HashTests(unittest.TestCase):
    def test_indexes_in_range_and_deterministic(self):
        for scheme in HashFamily.SCHEMES:
            h = HashFamily(1000, 7, scheme)
            a = h.indexes("x")
            self.assertEqual(a, h.indexes("x"))
            self.assertEqual(len(a), 7)
            self.assertTrue(all(0 <= i < 1000 for i in a))

    def test_unknown_scheme(self):
        with self.assertRaises(ValueError):
            HashFamily(10, 2, "nope")


class BloomTests(unittest.TestCase):
    def test_no_false_negatives(self):
        for scheme in HashFamily.SCHEMES:
            bf = BloomFilter.for_capacity(5000, 0.01, scheme)
            items = urls(5000)
            for u in items:
                bf.add(u)
            self.assertTrue(all(u in bf for u in items))

    def test_false_positive_rate_close_to_theory(self):
        n = 5000
        bf = BloomFilter.for_capacity(n, 0.02)
        for u in urls(n):
            bf.add(u)
        probes = urls(40000, "http://b.example.com/q/")
        observed = sum(1 for u in probes if u in bf) / len(probes)
        expected = theoretical_fpr(bf.m, n, bf.k)
        self.assertLess(abs(observed - expected), 0.01)

    def test_parameter_formulas(self):
        m = optimal_m(1000, 0.01)
        self.assertEqual(optimal_k(m, 1000), 7)
        self.assertAlmostEqual(m / 1000, 9.585, delta=0.01)


class CountingTests(unittest.TestCase):
    def test_add_remove_roundtrip(self):
        cbf = CountingBloomFilter.for_capacity(2000, 0.01)
        items = urls(2000)
        for u in items:
            cbf.add(u)
        for u in items[:1000]:
            self.assertTrue(cbf.remove(u))
        self.assertTrue(all(u in cbf for u in items[1000:]))
        removed_present = sum(1 for u in items[:1000] if u in cbf)
        self.assertLess(removed_present, 60)

    def test_full_removal_empties_filter(self):
        cbf = CountingBloomFilter.for_capacity(500, 0.01)
        items = urls(500)
        for u in items:
            cbf.add(u)
        for u in items:
            cbf.remove(u)
        self.assertEqual(cbf.nonzero_ratio(), 0.0)

    def test_duplicate_insert_needs_duplicate_remove(self):
        cbf = CountingBloomFilter(1000, 4)
        cbf.add("u")
        cbf.add("u")
        cbf.remove("u")
        self.assertIn("u", cbf)
        cbf.remove("u")
        self.assertNotIn("u", cbf)

    def test_guard_refuses_absent_item(self):
        cbf = CountingBloomFilter(100000, 5)
        self.assertFalse(cbf.remove("never added"))
        self.assertEqual(cbf.underflow_refusals, 1)

    def test_sticky_saturation_prevents_false_negatives(self):
        cbf = CountingBloomFilter(4000, 3, counter_bits=2, sticky=True)
        items = urls(2000)
        for u in items:
            cbf.add(u)
        self.assertGreater(cbf.overflow_events, 0)
        for u in items[:1000]:
            cbf.remove(u)
        self.assertTrue(all(u in cbf for u in items[1000:]))

    def test_packed_widths_agree(self):
        for bits in (1, 2, 4, 8):
            cbf = CountingBloomFilter(257, 3, counter_bits=bits)
            for i in range(257):
                cbf._set(i, i % (cbf.max_value + 1))
            for i in range(257):
                self.assertEqual(cbf._get(i), i % (cbf.max_value + 1))

    def test_to_bloom_matches_membership(self):
        cbf = CountingBloomFilter.for_capacity(500, 0.01)
        items = urls(500)
        for u in items:
            cbf.add(u)
        bf = cbf.to_bloom()
        self.assertTrue(all(u in bf for u in items))


if __name__ == "__main__":
    unittest.main()
