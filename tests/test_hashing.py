import unittest
import hashlib
import json
from pathlib import Path
from src.bloom.HashAlgorithms import murmur3, xxhash, fnv1a, md5, blake2b
from math import gcd
from src.bloom.hashing import ALGORITHMS, MAX_SEED, hash_pair, coprime_step, hash_indexes
from src.bloom.standard import BloomFilter
from src.bloom.CountingBloomFilter import CountingBloomFilter


class HashTests(unittest.TestCase):
    def test_vectors(self):
        self.assertEqual(murmur3(b''), 0)
        self.assertEqual(murmur3(b'foo'), 0xf6a5c420)
        self.assertEqual(murmur3(b'hello'), 0x248bfa47)
        self.assertEqual(xxhash(b''), 0x02cc5d05)
        self.assertEqual(xxhash(b'a'), 0x550d7456)
        self.assertEqual(xxhash(b'abc'), 0x32d153ff)
        self.assertEqual(fnv1a(b''), 0xcbf29ce484222325)
        self.assertEqual(fnv1a(b'hello'), 0xa430d84680aabd0b)

    def test_previous_implementation_compatibility(self):
        vectors = json.loads(Path(__file__).with_name('hash_compatibility_vectors.json').read_text())
        for case in vectors['cases']:
            data = bytes.fromhex(case['data'])
            with self.subTest(length=len(data)):
                for function in (murmur3, xxhash):
                    actual = [function(data, seed) for seed in (0, 1, -1, 2**32+7)]
                    self.assertEqual(actual, case[function.__name__])
                    self.assertTrue(all(type(value) is int and 0 <= value < 2**32 for value in actual))
                self.assertEqual(fnv1a(data), case['fnv1a'])
                self.assertTrue(type(fnv1a(data)) is int and 0 <= fnv1a(data) < 2**64)
                self.assertIsInstance(md5(data), bytes)
                self.assertEqual(md5(data).hex(), case['md5'])
                self.assertEqual([blake2b(data, 16, salt).hex() for salt in
                    (b'', b'12345678', bytes(range(16)))], case['blake2b'])
        for case in vectors['positions']:
            with self.subTest(algorithm=case['algorithm'], seed=case['seed']):
                self.assertEqual(list(hash_pair(b'scompatibility', case['seed'], case['algorithm'])), case['pair'])
                self.assertEqual(hash_indexes('compatibility', 120, 17, case['seed'], case['algorithm']), case['indices'])

    def test_registry(self):
        self.assertEqual(ALGORITHMS, ("murmur3", "xxhash", "fnv1a", "md5", "blake2b"))

    def test_seeded_murmur_vectors(self):
        # Published mmh3 test vectors: tests/test_mmh3.py in hajimes/mmh3.
        for data, seed, expected in (
            (b'', 1, 0x514e28b7), (b'', 0xffffffff, 0x81f16f39),
            (bytes.fromhex('21436587'), 0, 0xf55b516b),
            (bytes.fromhex('21436587'), 0x5082edee, 0x2362f9de),
            (bytes.fromhex('214365'), 0, 0x7e4a8634),
            (bytes.fromhex('2143'), 0, 0xa0f7b07a),
            (bytes.fromhex('21'), 0, 0x72661cf4),
            (b'a'*256, 0x9747b28c, 0x37405bdc),
        ):
            with self.subTest(data=data, seed=seed):
                self.assertEqual(murmur3(data, seed), expected)

    def test_xxhash_multiblock_vectors(self):
        # Official xxHash v0.8.2 cli/xsum_sanity_check.c XXH32 vectors.
        data = bytearray()
        state = 2654435761
        for _ in range(222):
            data.append(state >> 56)
            state = state * 11400714785074694797 & ((1 << 64)-1)
        for length, seed, expected in (
            (0, 2654435761, 0x36b78ae7),
            (1, 0, 0xcf65b03e), (1, 2654435761, 0xb4545aa4),
            (14, 0, 0x1208e7e2), (14, 2654435761, 0x6af1d1fe),
            (222, 0, 0x5bd11dbd), (222, 2654435761, 0x58803c5f),
        ):
            with self.subTest(length=length, seed=seed):
                self.assertEqual(xxhash(bytes(data[:length]), seed), expected)

    def test_digest_vectors(self):
        self.assertEqual(fnv1a(b'foobar'), 0x85944171f73967e8)
        self.assertEqual(md5(b'' ).hex(), 'd41d8cd98f00b204e9800998ecf8427e')
        self.assertEqual(md5(b'abc').hex(), '900150983cd24fb0d6963f7d28e17f72')
        self.assertEqual(blake2b(b'abc', 64).hex(),
            'ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1'
            '7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923')

    def test_index_edge_cases(self):
        seeds = (0, 1, 1 << 32, MAX_SEED)
        for size in (1, 2, 12, 17, 64):
            for raw in (0, 1, size, size-1, MAX_SEED):
                step = coprime_step(raw, size)
                self.assertEqual(gcd(step, size), 1)
                self.assertTrue(1 <= step <= max(1, size-1))
        for algorithm in ALGORITHMS:
            pairs = [hash_pair(b'test', seed, algorithm) for seed in seeds]
            self.assertEqual(len(set(pairs)), len(seeds))
            for seed in seeds:
                self.assertEqual(hash_pair(b'test', seed, algorithm), hash_pair(b'test', seed, algorithm))
                for size in (1, 2, 12, 17, 64):
                    for count in (1, size, size+3):
                        with self.subTest(algorithm=algorithm, seed=seed, size=size, count=count):
                            indices = hash_indexes('test', size, count, seed, algorithm)
                            self.assertEqual(indices, hash_indexes('test', size, count, seed, algorithm))
                            self.assertEqual(len(indices), count)
                            self.assertTrue(all(0 <= i < size for i in indices))
                            self.assertEqual(len(set(indices)), min(size, count))
                            for i in range(size, count):
                                self.assertEqual(indices[i], indices[i-size])
            for cls in (BloomFilter, CountingBloomFilter):
                tiny = cls(1, 3, hash_algorithm=algorithm)
                self.assertNotIn('item', tiny)
                tiny.add('item')
                self.assertIn('item', tiny)
        self.assertEqual(hash_indexes('test', 100, 7), hash_indexes('test', 100, 7, algorithm='blake2b'))
        # Preserve original BLAKE2b salt bytes and big-endian word splitting.
        for seed in seeds:
            digest = hashlib.blake2b(b'test', digest_size=16, salt=seed.to_bytes(8, 'big')).digest()
            self.assertEqual(hash_pair(b'test', seed, 'blake2b'),
                (int.from_bytes(digest[:8], 'big'), int.from_bytes(digest[8:], 'big')))
        with self.assertRaises(ValueError):
            hash_pair(b'test', 0, 'unknown')

    def test_digests_against_reference(self):
        # Empty, padding boundaries, full blocks and multi-block binary inputs.
        for length in (0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 255, 256, 257, 1000):
            data = bytes((i*37+11)%256 for i in range(length))
            with self.subTest(length=length):
                self.assertEqual(md5(data), hashlib.md5(data, usedforsecurity=False).digest())
                for size in (1, 16, 32, 64):
                    for salt in (b'', b'12345678', bytes(range(16))):
                        self.assertEqual(blake2b(data, size, salt), hashlib.blake2b(data, digest_size=size, salt=salt).digest())

    def test_filters(self):
        for algorithm in ALGORITHMS:
            for cls in (BloomFilter, CountingBloomFilter):
                with self.subTest(algorithm=algorithm, cls=cls):
                    bloom = cls.from_capacity(100, .01, hash_algorithm=algorithm)
                    bloom.update(range(100))
                    self.assertTrue(all(x in bloom for x in range(100)))
                    clone = bloom.copy()
                    bloom.clear()
                    self.assertTrue(all(x in clone for x in range(100)))
                    self.assertEqual(clone.hash_algorithm, algorithm)
                    if cls is CountingBloomFilter:
                        for x in range(50):
                            clone.remove(x)
                        self.assertTrue(all(x in clone for x in range(50, 100)))
            indices = hash_indexes('test', 100, 20, 2**64-1, algorithm)
            self.assertEqual(len(set(indices)), 20)
        with self.assertRaises(ValueError):
            BloomFilter(100, 3, hash_algorithm='invalid')

if __name__ == '__main__':
    unittest.main()
