import hashlib
import math


def optimal_m(n, p):
    return math.ceil(-n * math.log(p) / (math.log(2) ** 2))


def optimal_k(m, n):
    return max(1, round(m / n * math.log(2)))


def theoretical_fpr(m, n, k):
    return (1.0 - math.exp(-k * n / m)) ** k


def exact_fpr(m, n, k):
    return (1.0 - (1.0 - 1.0 / m) ** (k * n)) ** k


class HashFamily:
    SCHEMES = ("double", "salted")

    def __init__(self, m, k, scheme="double"):
        if scheme not in self.SCHEMES:
            raise ValueError("unknown scheme")
        self.m = m
        self.k = k
        self.scheme = scheme

    def indexes(self, item):
        data = item if isinstance(item, bytes) else str(item).encode("utf8")
        m, k = self.m, self.k
        if self.scheme == "double":
            digest = hashlib.blake2b(data, digest_size=16).digest()
            h1 = int.from_bytes(digest[:8], "little")
            h2 = int.from_bytes(digest[8:], "little") | 1
            return [(h1 + i * h2) % m for i in range(k)]
        out = []
        for i in range(k):
            digest = hashlib.blake2b(
                data, digest_size=8, salt=i.to_bytes(8, "little")
            ).digest()
            out.append(int.from_bytes(digest, "little") % m)
        return out


class BloomFilter:
    def __init__(self, m, k, scheme="double"):
        self.m = m
        self.k = k
        self.hashes = HashFamily(m, k, scheme)
        self.bits = bytearray((m + 7) // 8)
        self.count = 0

    @classmethod
    def for_capacity(cls, n, p, scheme="double"):
        m = optimal_m(n, p)
        return cls(m, optimal_k(m, n), scheme)

    def add(self, item):
        for i in self.hashes.indexes(item):
            self.bits[i >> 3] |= 1 << (i & 7)
        self.count += 1

    def __contains__(self, item):
        bits = self.bits
        for i in self.hashes.indexes(item):
            if not bits[i >> 3] & (1 << (i & 7)):
                return False
        return True

    def fill_ratio(self):
        return sum(bin(b).count("1") for b in self.bits) / self.m

    def estimated_fpr(self):
        return self.fill_ratio() ** self.k

    def expected_fpr(self):
        return theoretical_fpr(self.m, self.count, self.k)

    def memory_bytes(self):
        return len(self.bits)

    def clear(self):
        self.bits = bytearray(len(self.bits))
        self.count = 0


class CountingBloomFilter:
    def __init__(self, m, k, counter_bits=4, scheme="double", sticky=True):
        if counter_bits not in (1, 2, 4, 8):
            raise ValueError("counter_bits must be 1, 2, 4 or 8")
        self.m = m
        self.k = k
        self.counter_bits = counter_bits
        self.max_value = (1 << counter_bits) - 1
        self.per_byte = 8 // counter_bits
        self.sticky = sticky
        self.hashes = HashFamily(m, k, scheme)
        self.cells = bytearray((m + self.per_byte - 1) // self.per_byte)
        self.count = 0
        self.overflow_events = 0
        self.underflow_refusals = 0

    @classmethod
    def for_capacity(cls, n, p, counter_bits=4, scheme="double", sticky=True):
        m = optimal_m(n, p)
        return cls(m, optimal_k(m, n), counter_bits, scheme, sticky)

    def _get(self, i):
        shift = (i % self.per_byte) * self.counter_bits
        return (self.cells[i // self.per_byte] >> shift) & self.max_value

    def _set(self, i, value):
        idx = i // self.per_byte
        shift = (i % self.per_byte) * self.counter_bits
        mask = self.max_value << shift
        self.cells[idx] = (self.cells[idx] & ~mask & 0xFF) | (value << shift)

    def add(self, item):
        for i in self.hashes.indexes(item):
            c = self._get(i)
            if c == self.max_value:
                self.overflow_events += 1
            else:
                self._set(i, c + 1)
        self.count += 1

    def __contains__(self, item):
        for i in self.hashes.indexes(item):
            if self._get(i) == 0:
                return False
        return True

    def remove(self, item, guard=True):
        idx = self.hashes.indexes(item)
        if guard and any(self._get(i) == 0 for i in idx):
            self.underflow_refusals += 1
            return False
        for i in idx:
            c = self._get(i)
            if c == 0:
                continue
            if c == self.max_value and self.sticky:
                continue
            self._set(i, c - 1)
        self.count = max(0, self.count - 1)
        return True

    def max_counter(self):
        return max(self._get(i) for i in range(self.m))

    def nonzero_ratio(self):
        return sum(1 for i in range(self.m) if self._get(i)) / self.m

    def counter_histogram(self):
        hist = [0] * (self.max_value + 1)
        for i in range(self.m):
            hist[self._get(i)] += 1
        return hist

    def expected_fpr(self):
        return theoretical_fpr(self.m, max(self.count, 1), self.k)

    def memory_bytes(self):
        return len(self.cells)

    def theoretical_bytes(self):
        return math.ceil(self.m * self.counter_bits / 8)

    def to_bloom(self):
        bf = BloomFilter(self.m, self.k, self.hashes.scheme)
        for i in range(self.m):
            if self._get(i):
                bf.bits[i >> 3] |= 1 << (i & 7)
        bf.count = self.count
        return bf
