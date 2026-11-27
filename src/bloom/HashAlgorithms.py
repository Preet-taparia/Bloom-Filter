"""Library-backed hashes preserving the original Bloom filter hash interface.

MurmurHash3, xxHash32 and FNV-1a use native-backed mmh3, xxhash and
fnv-c respectively. MD5 and BLAKE2b use Python's native-backed hashlib.
"""
import hashlib

import mmh3
import xxhash as _xxhash
from fnv_c import fnv1a_64


def murmur3(data: bytes, seed: int = 0) -> int:
    """Return unsigned MurmurHash3 x86-32, preserving 32-bit seed wrapping."""
    return mmh3.hash(data, seed=seed & 0xffffffff, signed=False)


def xxhash(data: bytes, seed: int = 0) -> int:
    """Return unsigned xxHash32, preserving 32-bit seed wrapping."""
    return _xxhash.xxh32_intdigest(data, seed=seed & 0xffffffff)


def fnv1a(data: bytes) -> int:
    """Return unsigned FNV-1a 64 using fnv-c's native implementation."""
    return fnv1a_64(data)


def md5(data: bytes) -> bytes:
    """Return the 16-byte MD5 digest for non-security Bloom filter use."""
    return hashlib.md5(data, usedforsecurity=False).digest()


def blake2b(data: bytes, digest_size: int = 16, salt: bytes = b'') -> bytes:
    """Return BLAKE2b bytes, preserving digest size and zero-padded salt."""
    return hashlib.blake2b(data, digest_size=digest_size, salt=salt).digest()
