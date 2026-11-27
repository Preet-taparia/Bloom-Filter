from math import gcd

from .exceptions import InvalidItemError

from contextvars import ContextVar
from .HashAlgorithms import murmur3, xxhash, fnv1a, md5, blake2b

ALGORITHMS = ("murmur3", "xxhash", "fnv1a", "md5", "blake2b")
DEFAULT_ALGORITHM = ContextVar("hash_algorithm", default="blake2b")


def hash_pair(data: bytes, seed: int, algorithm: str) -> tuple[int, int]:
    payload = seed.to_bytes(8, "big") + data
    if algorithm in ("murmur3", "xxhash"):
        function = murmur3 if algorithm == "murmur3" else xxhash
        return function(payload, 0), function(payload, 0x9e3779b9)
    if algorithm == "fnv1a":
        return fnv1a(b"\x00" + payload), fnv1a(b"\x01" + payload)
    if algorithm == "md5":
        digest = md5(payload)
    elif algorithm == "blake2b":
        digest = blake2b(data, digest_size=16, salt=seed.to_bytes(8, "big"))
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:], "big")


MAX_SEED = (1 << 64) - 1


def item_to_bytes(item: object) -> bytes:
    """Convert a supported item into a type tagged byte string.

    The leading tag keeps values of different types apart, so the string
    "1", the integer 1 and the bytes b"1" all hash differently.

    Args:
        item: A bytes, bytearray, str or int. Booleans are rejected.

    Returns:
        The tagged byte representation of the item.

    Raises:
        InvalidItemError: If the type is unsupported, a string cannot be
            encoded as UTF 8, or an integer is too large to convert.
    """
    try:
        if isinstance(item, (bytes, bytearray)):
            return b"b" + bytes(item)

        if isinstance(item, str):
            return b"s" + item.encode("utf-8")

        if isinstance(item, int) and not isinstance(item, bool):
            return b"i" + str(item).encode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise InvalidItemError(f"item cannot be converted to bytes: {exc}") from exc

    raise InvalidItemError(f"item must be bytes, str or int, got {type(item).__name__}")


def coprime_step(raw: int, size: int) -> int:
    """Map a raw hash value to a step that shares no factor with size.

    A step coprime to size makes the sequence start, start plus step, start
    plus two steps, and so on, visit size different positions before it
    repeats. This guarantees that an item never maps to the same slot twice
    as long as the hash count does not exceed the size.

    Args:
        raw: Any nonnegative integer, normally 64 bits of a digest.
        size: Number of slots in the array, at least one.

    Returns:
        A step in the range 1 to max(1, size minus 1) with gcd(step, size)
        equal to 1.
    """
    step = raw % size or 1

    while gcd(step, size) != 1:
        step = step + 1 if step + 1 < size else 1

    return step


def hash_indexes(
    item: object,
    size: int,
    hash_count: int,
    seed: int = 0,
    algorithm: str = "blake2b",
) -> list[int]:
    """Derive positions using the selected hash pair and double hashing.

    The step is adjusted to be coprime to size, so positions do not repeat
    when hash_count <= size. Callers validate size, hash_count and seed.
    BLAKE2b remains the backward-compatible default.
    """
    first, second = hash_pair(item_to_bytes(item), seed, algorithm)
    start = first % size
    step = coprime_step(second, size)
    return [(start + i * step) % size for i in range(hash_count)]
