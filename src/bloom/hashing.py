import hashlib
from math import gcd

from Exceptions import InvalidItemError

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
) -> list[int]:
    """Derive array positions for an item using double hashing.

    One keyed BLAKE2b digest is split into two independent 64 bit values.
    The first gives the starting position and the second gives the step.
    The i th position is start plus i times step, reduced modulo size. The
    step is adjusted to be coprime to size, so positions are distinct for
    any hash_count up to size. This is the Kirsch and Mitzenmacher scheme,
    which matches the false positive rate of hash_count independent hash
    functions while computing only one digest.

    Args:
        item: The item to hash.
        size: Number of slots in the target array, at least one.
        hash_count: How many positions to produce, at least one.
        seed: Value in the range 0 to MAX_SEED that selects an independent
            hash family. Callers are expected to have validated it.

    Returns:
        A list of hash_count positions, each in the range 0 to size minus 1.

    Raises:
        InvalidItemError: If the item type is unsupported.
    """
    digest = hashlib.blake2b(
        item_to_bytes(item),
        digest_size=16,
        salt=seed.to_bytes(8, "big"),
    ).digest()

    start = int.from_bytes(digest[:8], "big") % size
    step = coprime_step(int.from_bytes(digest[8:], "big"), size)

    return [(start + i * step) % size for i in range(hash_count)]
