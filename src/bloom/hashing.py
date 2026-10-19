import hashlib

from bloom.exceptions import InvalidParameterError

MAX_SEED = (1 << 64) - 1


def item_to_bytes(item: object) -> bytes:
    """Convert a supported item into a type tagged byte string."""
    if isinstance(item, (bytes, bytearray)):
        return b"b" + bytes(item)

    if isinstance(item, str):
        return b"s" + item.encode("utf-8")

    if isinstance(item, int) and not isinstance(item, bool):
        return b"i" + str(item).encode("ascii")

    raise InvalidParameterError(
        f"item must be bytes, str or int, got {type(item).__name__}"
    )


def base_hashes(item: object, seed: int = 0) -> tuple[int, int]:
    """Return two 64 bit hashes used for double hashing."""
    salt = seed.to_bytes(8, "big")
    digest = hashlib.blake2b(
        item_to_bytes(item),
        digest_size=16,
        salt=salt,
    ).digest()

    first = int.from_bytes(digest[:8], "big")
    second = int.from_bytes(digest[8:], "big") | 1

    return first, second


def hash_indexes(
    item: object,
    size: int,
    hash_count: int,
    seed: int = 0,
) -> list[int]:
    """Derive hash_count array positions with Kirsch Mitzenmacher double hashing."""
    first, second = base_hashes(item, seed)

    return [(first + step * second) % size for step in range(hash_count)]
