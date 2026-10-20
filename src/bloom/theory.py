import math
from collections.abc import Iterable

from Bloom.Validation import (
    require_fraction,
    require_int,
    require_open_probability,
)

LN2 = math.log(2.0)
LN2_SQUARED = LN2**2


def optimal_hash_count(size: int, capacity: int) -> int:
    """Return the hash count that minimises the false positive rate.

    Uses k = (m / n) ln 2 rounded to the nearest integer, never below one.

    Args:
        size: Number of slots m in the filter.
        capacity: Number of distinct items n the filter should hold.

    Raises:
        InvalidParameterError: If either argument is not a positive integer.
    """
    require_int("size", size, minimum=1)
    require_int("capacity", capacity, minimum=1)

    return max(1, round(LN2 * size / capacity))


def optimal_size(capacity: int, error_rate: float) -> int:
    """Return the number of slots needed for a target false positive rate.

    Uses m = ceil(n ln(1 / p) / (ln 2) squared).

    Args:
        capacity: Number of distinct items n the filter should hold.
        error_rate: Target false positive probability p, strictly between
            0 and 1.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    require_int("capacity", capacity, minimum=1)
    error_rate = require_open_probability("error_rate", error_rate)

    return max(1, math.ceil(-capacity * math.log(error_rate) / LN2_SQUARED))


def optimal_capacity(size: int, error_rate: float) -> int:
    """Return how many items fit in a filter while meeting an error rate.

    This inverts optimal_size and rounds down so the target is respected.

    Args:
        size: Number of slots m in the filter.
        error_rate: Target false positive probability, strictly between
            0 and 1.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    require_int("size", size, minimum=1)
    error_rate = require_open_probability("error_rate", error_rate)

    return max(1, math.floor(-size * LN2_SQUARED / math.log(error_rate)))


def bits_per_item(error_rate: float) -> float:
    """Return the optimal number of slots per item for an error rate.

    Raises:
        InvalidParameterError: If error_rate is not strictly between 0 and 1.
    """
    error_rate = require_open_probability("error_rate", error_rate)

    return -math.log(error_rate) / LN2_SQUARED


def expected_fill_ratio(size: int, hash_count: int, item_count: int) -> float:
    """Return the expected fraction of occupied slots after insertions.

    Uses 1 minus e to the power of minus k n / m, assuming independent
    uniform hashing.

    Args:
        size: Number of slots m.
        hash_count: Hash functions per item k.
        item_count: Number of distinct items inserted n, zero or more.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    require_int("size", size, minimum=1)
    require_int("hash_count", hash_count, minimum=1)
    require_int("item_count", item_count, minimum=0)

    if item_count == 0:
        return 0.0

    return -math.expm1(-hash_count * item_count / size)


def expected_false_positive_rate(
    size: int,
    hash_count: int,
    item_count: int,
) -> float:
    """Return the expected false positive probability.

    Computed as the expected fill ratio raised to the hash count.

    Args:
        size: Number of slots m.
        hash_count: Hash functions per item k.
        item_count: Number of distinct items inserted n.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    fill_ratio = expected_fill_ratio(size, hash_count, item_count)

    return fill_ratio**hash_count


def estimate_item_count(size: int, hash_count: int, fill_ratio: float) -> float:
    """Estimate the number of distinct items from a measured fill ratio.

    Uses n = minus (m / k) ln(1 minus fill ratio). A completely full filter
    carries no information, so infinity is returned for it.

    Args:
        size: Number of slots m.
        hash_count: Hash functions per item k.
        fill_ratio: Measured fraction of occupied slots, from 0 to 1.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    require_int("size", size, minimum=1)
    require_int("hash_count", hash_count, minimum=1)
    fill_ratio = require_fraction("fill_ratio", fill_ratio)

    if fill_ratio == 0.0:
        return 0.0

    if fill_ratio == 1.0:
        return math.inf

    return -(size / hash_count) * math.log1p(-fill_ratio)


def combined_false_positive_rate(rates: Iterable[float]) -> float:
    """Return the product of false positive rates of independent filters.

    A query is a false positive only if every filter wrongly accepts it,
    so independent rates multiply. The result is only valid when the
    filters use independent hash families.

    Raises:
        InvalidParameterError: If any rate is not between 0 and 1.
    """
    combined = 1.0

    for rate in rates:
        combined *= require_fraction("rate", rate)

    return combined


def split_error_budget(error_rate: float, layer_count: int) -> float:
    """Return the equal per filter rate whose product meets an error rate.

    This is the layer_count th root of error_rate.

    Args:
        error_rate: Combined target rate, strictly between 0 and 1.
        layer_count: Number of independent filters sharing the budget.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    error_rate = require_open_probability("error_rate", error_rate)
    require_int("layer_count", layer_count, minimum=1)

    return error_rate ** (1.0 / layer_count)


def counter_overflow_bound(
    size: int,
    hash_count: int,
    item_count: int,
    counter_bits: int,
) -> float:
    """Return an upper bound on the chance that any counter overflows.

    Each counter is approximately Poisson with mean k n / m. A Chernoff
    tail bound gives the chance a single counter reaches 2 to the power of
    counter_bits, and a union bound over all counters scales it by m. The
    result is clamped to 1.

    Args:
        size: Number of counters m.
        hash_count: Hash functions per item k.
        item_count: Number of items inserted n.
        counter_bits: Width of each counter in bits.

    Raises:
        InvalidParameterError: If an argument is invalid.
    """
    require_int("size", size, minimum=1)
    require_int("hash_count", hash_count, minimum=1)
    require_int("item_count", item_count, minimum=0)
    require_int("counter_bits", counter_bits, minimum=1)

    if item_count == 0:
        return 0.0

    threshold = 2**counter_bits
    load = hash_count * item_count / size

    try:
        bound = size * (math.e * load / threshold) ** threshold
    except OverflowError:
        return 1.0

    return min(1.0, bound)
