from __future__ import annotations

import math
from collections.abc import Iterable

from validation import (
    require_fraction,
    require_int,
    require_open_probability,
)

LN2 = math.log(2.0)
LN2_SQUARED = LN2**2


def optimal_hash_count(size: int, capacity: int) -> int:
    """Calculate the approximately optimal number of hash functions."""
    require_int("size", size, minimum=1)
    require_int("capacity", capacity, minimum=1)

    return max(1, round(LN2 * size / capacity))


def optimal_size(capacity: int, error_rate: float) -> int:
    """Calculate the required bit-array size for a target false positive rate."""
    require_int("capacity", capacity, minimum=1)
    error_rate = require_open_probability("error_rate", error_rate)

    return max(
        1,
        math.ceil(-capacity * math.log(error_rate) / LN2_SQUARED),
    )


def optimal_capacity(size: int, error_rate: float) -> int:
    """Estimate the capacity for a given size and false positive rate."""
    require_int("size", size, minimum=1)
    error_rate = require_open_probability("error_rate", error_rate)

    return max(
        1,
        math.floor(-size * LN2_SQUARED / math.log(error_rate)),
    )


def bits_per_item(error_rate: float) -> float:
    """Calculate the optimal number of bits per item."""
    error_rate = require_open_probability("error_rate", error_rate)

    return -math.log(error_rate) / LN2_SQUARED


def expected_fill_ratio(
    size: int,
    hash_count: int,
    item_count: int,
) -> float:
    """Estimate the fraction of bits set after inserting items."""
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
    """Estimate the false positive probability."""
    fill_ratio = expected_fill_ratio(size, hash_count, item_count)

    return fill_ratio**hash_count


def estimate_item_count(
    size: int,
    hash_count: int,
    fill_ratio: float,
) -> float:
    """Estimate the number of inserted items from the fill ratio."""
    require_int("size", size, minimum=1)
    require_int("hash_count", hash_count, minimum=1)
    fill_ratio = require_fraction("fill_ratio", fill_ratio)

    if fill_ratio == 0.0:
        return 0.0

    if fill_ratio == 1.0:
        return math.inf

    return -(size / hash_count) * math.log1p(-fill_ratio)


def combined_false_positive_rate(rates: Iterable[float]) -> float:
    """Calculate the product of independent false positive probabilities."""
    combined = 1.0

    for rate in rates:
        combined *= require_fraction("rate", rate)

    return combined


def split_error_budget(error_rate: float, layer_count: int) -> float:
    """Calculate an equal per-layer rate for a multiplicative error budget."""
    error_rate = require_open_probability("error_rate", error_rate)
    require_int("layer_count", layer_count, minimum=1)

    return error_rate ** (1.0 / layer_count)


def counter_overflow_bound(
    size: int,
    hash_count: int,
    item_count: int,
    counter_bits: int,
) -> float:
    """Estimate an upper bound on counter overflow probability."""
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
