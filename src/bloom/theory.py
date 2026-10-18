from __future__ import annotations

import math
from collections.abc import Iterable

from validation import require_int, require_open_probability

LN2 = math.log(2.0)
LN2_SQUARED = LN2 * LN2


def optimal_hash_count(size: int, capacity: int) -> int:
    require_int("size", size, minimum=1)
    require_int("capacity", capacity, minimum=1)
    return max(1, round(LN2 * size / capacity))


def optimal_size(capacity: int, error_rate: float) -> int:
    require_int("capacity", capacity, minimum=1)
    require_open_probability("error_rate", error_rate)
    return max(1, math.ceil(-capacity * math.log(error_rate) / LN2_SQUARED))


def optimal_capacity(size: int, error_rate: float) -> int:
    require_int("size", size, minimum=1)
    require_open_probability("error_rate", error_rate)
    return max(1, math.floor(-size * LN2_SQUARED / math.log(error_rate)))


def bits_per_item(error_rate: float) -> float:
    require_open_probability("error_rate", error_rate)
    return -math.log(error_rate) / LN2_SQUARED


def expected_fill_ratio(size: int, hash_count: int, item_count: int) -> float:
    if item_count <= 0:
        return 0.0
    return 1.0 - math.exp(-hash_count * item_count / size)


def expected_false_positive_rate(size: int, hash_count: int, item_count: int) -> float:
    return expected_fill_ratio(size, hash_count, item_count) ** hash_count


def estimate_item_count(size: int, hash_count: int, fill_ratio: float) -> float:
    if fill_ratio <= 0.0:
        return 0.0
    if fill_ratio >= 1.0:
        return math.inf
    return -(size / hash_count) * math.log(1.0 - fill_ratio)


def combined_false_positive_rate(rates: Iterable[float]) -> float:
    combined = 1.0
    for rate in rates:
        combined *= rate
    return combined


def split_error_budget(error_rate: float, layer_count: int) -> float:
    require_open_probability("error_rate", error_rate)
    require_int("layer_count", layer_count, minimum=1)
    return error_rate ** (1.0 / layer_count)


def counter_overflow_bound(size: int, hash_count: int, item_count: int, counter_bits: int) -> float:
    if item_count <= 0:
        return 0.0
    threshold = 2 ** counter_bits
    load = hash_count * item_count / size
    try:
        bound = size * (math.e * load / threshold) ** threshold
    except OverflowError:
        return 1.0
    return min(1.0, bound)
