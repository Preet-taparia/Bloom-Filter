import math
from statistics import mean

from bloom.counting import CountingBloomFilter
from bloom.layered import LayeredBloomFilter
from bloom.standard import BloomFilter
from bloom.theory import (
    counter_overflow_bound,
    expected_false_positive_rate,
    optimal_hash_count,
)
from simulation.workloads import member_keys, probe_keys

Row = dict[str, float | int | str]


def measure_false_positive_rate(candidate: object, probes: list[str]) -> float:
    return sum(1 for probe in probes if probe in candidate) / len(probes)


def fpr_vs_size(
    item_count: int,
    bits_per_item_values: list[float],
    trials: int,
    probes: int,
    seed: int = 0,
) -> list[Row]:
    rows: list[Row] = []

    for bits in bits_per_item_values:
        size = max(1, round(bits * item_count))
        hash_count = optimal_hash_count(size, item_count)
        measured = []

        for trial in range(trials):
            flt = BloomFilter(size, hash_count, seed + trial)
            flt.update(member_keys(item_count, trial))
            measured.append(measure_false_positive_rate(flt, probe_keys(probes, trial)))

        rows.append(
            {
                "bits_per_item": bits,
                "size": size,
                "hash_count": hash_count,
                "theory": expected_false_positive_rate(size, hash_count, item_count),
                "measured": mean(measured),
            }
        )

    return rows


def fpr_vs_hash_count(
    item_count: int,
    bits_per_item: float,
    hash_counts: list[int],
    trials: int,
    probes: int,
    seed: int = 0,
) -> list[Row]:
    size = round(bits_per_item * item_count)
    rows: list[Row] = []

    for hash_count in hash_counts:
        measured = []

        for trial in range(trials):
            flt = BloomFilter(size, hash_count, seed + trial)
            flt.update(member_keys(item_count, trial))
            measured.append(measure_false_positive_rate(flt, probe_keys(probes, trial)))

        rows.append(
            {
                "hash_count": hash_count,
                "theory": expected_false_positive_rate(size, hash_count, item_count),
                "measured": mean(measured),
            }
        )

    return rows


def optimal_hash_summary(
    rows: list[Row],
    item_count: int,
    bits_per_item: float,
) -> dict[str, float | int]:
    size = round(bits_per_item * item_count)
    continuous = math.log(2.0) * size / item_count
    best_measured = min(rows, key=lambda row: row["measured"])
    best_theory = min(rows, key=lambda row: row["theory"])

    return {
        "bits_per_item": bits_per_item,
        "formula_k": continuous,
        "formula_k_rounded": optimal_hash_count(size, item_count),
        "best_theory_k": int(best_theory["hash_count"]),
        "best_measured_k": int(best_measured["hash_count"]),
        "best_measured_fpr": best_measured["measured"],
    }


def fpr_vs_item_count(
    size: int,
    hash_count: int,
    item_counts: list[int],
    trials: int,
    probes: int,
    seed: int = 0,
) -> list[Row]:
    checkpoints = sorted(item_counts)
    measured: dict[int, list[float]] = {count: [] for count in checkpoints}

    for trial in range(trials):
        flt = BloomFilter(size, hash_count, seed + trial)
        keys = member_keys(checkpoints[-1], trial)
        probe_set = probe_keys(probes, trial)
        inserted = 0

        for count in checkpoints:
            flt.update(keys[inserted:count])
            inserted = count
            measured[count].append(measure_false_positive_rate(flt, probe_set))

    return [
        {
            "item_count": count,
            "load": count / size,
            "theory": expected_false_positive_rate(size, hash_count, count),
            "measured": mean(measured[count]),
        }
        for count in checkpoints
    ]


def layered_fixed_memory(
    item_count: int,
    bits_per_item: float,
    layer_counts: list[int],
    trials: int,
    probes: int,
    seed: int = 0,
) -> list[Row]:
    total_bits = round(bits_per_item * item_count)
    rows: list[Row] = []

    for layers in layer_counts:
        layer_size = max(1, total_bits // layers)
        hash_count = optimal_hash_count(layer_size, item_count)
        measured = []
        theory = 0.0

        for trial in range(trials):
            flt = LayeredBloomFilter(layer_size, hash_count, layers, seed + 10 * trial)
            flt.update(member_keys(item_count, trial))
            theory = flt.expected_false_positive_rate()
            measured.append(measure_false_positive_rate(flt, probe_keys(probes, trial)))

        rows.append(
            {
                "layers": layers,
                "layer_size": layer_size,
                "hash_count": hash_count,
                "memory_bits": layer_size * layers,
                "theory": theory,
                "measured": mean(measured),
            }
        )

    return rows


def layered_fixed_target(
    item_count: int,
    error_rate: float,
    layer_counts: list[int],
    trials: int,
    probes: int,
    seed: int = 0,
) -> list[Row]:
    rows: list[Row] = []

    for layers in layer_counts:
        measured = []
        theory = 0.0
        memory = 0
        hash_count = 0

        for trial in range(trials):
            flt = LayeredBloomFilter.from_capacity(
                item_count, error_rate, layers, seed + 10 * trial
            )
            flt.update(member_keys(item_count, trial))
            theory = flt.expected_false_positive_rate()
            memory = flt.memory_bits
            hash_count = flt.hash_count
            measured.append(measure_false_positive_rate(flt, probe_keys(probes, trial)))

        rows.append(
            {
                "layers": layers,
                "hash_count": hash_count,
                "memory_bits": memory,
                "target": error_rate,
                "theory": theory,
                "measured": mean(measured),
            }
        )

    return rows


def counting_deletion(
    item_count: int,
    error_rate: float,
    remove_fraction: float,
    probes: int,
    counter_bits: int = 4,
    seed: int = 0,
) -> dict[str, float | int]:
    keys = member_keys(item_count)
    removed = keys[: int(item_count * remove_fraction)]
    remaining = keys[len(removed) :]
    fresh = probe_keys(probes)

    counting = CountingBloomFilter.from_capacity(
        item_count, error_rate, counter_bits, seed
    )
    standard = BloomFilter.from_capacity(item_count, error_rate, seed)
    counting.update(keys)
    standard.update(keys)

    successful = sum(1 for key in removed if counting.remove(key))
    false_negatives = sum(1 for key in remaining if key not in counting)
    removed_still_present = sum(1 for key in removed if key in counting)
    standard_removed_present = sum(1 for key in removed if key in standard)

    return {
        "inserted": item_count,
        "removed": len(removed),
        "removals_succeeded": successful,
        "false_negatives_among_remaining": false_negatives,
        "counting_removed_still_present": removed_still_present / len(removed),
        "standard_removed_still_present": standard_removed_present / len(removed),
        "counting_fpr_on_fresh_keys": measure_false_positive_rate(counting, fresh),
        "theory_fpr_after_removal": expected_false_positive_rate(
            counting.size, counting.hash_count, len(remaining)
        ),
        "standard_memory_bits": standard.memory_bits,
        "counting_memory_bits": counting.memory_bits,
    }


def counter_overhead(
    item_count: int,
    bits_per_item: float,
    counter_bits_values: list[int],
    trials: int,
    seed: int = 0,
) -> list[Row]:
    size = round(bits_per_item * item_count)
    hash_count = optimal_hash_count(size, item_count)
    rows: list[Row] = []

    for counter_bits in counter_bits_values:
        overflow, maximum, saturated = [], [], []

        for trial in range(trials):
            flt = CountingBloomFilter(size, hash_count, counter_bits, seed + trial)
            flt.update(member_keys(item_count, trial))
            overflow.append(flt.overflow_events)
            maximum.append(flt.max_counter)
            saturated.append(flt.saturated_counters)

        rows.append(
            {
                "counter_bits": counter_bits,
                "memory_bits": size * counter_bits,
                "memory_vs_standard": float(counter_bits),
                "overflow_bound": counter_overflow_bound(
                    size, hash_count, item_count, counter_bits
                ),
                "avg_overflow_events": mean(overflow),
                "avg_max_counter": mean(maximum),
                "avg_saturated_counters": mean(saturated),
            }
        )

    return rows
