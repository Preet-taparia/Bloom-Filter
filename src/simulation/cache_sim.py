import random
from dataclasses import dataclass

from bloom.counting import CountingBloomFilter
from bloom.standard import BloomFilter
from bloom.theory import optimal_hash_count
from simulation.workloads import ZipfSampler, url_universe


@dataclass
class WorkloadConfig:
    universe: int = 40000
    stored: int = 8000
    queries_per_epoch: int = 20000
    epochs: int = 5
    churn: float = 0.3
    zipf_exponent: float = 0.8
    bits_per_item: float = 10.0
    counter_bits: int = 4
    seed: int = 1


STRATEGIES = (
    "no filter",
    "standard stale",
    "standard rebuilt",
    "counting",
)


@dataclass
class WorkloadResult:
    summary: list[dict[str, float | int | str]]
    epochs: list[dict[str, float | int | str]]


def _build_standard(config: WorkloadConfig, items: set[str], seed: int) -> BloomFilter:
    size = round(config.bits_per_item * config.stored)
    flt = BloomFilter(size, optimal_hash_count(size, config.stored), seed)
    flt.update(items)

    return flt


def run_workload(config: WorkloadConfig) -> WorkloadResult:
    rng = random.Random(config.seed)
    urls = url_universe(config.universe)
    stored = set(rng.sample(urls, config.stored))
    ranked = urls[:]
    rng.shuffle(ranked)
    sampler = ZipfSampler(ranked, config.zipf_exponent, rng)

    stale = _build_standard(config, stored, config.seed)
    rebuilt = _build_standard(config, stored, config.seed)
    counting = CountingBloomFilter(
        stale.size, stale.hash_count, config.counter_bits, config.seed
    )
    counting.update(stored)

    totals = {
        name: {"lookups": 0, "wasted": 0, "false_negatives": 0, "maintenance": 0}
        for name in STRATEGIES
    }
    totals["standard rebuilt"]["maintenance"] = len(stored)
    totals["standard stale"]["maintenance"] = len(stored)
    totals["counting"]["maintenance"] = len(stored)

    epoch_rows: list[dict[str, float | int | str]] = []
    total_queries = 0

    for epoch in range(1, config.epochs + 1):
        queries = sampler.sample(config.queries_per_epoch)
        total_queries += len(queries)
        filters = {
            "standard stale": stale,
            "standard rebuilt": rebuilt,
            "counting": counting,
        }
        epoch_wasted = {name: 0 for name in STRATEGIES}
        negatives = 0

        for url in queries:
            present = url in stored
            totals["no filter"]["lookups"] += 1

            if not present:
                negatives += 1
                epoch_wasted["no filter"] += 1
                totals["no filter"]["wasted"] += 1

            for name, flt in filters.items():
                answer = url in flt

                if answer:
                    totals[name]["lookups"] += 1

                    if not present:
                        totals[name]["wasted"] += 1
                        epoch_wasted[name] += 1
                elif present:
                    totals[name]["false_negatives"] += 1

        for name in STRATEGIES:
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "strategy": name,
                    "wasted_rate": epoch_wasted[name] / max(1, negatives),
                }
            )

        evicted = set(rng.sample(sorted(stored), int(len(stored) * config.churn)))
        candidates = [url for url in urls if url not in stored]
        arrivals = set(rng.sample(candidates, len(evicted)))
        stored = (stored - evicted) | arrivals

        stale.update(arrivals)
        totals["standard stale"]["maintenance"] += len(arrivals)

        rebuilt = _build_standard(config, stored, config.seed + epoch)
        totals["standard rebuilt"]["maintenance"] += len(stored)

        for url in evicted:
            counting.remove(url)

        counting.update(arrivals)
        totals["counting"]["maintenance"] += len(evicted) + len(arrivals)

    memory = {
        "no filter": 0,
        "standard stale": stale.memory_bits,
        "standard rebuilt": rebuilt.memory_bits,
        "counting": counting.memory_bits,
    }
    baseline = totals["no filter"]["lookups"]
    summary: list[dict[str, float | int | str]] = []

    for name in STRATEGIES:
        data = totals[name]
        summary.append(
            {
                "strategy": name,
                "queries": total_queries,
                "disk_lookups": data["lookups"],
                "wasted_lookups": data["wasted"],
                "lookups_avoided": baseline - data["lookups"],
                "avoided_pct": 100.0 * (baseline - data["lookups"]) / baseline,
                "false_negatives": data["false_negatives"],
                "memory_bits": memory[name],
                "maintenance_ops": data["maintenance"],
            }
        )

    return WorkloadResult(summary, epoch_rows)
