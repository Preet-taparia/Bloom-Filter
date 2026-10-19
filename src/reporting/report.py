from dataclasses import dataclass, field
from pathlib import Path

from reporting.plots import bar_plot, line_plot
from reporting.tables import format_mapping, format_table, write_csv
from simulation import experiments as exp
from simulation.cache_sim import STRATEGIES, WorkloadConfig, run_workload


@dataclass
class Profile:
    item_count: int
    trials: int
    probes: int
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)


QUICK = Profile(
    item_count=1000,
    trials=2,
    probes=5000,
    workload=WorkloadConfig(universe=12000, stored=2400, queries_per_epoch=6000, epochs=4),
)

FULL = Profile(
    item_count=5000,
    trials=5,
    probes=20000,
    workload=WorkloadConfig(),
)


def column(rows: list[dict], key: str) -> list[float]:
    return [row[key] for row in rows]


def build_report(output: Path, profile: Profile, seed: int = 0) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    n, trials, probes = profile.item_count, profile.trials, profile.probes
    sections: list[str] = []

    def add(title: str, body: str, note: str = "") -> None:
        sections.append(f"## {title}\n\n{note}\n\n```\n{body}\n```\n".replace("\n\n\n", "\n\n"))

    size_rows = exp.fpr_vs_size(n, [4, 6, 8, 10, 12, 16], trials, probes, seed)
    write_csv(output / "fpr_vs_size.csv", size_rows)
    bits = column(size_rows, "bits_per_item")
    line_plot(
        output / "fpr_vs_size.png",
        "False positive rate against bits per item",
        "bits per item (m / n)",
        "false positive rate",
        {
            "theory": (bits, column(size_rows, "theory"), "line"),
            "measured": (bits, column(size_rows, "measured"), "points"),
        },
        log_y=True,
    )
    add(
        "False positive rate against array size",
        format_table(size_rows),
        f"Items inserted: {n}. Hash count is the rounded optimum for every size.",
    )

    k_bits = 8.0
    k_rows = exp.fpr_vs_hash_count(n, k_bits, list(range(1, 17)), trials, probes, seed)
    write_csv(output / "fpr_vs_hash_count.csv", k_rows)
    ks = column(k_rows, "hash_count")
    line_plot(
        output / "fpr_vs_hash_count.png",
        "False positive rate against hash count",
        "number of hash functions k",
        "false positive rate",
        {
            "theory": (ks, column(k_rows, "theory"), "line"),
            "measured": (ks, column(k_rows, "measured"), "points"),
        },
        log_y=True,
    )
    summary = exp.optimal_hash_summary(k_rows, n, k_bits)
    add(
        "Optimal number of hash functions",
        format_table(k_rows) + "\n\n" + format_mapping(summary),
        f"Array holds {k_bits:g} bits per item. The formula is k = ln(2) * m / n.",
    )

    size = round(10 * n * 2)
    counts = [n // 2, n, n * 2, n * 3, n * 4, n * 6, n * 8]
    n_rows = exp.fpr_vs_item_count(size, 7, counts, trials, probes, seed)
    write_csv(output / "fpr_vs_item_count.csv", n_rows)
    xs = column(n_rows, "item_count")
    line_plot(
        output / "fpr_vs_item_count.png",
        "False positive rate as the filter fills",
        "items inserted n",
        "false positive rate",
        {
            "theory": (xs, column(n_rows, "theory"), "line"),
            "measured": (xs, column(n_rows, "measured"), "points"),
        },
        log_y=True,
    )
    add(
        "False positive rate against inserted items",
        format_table(n_rows),
        f"Array size m = {size}, k = 7 fixed while the item count grows.",
    )

    memory_rows = exp.layered_fixed_memory(n, 12, [1, 2, 3, 4], trials, probes, seed)
    write_csv(output / "layered_fixed_memory.csv", memory_rows)
    add(
        "Multiple filters with the same total memory",
        format_table(memory_rows),
        "The total bit budget is split across layers that must all agree.",
    )

    target_rows = exp.layered_fixed_target(n, 0.01, [1, 2, 3, 4], trials, probes, seed)
    write_csv(output / "layered_fixed_target.csv", target_rows)
    line_plot(
        output / "layered_memory.png",
        "Memory needed for a one percent target",
        "layers",
        "total bits",
        {"memory": (column(target_rows, "layers"), column(target_rows, "memory_bits"), "line")},
    )
    add(
        "Multiple filters aiming at a one percent error target",
        format_table(target_rows),
        "The error budget is split so the product of layer rates meets the target.",
    )

    deletion = exp.counting_deletion(n, 0.01, 0.5, probes, 4, seed)
    add(
        "Counting filter deletion",
        format_mapping(deletion),
        "Half of the inserted items are removed from a counting filter. A standard filter keeps reporting them.",
    )

    overhead = exp.counter_overhead(n, 10, [2, 3, 4, 8], trials, seed)
    write_csv(output / "counter_overhead.csv", overhead)
    add(
        "Counter width overhead",
        format_table(overhead),
        "Memory relative to a standard filter of the same length equals the counter width.",
    )

    result = run_workload(profile.workload)
    write_csv(output / "workload_summary.csv", result.summary)
    write_csv(output / "workload_epochs.csv", result.epochs)
    bar_plot(
        output / "workload_disk_lookups.png",
        "Disk lookups for the URL filter workload",
        "disk lookups",
        [row["strategy"] for row in result.summary],
        [row["disk_lookups"] for row in result.summary],
    )
    epochs = sorted({row["epoch"] for row in result.epochs})
    series = {}

    for name in STRATEGIES[1:]:
        rates = [
            row["wasted_rate"]
            for row in result.epochs
            if row["strategy"] == name
        ]
        series[name] = (epochs, rates, "line")

    line_plot(
        output / "workload_wasted_rate.png",
        "Share of absent URLs that still hit the disk",
        "epoch",
        "wasted lookup rate",
        series,
    )
    config = profile.workload
    add(
        "URL filter workload",
        format_table(result.summary),
        f"Universe {config.universe} URLs, {config.stored} stored, {config.queries_per_epoch} Zipf queries per epoch, "
        f"{config.epochs} epochs, {int(config.churn * 100)} percent churn per epoch.",
    )

    document = "# Bloom filter experiment results\n\n" + "\n".join(sections)
    path = output / "results.md"
    path.write_text(document, encoding="utf-8")

    return path
