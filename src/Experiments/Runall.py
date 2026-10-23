import time
from dataclasses import dataclass, field
from pathlib import Path

from .CacheSimulation import ACTIVE_STRATEGIES, simulate_cache
from .CountingStudies import (
    churn_study,
    compare_standard_counting,
    counter_width_study,
)
from .CrossCheckStudies import MEMBER_KINDS, cross_check_study
from .CsvExport import write_csv
from .Figures import plot_optimal_k, plot_sweep
from .ParameterStudies import (
    optimal_k_study,
    sweep_hash_count,
    sweep_items,
    sweep_size,
)
from .StudyFigures import (
    plot_churn,
    plot_counter_widths,
    plot_counting_comparison,
    plot_cross_check,
)
from .UrlFilterSimulation import simulate_url_filter
from .Workload import CostModel
from .WorkloadFigures import plot_cache_results, plot_url_filter_results


@dataclass(frozen=True)
class RunSettings:
    seed: int = 0
    trials: int = 5
    queries: int = 20000

    sweep_items_per_filter: int = 5000
    sweep_hash_count: int = 7
    sweep_size_ratios: tuple[float, ...] = (4, 6, 8, 10, 12, 16, 20)
    sweep_slots: int = 50000
    sweep_item_counts: tuple[int, ...] = (
        1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000,
    )
    sweep_hash_counts: tuple[int, ...] = tuple(range(1, 21))
    optimal_items: int = 2000
    optimal_ratios: tuple[float, ...] = (6, 8, 10, 12, 16)
    optimal_span: float = 3.0

    counting_ratios: tuple[float, ...] = (4, 6, 8, 10, 12, 16)
    counter_bits: int = 4
    width_values: tuple[int, ...] = (2, 3, 4, 8)
    width_slots: int = 20000
    width_hash_count: int = 7
    churn_slots: int = 50000
    churn_hash_count: int = 7
    churn_live: int = 5000
    churn_rounds: int = 20
    churn_fraction: float = 0.2
    churn_probes: int = 5000

    cross_items: int = 5000
    cross_error_rates: tuple[float, ...] = (0.1, 0.01, 0.001)
    cross_filter_counts: tuple[int, ...] = (2, 3, 4)
    cross_queries: int = 50000

    cache_trials: int = 3
    cache_universe: int = 50000
    cache_capacity: int = 2000
    cache_requests: int = 50000
    cache_ratios: tuple[float, ...] = (4, 8, 12, 16)
    cache_rebuild_interval: int = 5000
    cache_window: int = 5000

    url_trials: int = 3
    url_blocklist: int = 10000
    url_benign: int = 50000
    url_requests: int = 100000
    url_malicious_fraction: float = 0.02
    url_ratios: tuple[float, ...] = (4, 6, 8, 10, 12, 16)

    zipf_exponent: float = 1.0
    cost: CostModel = field(default_factory=CostModel)


class _Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.line("")
        self.line(title)
        self.line("=" * len(title))

    def line(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")

        return path


def run_parameter_studies(s: RunSettings, csv: Path, fig: Path, report: _Report) -> None:
    report.section("False positive rate against parameters")

    by_size = sweep_size(
        [round(r * s.sweep_items_per_filter) for r in s.sweep_size_ratios],
        s.sweep_items_per_filter,
        s.sweep_hash_count,
        s.queries,
        s.trials,
        s.seed,
    )
    by_items = sweep_items(
        s.sweep_item_counts, s.sweep_slots, s.sweep_hash_count, s.queries, s.trials, s.seed
    )
    by_hashes = sweep_hash_count(
        s.sweep_hash_counts,
        s.sweep_slots,
        s.sweep_items_per_filter,
        s.queries,
        s.trials,
        s.seed,
    )
    rows, summaries = optimal_k_study(
        s.optimal_ratios, s.optimal_items, s.optimal_span, s.queries, s.trials, s.seed
    )

    for name, results in (("size", by_size), ("items", by_items), ("hashes", by_hashes)):
        report.line(f"sweep over {name}")

        for r in results:
            report.line(
                f"  m={r.size} k={r.hash_count} n={r.item_count} "
                f"empirical={r.empirical_rate:.6f} theory={r.theoretical_rate:.6f} "
                f"false_negatives={r.false_negatives}"
            )

    report.line("optimal hash count")

    for x in summaries:
        report.line(
            f"  m/n={x.bits_per_item:g} predicted={x.predicted_k_exact:.2f} "
            f"rounded={x.predicted_k_rounded} theory_best={x.theoretical_best_k} "
            f"empirical_best={x.empirical_best_k} "
            f"rate_at_predicted={x.rate_at_predicted_k:.6f} "
            f"best_rate={x.empirical_best_rate:.6f}"
        )

    write_csv(by_size, csv / "sweep_size.csv")
    write_csv(by_items, csv / "sweep_items.csv")
    write_csv(by_hashes, csv / "sweep_hashes.csv")
    write_csv(rows, csv / "optimal_k_rows.csv")
    write_csv(summaries, csv / "optimal_k_summary.csv")

    plot_sweep(
        by_size,
        "size",
        "array size m",
        f"False positive rate versus array size (n={s.sweep_items_per_filter}, k={s.sweep_hash_count})",
        fig / "sweep_size.png",
    )
    plot_sweep(
        by_items,
        "item_count",
        "inserted items n",
        f"False positive rate versus items (m={s.sweep_slots}, k={s.sweep_hash_count})",
        fig / "sweep_items.png",
    )
    plot_sweep(
        by_hashes,
        "hash_count",
        "hash count k",
        f"False positive rate versus hash count (m={s.sweep_slots}, n={s.sweep_items_per_filter})",
        fig / "sweep_hashes.png",
    )
    plot_optimal_k(rows, summaries, fig / "optimal_k.png")


def run_counting_studies(s: RunSettings, csv: Path, fig: Path, report: _Report) -> None:
    report.section("Counting Bloom filter studies")

    comparison = compare_standard_counting(
        s.counting_ratios,
        s.sweep_items_per_filter,
        s.counter_bits,
        s.queries,
        s.trials,
        s.seed,
    )
    widths = counter_width_study(
        s.width_values,
        s.width_slots,
        s.width_hash_count,
        s.sweep_items_per_filter,
        s.trials,
        s.seed,
    )
    churn = churn_study(
        s.churn_slots,
        s.churn_hash_count,
        s.churn_live,
        s.churn_rounds,
        s.churn_fraction,
        s.churn_probes,
        s.counter_bits,
        s.seed,
    )

    report.line("standard against counting at equal slot count")

    for r in comparison:
        report.line(
            f"  m/n={r.bits_per_item:g} k={r.hash_count} standard={r.standard_rate:.6f} "
            f"counting={r.counting_rate:.6f} theory={r.theoretical_rate:.6f} "
            f"standard_bytes={r.standard_memory_bytes} "
            f"counting_bytes={r.counting_memory_bytes} false_negatives={r.false_negatives}"
        )

    report.line("counter width")

    for r in widths:
        report.line(
            f"  bits={r.counter_bits} bytes={r.memory_bytes} "
            f"saturated={r.mean_saturated_counters:.1f} "
            f"dropped={r.mean_overflow_events:.1f} max_counter={r.mean_max_counter:.1f} "
            f"overflow_trials={r.overflow_trial_fraction:.2f} bound={r.overflow_bound:.4g} "
            f"false_negatives={r.false_negatives}"
        )

    report.line("add and remove churn")

    for r in churn:
        report.line(
            f"  round={r.round_number} counting_fp={r.counting_false_positive_rate:.6f} "
            f"standard_fp={r.standard_false_positive_rate:.6f} "
            f"counting_fill={r.counting_fill_ratio:.4f} "
            f"standard_fill={r.standard_fill_ratio:.4f} stale={r.stale_positive_rate:.6f} "
            f"saturated={r.saturated_counters} false_negatives={r.counting_false_negatives}"
        )

    write_csv(comparison, csv / "compare_counting.csv")
    write_csv(widths, csv / "counter_width.csv")
    write_csv(churn, csv / "churn.csv")

    plot_counting_comparison(comparison, fig / "compare_counting.png")
    plot_counter_widths(widths, fig / "counter_width.png")
    plot_churn(churn, fig / "churn.png")


def run_cross_check_studies(s: RunSettings, csv: Path, fig: Path, report: _Report) -> None:
    report.section("Cross checked Bloom filter studies")

    for kind in MEMBER_KINDS:
        results = cross_check_study(
            kind,
            s.cross_filter_counts,
            s.cross_error_rates,
            s.cross_items,
            s.cross_queries,
            s.trials,
            s.seed,
        )

        report.line(f"member kind {kind}")

        for r in results:
            report.line(
                f"  members={r.filter_count} target={r.error_rate:g} bytes={r.memory_bytes} "
                f"combined={r.combined_rate:.6f} theory={r.theoretical_rate:.6f} "
                f"member_rate={r.mean_member_rate:.6f} "
                f"disagreement={r.disagreement_rate:.6f} catch={r.catch_rate:.4f} "
                f"single_same_memory={r.single_filter_rate:.6f} "
                f"false_negatives={r.false_negatives}"
            )

        write_csv(results, csv / f"cross_check_{kind}.csv")
        plot_cross_check(results, fig / f"cross_check_{kind}.png")


def run_cache_simulation(s: RunSettings, csv: Path, fig: Path, report: _Report) -> None:
    report.section("Web cache simulation")

    results, timeline = simulate_cache(
        s.cache_ratios,
        ACTIVE_STRATEGIES,
        s.cache_universe,
        s.cache_capacity,
        s.cache_requests,
        s.zipf_exponent,
        s.cache_trials,
        s.cost,
        s.counter_bits,
        s.cache_rebuild_interval,
        s.cache_window,
        s.seed,
    )

    for r in results:
        report.line(
            f"  strategy={r.strategy} m/n={r.bits_per_item:g} hit_rate={r.hit_rate:.4f} "
            f"disk_lookups={r.disk_lookups:.0f} avoided={r.disk_lookups_avoided:.0f} "
            f"wasted={r.wasted_disk_lookups:.0f} wasted_per_miss={r.wasted_per_miss:.4f} "
            f"false_negatives={r.false_negatives:.0f} total_ms={r.total_ms:.0f} "
            f"time_saved={r.time_saved_fraction:.4f} "
            f"filter_bytes={r.filter_memory_bytes:.0f} fill={r.final_fill_ratio:.4f}"
        )

    write_csv(results, csv / "cache_simulation.csv")
    write_csv(timeline, csv / "cache_timeline.csv")
    plot_cache_results(
        results,
        timeline,
        s.cache_ratios[len(s.cache_ratios) // 2],
        fig / "cache_simulation.png",
    )


def run_url_filter_simulation(s: RunSettings, csv: Path, fig: Path, report: _Report) -> None:
    report.section("URL filter simulation")

    results = simulate_url_filter(
        s.url_ratios,
        s.url_blocklist,
        s.url_benign,
        s.url_requests,
        s.url_malicious_fraction,
        s.zipf_exponent,
        s.url_trials,
        s.cost,
        s.seed,
    )

    for r in results:
        report.line(
            f"  m/n={r.bits_per_item:g} k={r.hash_count:.0f} blocked={r.blocked:.0f} "
            f"missed={r.missed:.0f} remote_queries={r.remote_queries:.0f} "
            f"avoided={r.avoided_fraction:.4f} fp_request={r.false_positive_rate:.6f} "
            f"fp_distinct={r.distinct_false_positive_rate:.6f} "
            f"theory={r.theoretical_rate:.6f} time_saved={r.time_saved_fraction:.4f} "
            f"filter_bytes={r.filter_bytes:.0f} set_bytes={r.set_bytes:.0f} "
            f"memory_ratio={r.memory_ratio:.1f}"
        )

    write_csv(results, csv / "url_filter_simulation.csv")
    plot_url_filter_results(results, fig / "url_filter_simulation.png")


def run_all(out_dir: Path = Path("results"), settings: RunSettings = RunSettings()) -> Path:
    csv = out_dir / "csv"
    fig = out_dir / "figures"
    report = _Report()
    steps = (
        run_parameter_studies,
        run_counting_studies,
        run_cross_check_studies,
        run_cache_simulation,
        run_url_filter_simulation,
    )

    for step in steps:
        started = time.perf_counter()

        step(settings, csv, fig, report)
        report.line(f"finished {step.__name__} in {time.perf_counter() - started:.1f} s")

    return report.save(out_dir / "summary.txt")