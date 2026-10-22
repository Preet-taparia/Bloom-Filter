import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from Experiments.CsvExport import write_csv
from Experiments.Figures import plot_optimal_k, plot_sweep
from Experiments.ParameterStudies import (
    optimal_k_study,
    sweep_hash_count,
    sweep_items,
    sweep_size,
)


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true")


def _report(results: Sequence[object]) -> None:
    for result in results:
        print(
            f"m={result.size} k={result.hash_count} n={result.item_count} "
            f"empirical={result.empirical_rate:.6f} "
            f"theory={result.theoretical_rate:.6f} "
            f"false_negatives={result.false_negatives}"
        )


def _run_size(args: argparse.Namespace) -> None:
    sizes = [round(ratio * args.n) for ratio in args.ratios]
    results = sweep_size(sizes, args.n, args.k, args.queries, args.trials, args.seed)

    _report(results)
    print(write_csv(results, args.out_dir / "sweep_size.csv"))

    if not args.no_plot:
        print(
            plot_sweep(
                results,
                "size",
                "array size m",
                f"False positive rate versus array size (n={args.n}, k={args.k})",
                args.out_dir / "sweep_size.png",
            )
        )


def _run_items(args: argparse.Namespace) -> None:
    results = sweep_items(
        args.counts, args.m, args.k, args.queries, args.trials, args.seed
    )

    _report(results)
    print(write_csv(results, args.out_dir / "sweep_items.csv"))

    if not args.no_plot:
        print(
            plot_sweep(
                results,
                "item_count",
                "inserted items n",
                f"False positive rate versus items (m={args.m}, k={args.k})",
                args.out_dir / "sweep_items.png",
            )
        )


def _run_hashes(args: argparse.Namespace) -> None:
    results = sweep_hash_count(
        args.ks, args.m, args.n, args.queries, args.trials, args.seed
    )

    _report(results)
    print(write_csv(results, args.out_dir / "sweep_hashes.csv"))

    if not args.no_plot:
        print(
            plot_sweep(
                results,
                "hash_count",
                "hash count k",
                f"False positive rate versus hash count (m={args.m}, n={args.n})",
                args.out_dir / "sweep_hashes.png",
            )
        )


def _run_optimal_k(args: argparse.Namespace) -> None:
    rows, summaries = optimal_k_study(
        args.ratios, args.n, args.span, args.queries, args.trials, args.seed
    )

    for summary in summaries:
        print(
            f"m/n={summary.bits_per_item:g} predicted={summary.predicted_k_exact:.2f} "
            f"rounded={summary.predicted_k_rounded} "
            f"theory_best={summary.theoretical_best_k} "
            f"empirical_best={summary.empirical_best_k} "
            f"rate_at_predicted={summary.rate_at_predicted_k:.6f} "
            f"best_rate={summary.empirical_best_rate:.6f}"
        )

    print(write_csv(rows, args.out_dir / "optimal_k_rows.csv"))
    print(write_csv(summaries, args.out_dir / "optimal_k_summary.csv"))

    if not args.no_plot:
        print(plot_optimal_k(rows, summaries, args.out_dir / "optimal_k.png"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloom")
    commands = parser.add_subparsers(dest="command", required=True)

    size = commands.add_parser("sweep-size")
    size.add_argument("--n", type=int, default=5000)
    size.add_argument("--k", type=int, default=7)
    size.add_argument(
        "--ratios", type=_float_list, default=_float_list("4,6,8,10,12,16,20")
    )
    _add_common(size)
    size.set_defaults(handler=_run_size)

    items = commands.add_parser("sweep-items")
    items.add_argument("--m", type=int, default=50000)
    items.add_argument("--k", type=int, default=7)
    items.add_argument(
        "--counts",
        type=_int_list,
        default=_int_list("1000,2000,3000,4000,5000,6000,8000,10000,12000"),
    )
    _add_common(items)
    items.set_defaults(handler=_run_items)

    hashes = commands.add_parser("sweep-hashes")
    hashes.add_argument("--m", type=int, default=50000)
    hashes.add_argument("--n", type=int, default=5000)
    hashes.add_argument("--ks", type=_int_list, default=list(range(1, 21)))
    _add_common(hashes)
    hashes.set_defaults(handler=_run_hashes)

    optimal = commands.add_parser("optimal-k")
    optimal.add_argument("--n", type=int, default=2000)
    optimal.add_argument(
        "--ratios", type=_float_list, default=_float_list("6,8,10,12,16")
    )
    optimal.add_argument("--span", type=float, default=3.0)
    _add_common(optimal)
    optimal.set_defaults(handler=_run_optimal_k)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], None] = args.handler
    handler(args)

    return 0