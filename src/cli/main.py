import argparse
from pathlib import Path

from bloom.counting import CountingBloomFilter
from bloom.standard import BloomFilter
from reporting.report import FULL, QUICK, build_report
from reporting.tables import format_mapping, format_table
from simulation import experiments as exp
from simulation.cache_sim import WorkloadConfig, run_workload


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=2000, help="items to insert")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--probes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloom", description="Bloom filter study")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("demo", help="small tour of both filters")

    fpr = commands.add_parser("fpr", help="false positive rate against size")
    _common(fpr)

    optimal = commands.add_parser("optimal", help="check the optimal hash count")
    _common(optimal)
    optimal.add_argument("--bits", type=float, default=8.0, help="bits per item")

    layered = commands.add_parser("layered", help="multiple filter cross checking")
    _common(layered)
    layered.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4])
    layered.add_argument("--bits", type=float, default=12.0)

    counting = commands.add_parser("counting", help="deletion and counter overhead")
    _common(counting)
    counting.add_argument("--rate", type=float, default=0.01)

    workload = commands.add_parser("workload", help="simulated URL filter workload")
    workload.add_argument("--stored", type=int, default=8000)
    workload.add_argument("--universe", type=int, default=40000)
    workload.add_argument("--queries", type=int, default=20000)
    workload.add_argument("--epochs", type=int, default=5)
    workload.add_argument("--churn", type=float, default=0.3)
    workload.add_argument("--seed", type=int, default=1)

    report = commands.add_parser("report", help="write tables, plots and results")
    report.add_argument("--out", type=Path, default=Path("output"))
    report.add_argument("--quick", action="store_true", help="smaller and faster run")
    report.add_argument("--seed", type=int, default=0)

    return parser


def _demo() -> None:
    standard = BloomFilter.from_capacity(1000, 0.01)
    standard.update(f"user{index}" for index in range(1000))
    counting = CountingBloomFilter.from_capacity(1000, 0.01)
    counting.update(f"user{index}" for index in range(1000))

    print(standard)
    print("user5 present:", "user5" in standard)
    print("stranger present:", "stranger" in standard)
    print("expected false positive rate:", f"{standard.expected_false_positive_rate():.5f}")
    print()
    print(counting)
    print("remove user5:", counting.remove("user5"))
    print("user5 present after removal:", "user5" in counting)
    print("standard memory bits:", standard.memory_bits)
    print("counting memory bits:", counting.memory_bits)


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        _demo()
    elif args.command == "fpr":
        rows = exp.fpr_vs_size(args.n, [4, 6, 8, 10, 12, 16], args.trials, args.probes, args.seed)
        print(format_table(rows))
    elif args.command == "optimal":
        rows = exp.fpr_vs_hash_count(
            args.n, args.bits, list(range(1, 17)), args.trials, args.probes, args.seed
        )
        print(format_table(rows))
        print()
        print(format_mapping(exp.optimal_hash_summary(rows, args.n, args.bits)))
    elif args.command == "layered":
        rows = exp.layered_fixed_memory(
            args.n, args.bits, args.layers, args.trials, args.probes, args.seed
        )
        print(format_table(rows))
        print()
        target = exp.layered_fixed_target(
            args.n, 0.01, args.layers, args.trials, args.probes, args.seed
        )
        print(format_table(target))
    elif args.command == "counting":
        print(
            format_mapping(
                exp.counting_deletion(args.n, args.rate, 0.5, args.probes, 4, args.seed)
            )
        )
        print()
        print(format_table(exp.counter_overhead(args.n, 10, [2, 3, 4, 8], args.trials, args.seed)))
    elif args.command == "workload":
        config = WorkloadConfig(
            universe=args.universe,
            stored=args.stored,
            queries_per_epoch=args.queries,
            epochs=args.epochs,
            churn=args.churn,
            seed=args.seed,
        )
        print(format_table(run_workload(config).summary))
    elif args.command == "report":
        path = build_report(args.out, QUICK if args.quick else FULL, args.seed)
        print(f"wrote {path}")

    return 0
