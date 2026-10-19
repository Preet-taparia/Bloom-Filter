import csv
from pathlib import Path


def _format_value(value: object) -> str:
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1.0:
            return f"{value:.7f}" if abs(value) < 0.0001 else f"{value:.5f}"

        return f"{value:.3f}"

    return str(value)


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render rows as an aligned plain text table."""
    if not rows:
        return "(no rows)"

    columns = columns or list(rows[0].keys())
    cells = [[_format_value(row[column]) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(line[index]) for line in cells))
        for index, column in enumerate(columns)
    ]

    header = "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    rule = "  ".join("=" * width for width in widths)
    body = [
        "  ".join(value.rjust(widths[index]) for index, value in enumerate(line))
        for line in cells
    ]

    return "\n".join([header, rule, *body])


def format_mapping(mapping: dict) -> str:
    width = max(len(str(key)) for key in mapping)

    return "\n".join(
        f"{str(key).ljust(width)}  {_format_value(value)}"
        for key, value in mapping.items()
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
