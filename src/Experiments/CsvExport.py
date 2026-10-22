import csv
from collections.abc import Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path


def write_csv(rows: Sequence[object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows or not all(is_dataclass(row) for row in rows):
        raise ValueError("write_csv needs a non empty sequence of dataclass rows")

    names = [field.name for field in fields(rows[0])]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))

    return path
