"""Typed CSV read/write helpers.

This is the only module in the project that imports ``pandas`` or ``csv``. Everything
tabular goes through here so that quoting, encoding, column order and the
``"unavailable"`` sentinel are handled in exactly one place.

Cells are always read back as ``str``, so ``"unavailable"`` never silently becomes ``NaN``
and then ``0``: each caller converts the cells it needs and treats the sentinel explicitly
(e.g. ``tomato3d.light.validate``); :func:`tomato3d.io.schemas.parse_maybe_float` is the
shared helper for that, but not every caller uses it yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from tomato3d.io.schemas import POSES_CSV_COLUMNS, Pose, SchemaError

#: Line terminator written on every platform, so that a session processed on Windows and
#: one processed on Linux produce byte-identical tables.
_LINE_TERMINATOR = "\n"
_ENCODING = "utf-8"


def write_table(path: Path, rows: Iterable[dict[str, str]], columns: Sequence[str]) -> None:
    """Write ``rows`` as CSV with exactly ``columns``, in that order.

    Raises:
        SchemaError: a row is missing a column or carries an unexpected one.
    """
    materialised = list(rows)
    expected = set(columns)
    for index, row in enumerate(materialised):
        missing = expected - set(row)
        unexpected = set(row) - expected
        if missing or unexpected:
            raise SchemaError(
                f"row {index} of {path.name} does not match the column contract "
                f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
            )
    frame = pd.DataFrame(materialised, columns=list(columns), dtype="object")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding=_ENCODING,
        lineterminator=_LINE_TERMINATOR,
    )


def read_table(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Read a CSV back as a list of string-valued dicts.

    Args:
        path: File to read.
        columns: When given, the header must match it exactly, in order.

    Raises:
        SchemaError: the file is unreadable or its header does not match ``columns``.
    """
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding=_ENCODING,
        )
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        # an empty or malformed CSV is an input error like a missing one (CLI exit 2), not a crash
        raise SchemaError(f"cannot read table {path}: {exc}") from exc
    if columns is not None and tuple(frame.columns) != tuple(columns):
        raise SchemaError(
            f"{path.name} header {tuple(frame.columns)} does not match the contract "
            f"{tuple(columns)}"
        )
    return frame.to_dict(orient="records")


def read_table_columns(path: Path) -> tuple[str, ...]:
    """Header of a CSV, also for a file with no data rows.

    Raises:
        SchemaError: the file is unreadable or has no header.
    """
    try:
        frame = pd.read_csv(path, dtype=str, nrows=0, encoding=_ENCODING)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SchemaError(f"cannot read the header of {path}: {exc}") from exc
    return tuple(frame.columns)


def write_poses_csv(path: Path, poses: Iterable[Pose]) -> None:
    """Write ``poses.csv``. Failed poses keep their identity and write ``unavailable``."""
    write_table(path, (pose.to_csv_row() for pose in poses), POSES_CSV_COLUMNS)


def read_poses_csv(path: Path) -> list[dict[str, str]]:
    """Read ``poses.csv`` as raw string cells, header order enforced."""
    return read_table(path, POSES_CSV_COLUMNS)
