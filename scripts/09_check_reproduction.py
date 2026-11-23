#!/usr/bin/env python
"""Check the generated tables against the values published in the paper.

Reads the CSV tables written by ``05_statistics.py`` and ``08_tables.py`` and compares them with
``tests/paper_values.json``. Numbers must agree within the tolerance given there (half a unit of
the last digit reported in the paper unless noted); text and boolean cells must match exactly.

Tables that have not been generated are reported as skipped. The exit code is 1 if any value
deviates, so the script can be used in automated checks.

Usage::

    python scripts/09_check_reproduction.py
    python scripts/09_check_reproduction.py --tables-dir reports/tables --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ecgath.analysis import DEFAULT_TABLES_DIR

logger = logging.getLogger("check")

DEFAULT_EXPECTED = Path("tests/paper_values.json")
FLOAT_SLACK = 1e-9  # absorbs binary rounding when a value sits exactly on the tolerance limit


@dataclass
class CheckResult:
    """Outcome of comparing one table."""

    name: str
    n_values: int = 0
    skipped: bool = False
    failures: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIPPED"
        return "FAILED" if self.failures else "OK"


def column_tolerance(check: Dict[str, Any], column: str) -> float:
    """Tolerance of a column: an entry matching the column name or its suffix, else the default."""
    for pattern, tolerance in check.get("column_tolerances", {}).items():
        if column == pattern or column.endswith(pattern):
            return tolerance
    return check.get("tolerance", 0)


def values_match(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return str(actual).strip().lower() == str(expected).lower()
    if isinstance(expected, str):
        return str(actual) == expected
    try:
        actual = float(actual)
    except (TypeError, ValueError):
        return False
    if math.isnan(actual):
        return False
    return abs(actual - expected) <= tolerance + FLOAT_SLACK


def find_row(table: pd.DataFrame, key: Dict[str, str]) -> Optional[pd.Series]:
    mask = pd.Series(True, index=table.index)
    for column, value in key.items():
        mask &= table[column].astype(str) == str(value)
    rows = table[mask]
    return rows.iloc[0] if len(rows) == 1 else None


def run_check(check: Dict[str, Any], tables_dir: Path) -> CheckResult:
    result = CheckResult(check["name"])
    path = tables_dir / check["file"]
    if not path.exists():
        result.skipped = True
        return result

    table = pd.read_csv(path)
    if "index_column" in check:  # table written with an unnamed index column
        table = table.rename(columns={table.columns[0]: check["index_column"]})

    for expected_row in check["rows"]:
        key = expected_row["key"]
        label = ", ".join(str(v) for v in key.values())
        row = find_row(table, key)
        if row is None:
            result.failures.append(f"{label}: row not found")
            continue
        for column, expected in expected_row["values"].items():
            result.n_values += 1
            if column not in row.index:
                result.failures.append(f"{label} / {column}: column not found")
            elif not values_match(row[column], expected, column_tolerance(check, column)):
                result.failures.append(
                    f"{label} / {column}: expected {expected}, got {row[column]}"
                )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="directory with the generated tables (default: reports/tables)",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="published values and tolerances (default: tests/paper_values.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="list every deviation")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checks = json.loads(args.expected.read_text(encoding="utf-8"))["checks"]
    results = [run_check(check, args.tables_dir) for check in checks]

    for result in results:
        detail = "" if result.skipped else f" ({result.n_values} values)"
        logger.info("%-8s %s%s", result.status, result.name, detail)
        shown = result.failures if args.verbose else result.failures[:5]
        for failure in shown:
            logger.info("           %s", failure)
        if len(result.failures) > len(shown):
            logger.info(
                "           ... %d more (use --verbose)", len(result.failures) - len(shown)
            )

    n_failed = sum(1 for r in results if r.failures)
    n_skipped = sum(1 for r in results if r.skipped)
    n_checked = sum(r.n_values for r in results)
    if n_failed:
        logger.info("\n%d of %d tables deviate from the published values.", n_failed, len(results))
        return 1
    if n_skipped == len(results):
        logger.info("\nNo tables found in %s. Run `make reports` first.", args.tables_dir)
        return 1
    skipped_note = f" ({n_skipped} tables not generated)" if n_skipped else ""
    logger.info("\nAll %d checked values agree with the paper%s.", n_checked, skipped_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())