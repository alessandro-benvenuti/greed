#!/usr/bin/env python3
"""Return success only for a complete, reusable GED result shard."""

import argparse
import csv
import math
from pathlib import Path


def shard_is_complete(
    path: Path, expected_rows: int, cost_config_id: str, solver: str, timeout: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return False

    def valid_result(row: dict[str, str]) -> bool:
        try:
            lower = float(row["lower_bound"])
            upper = float(row["upper_bound"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            bool(row.get("pair_id"))
            and row.get("solver_status") in {"ok", "analytic_empty_graph"}
            and math.isfinite(lower) and math.isfinite(upper) and lower <= upper
        )

    return (
        len(rows) == expected_rows
        and all(valid_result(row) for row in rows)
        and all(row.get("cost_config_id") == cost_config_id for row in rows)
        and all(row.get("solver") == solver for row in rows)
        and all(row.get("time_limit_seconds") == str(timeout) for row in rows)
        and len({row.get("pair_id") for row in rows}) == expected_rows
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--cost-config-id", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    return 0 if shard_is_complete(
        args.path, args.expected_rows, args.cost_config_id, args.solver, args.timeout,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
