"""Drop the pre-gap segment from nobitex CSV files.

Finds the largest inter-snapshot gap in each file and discards all rows
before it, keeping only the longest contiguous tail.

Usage::

    uv run python scripts/trim_gap.py [--data-dir data/nobitex_data]
                                      [--threshold-hours 1]
                                      [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


_TIMESTAMP_COLS = {
    "orderbook": "time",
    "trades": "snapshot_time",
}


def _file_kind(path: Path) -> str:
    if "orderbook" in path.name:
        return "orderbook"
    if "trades" in path.name:
        return "trades"
    raise ValueError(f"cannot determine kind for {path.name}")


def trim_file(path: Path, threshold: pd.Timedelta, dry_run: bool) -> None:
    kind = _file_kind(path)
    ts_col = _TIMESTAMP_COLS[kind]

    df = pd.read_csv(path, parse_dates=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    diffs = df[ts_col].diff()
    gap_idx = diffs[diffs > threshold].idxmax()
    largest_gap = diffs[gap_idx]

    if largest_gap <= threshold:
        print(f"  {path.name}: no gap above threshold — skipped")
        return

    cut_ts = df[ts_col].iloc[gap_idx]
    rows_before = gap_idx
    rows_after = len(df) - gap_idx

    print(
        f"  {path.name}: gap of {largest_gap} at index {gap_idx} "
        f"({df[ts_col].iloc[gap_idx - 1]} -> {cut_ts})"
    )
    print(f"    dropping {rows_before} rows before gap, keeping {rows_after}")

    if not dry_run:
        df.iloc[gap_idx:].to_csv(path, index=False)
        print(f"    saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim pre-gap rows from nobitex CSVs.")
    parser.add_argument(
        "--data-dir",
        default="data/nobitex_data",
        help="Directory containing nobitex CSV files (default: data/nobitex_data)",
    )
    parser.add_argument(
        "--threshold-hours",
        type=float,
        default=1.0,
        help="Minimum gap size in hours to trigger trimming (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be trimmed without writing files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    threshold = pd.Timedelta(hours=args.threshold_hours)
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        print(f"no CSV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"threshold: {threshold}  |  dry_run: {args.dry_run}\n")
    for path in csvs:
        trim_file(path, threshold, args.dry_run)


if __name__ == "__main__":
    main()
