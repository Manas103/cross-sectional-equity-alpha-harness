"""Builds data/real_dataset_screen_panel.csv (small, committed) from the
44 raw FRED CSV downloads (not committed). See README "Building and
running" for the exact download command.

Usage:
    python scripts/ingest_real_datasets.py --raw-dir <dir of 44 <SID>.csv files> --out-csv data/real_dataset_screen_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realscreen import config, ingest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-csv", default="data/real_dataset_screen_panel.csv")
    args = parser.parse_args()

    usable_quarters, growth_by_sid = ingest.build_screen_panel(args.raw_dir)
    rows = ingest.panel_rows(usable_quarters, growth_by_sid)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["year", "quarter"] + [sid for _, sid, _ in config.SERIES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"n_datasets={len(config.SERIES)}")
    print(f"n_usable_quarters={len(usable_quarters)}")
    if usable_quarters:
        print(f"first_quarter={usable_quarters[0]}")
        print(f"last_quarter={usable_quarters[-1]}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
