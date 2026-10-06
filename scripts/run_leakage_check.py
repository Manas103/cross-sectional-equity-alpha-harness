"""Builds a point-in-time vintage store with restatements, then runs the 7
seeded leakage bugs through the guarded accessor and confirms every one is
caught, plus one clean, correctly-written read that is not.

Writes docs/leakage_check_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xsalpha.leakage_bugs import ALL_LEAKAGE_BUGS
from xsalpha.leakage_detector import LeakageDetector, LeakageError
from xsalpha.vintage_store import build_simulated_vintage_store

NUM_ENTITIES = 500
NUM_DAYS = 2016


def main() -> None:
    trading_days = pd.bdate_range("2017-01-03", periods=NUM_DAYS)
    store = build_simulated_vintage_store(NUM_ENTITIES, trading_days)
    detector = LeakageDetector(store)

    simulation_clock = trading_days[1500]
    value_date = trading_days[1495]
    wall_clock_today = pd.Timestamp.today().normalize()

    lines = []
    caught = 0
    for bug_fn in ALL_LEAKAGE_BUGS:
        name = bug_fn.__name__
        try:
            if name == "bug_7_wall_clock_today":
                bug_fn(store, detector, value_date, simulation_clock, wall_clock_today)
            else:
                bug_fn(store, detector, value_date, simulation_clock)
            lines.append(f"NOT CAUGHT: {name}")
        except LeakageError as exc:
            caught += 1
            lines.append(f"caught: {name}: {exc}")

    clean_ok = True
    try:
        detector.guarded_as_of(value_date, simulation_clock, simulation_clock)
    except LeakageError as exc:
        clean_ok = False
        lines.append(f"FALSE POSITIVE on clean read: {exc}")
    if clean_ok:
        lines.append("clean read (query_date == simulation_clock): no false positive")

    summary = f"{caught} of {len(ALL_LEAKAGE_BUGS)} seeded leakage bugs caught"
    lines.insert(0, summary)
    output = "\n".join(lines)
    print(output)

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "leakage_check_output.txt").write_text(output + "\n", encoding="utf-8")

    assert caught == len(ALL_LEAKAGE_BUGS), summary
    assert clean_ok, "clean read incorrectly flagged as a leak"


if __name__ == "__main__":
    main()
