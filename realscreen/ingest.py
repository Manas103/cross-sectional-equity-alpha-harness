"""Parses the 44 raw FRED CSV downloads into one wide quarterly-growth
panel.

Every series is resampled to quarterly by keeping the last real
observation in each (year, quarter), a declared simplification over a
true quarter-end index value (stated in the README); this is the one
transformation every series goes through regardless of whether it was
reported daily, weekly, monthly or quarterly in the source, so the same
one-lag autocorrelation specification in `testing.py` can be applied
identically to all 44.
"""

from __future__ import annotations

import csv

from realscreen import config


def quarter_key(date_str: str) -> tuple[int, int]:
    year, month, _ = date_str.split("-")
    year, month = int(year), int(month)
    quarter = (month - 1) // 3 + 1
    return (year, quarter)


def load_quarterly_levels(csv_path: str) -> dict[tuple[int, int], float]:
    by_quarter: dict[tuple[int, int], float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2 or row[1] in ("", "."):
                continue
            by_quarter[quarter_key(row[0])] = float(row[1])
    return by_quarter


def _prior_quarter(qk: tuple[int, int]) -> tuple[int, int]:
    year, quarter = qk
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def levels_to_growth(levels: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Quarter-over-quarter percent growth, only between two literally
    consecutive quarters (a gap in the source data breaks the pair rather
    than silently growing across it)."""
    growth: dict[tuple[int, int], float] = {}
    for qk, cur_val in levels.items():
        prev_qk = _prior_quarter(qk)
        if prev_qk not in levels:
            continue
        prev_val = levels[prev_qk]
        if prev_val == 0:
            continue
        growth[qk] = (cur_val / prev_val - 1.0) * 100.0
    return growth


def build_screen_panel(raw_csv_dir: str) -> tuple[list[tuple[int, int]], dict[str, dict]]:
    """Returns (usable_quarters, growth_by_sid) where usable_quarters is
    every (year, quarter) for which every one of the 44 series has both
    its own growth value and its immediately prior quarter's growth value
    (the lag the AR(1) specification needs)."""
    growth_by_sid: dict[str, dict] = {}
    for _, sid, _ in config.SERIES:
        levels = load_quarterly_levels(f"{raw_csv_dir}/{sid}.csv")
        growth_by_sid[sid] = levels_to_growth(levels)

    common = set.intersection(*[set(g.keys()) for g in growth_by_sid.values()])
    usable = sorted(qk for qk in common if _prior_quarter(qk) in common)
    return usable, growth_by_sid


def panel_rows(usable_quarters: list[tuple[int, int]], growth_by_sid: dict[str, dict]) -> list[dict]:
    rows = []
    for year, quarter in usable_quarters:
        row = {"year": year, "quarter": quarter}
        for _, sid, _ in config.SERIES:
            row[sid] = growth_by_sid[sid][(year, quarter)]
        rows.append(row)
    return rows
