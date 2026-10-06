"""7 seeded, named leakage bugs, each a realistic way a backtester call site
reads a later vintage than it should. Each function takes (store, detector,
value_date, simulation_clock) and performs the buggy read through the
guarded accessor; `scripts/run_leakage_check.py` asserts the detector raises
LeakageError for every one of the 7.
"""
from __future__ import annotations

import pandas as pd

from .leakage_detector import LeakageDetector
from .vintage_store import VintageStore


def bug_1_use_final_restated_value(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock):
    """Classic bug: query with the dataset's final/max knowledge date instead of the rebalance date."""
    final_date = store.max_knowledge_date_overall()
    return detector.guarded_as_of(value_date, final_date, simulation_clock)


def bug_2_off_by_one_day_ahead(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock):
    """Off-by-one: query one calendar day after the simulation clock."""
    return detector.guarded_as_of(value_date, simulation_clock + pd.Timedelta(days=1), simulation_clock)


def bug_3_bfill_across_restatement(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock):
    """Backward-fill bug: look ahead 30 days for 'the' value and use whatever knowledge_date comes back,
    which can be a restatement published after the clock."""
    lookahead_date = simulation_clock + pd.Timedelta(days=30)
    return detector.guarded_as_of(value_date, lookahead_date, simulation_clock)


def bug_4_stale_future_cache(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock):
    """Cache-reuse bug: a value fetched at a later point in the backtest loop gets reused at an earlier date."""
    later_clock_used_by_mistake = simulation_clock + pd.Timedelta(days=15)
    return detector.guarded_as_of(value_date, later_clock_used_by_mistake, simulation_clock)


def bug_5_wrong_value_date_future_row(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock):
    """Row-selection bug: ask about tomorrow's value_date while still claiming today's simulation clock."""
    tomorrow_value_date = value_date + pd.Timedelta(days=1)
    return detector.guarded_as_of(tomorrow_value_date, simulation_clock + pd.Timedelta(days=1), simulation_clock)


def bug_6_global_max_knowledge_join(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock):
    """Join bug: ignore the per-row knowledge_date entirely and query as of the last day in the dataset,
    the classic 'joined on entity only' leak that applies the dataset's final vintage to every historical row."""
    return detector.guarded_as_of(value_date, store.max_knowledge_date_overall(), simulation_clock)


def bug_7_wall_clock_today(store: VintageStore, detector: LeakageDetector, value_date, simulation_clock, wall_clock_today):
    """Wall-clock bug: a historical backtest defaults an unset as-of parameter to 'today' instead of the
    simulated rebalance date."""
    return detector.guarded_as_of(value_date, wall_clock_today, simulation_clock)


ALL_LEAKAGE_BUGS = [
    bug_1_use_final_restated_value,
    bug_2_off_by_one_day_ahead,
    bug_3_bfill_across_restatement,
    bug_4_stale_future_cache,
    bug_5_wrong_value_date_future_row,
    bug_6_global_max_knowledge_join,
    bug_7_wall_clock_today,
]

assert len(ALL_LEAKAGE_BUGS) == 7
