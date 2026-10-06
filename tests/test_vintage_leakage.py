import pandas as pd
import pytest

from xsalpha.leakage_bugs import ALL_LEAKAGE_BUGS
from xsalpha.leakage_detector import LeakageDetector, LeakageError
from xsalpha.run_manifest import build_manifest, compute_pinned_statistic
from xsalpha.vintage_store import build_simulated_vintage_store


@pytest.fixture(scope="module")
def small_store():
    trading_days = pd.bdate_range("2020-01-02", periods=120)
    return build_simulated_vintage_store(30, trading_days, seed=55), trading_days


def test_vintage_store_never_returns_a_later_knowledge_date(small_store):
    store, trading_days = small_store
    query_date = trading_days[60]
    for value_date in trading_days[:70]:
        rows = store.as_of(value_date, query_date)
        if not rows.empty:
            assert bool((rows["knowledge_date"] <= query_date).all())


def test_restatements_exist_and_differ_from_preliminary(small_store):
    store, trading_days = small_store
    value_date = trading_days[0]
    frame = store._as_frame()
    rows_for_day = frame[frame["value_date"] == value_date]
    assert len(rows_for_day) >= 30


def test_all_seven_leakage_bugs_are_caught(small_store):
    store, trading_days = small_store
    detector = LeakageDetector(store)
    simulation_clock = trading_days[80]
    value_date = trading_days[75]
    wall_clock_today = pd.Timestamp.today().normalize()

    caught = 0
    for bug_fn in ALL_LEAKAGE_BUGS:
        with pytest.raises(LeakageError):
            if bug_fn.__name__ == "bug_7_wall_clock_today":
                bug_fn(store, detector, value_date, simulation_clock, wall_clock_today)
            else:
                bug_fn(store, detector, value_date, simulation_clock)
        caught += 1
    assert caught == 7


def test_clean_read_is_not_flagged(small_store):
    store, trading_days = small_store
    detector = LeakageDetector(store)
    simulation_clock = trading_days[80]
    value_date = trading_days[75]
    detector.guarded_as_of(value_date, simulation_clock, simulation_clock)


def test_pinned_runs_reproduce_exactly(small_store):
    store, trading_days = small_store
    as_of_date = trading_days[80]
    manifest = build_manifest(store, seed=99)
    run_1 = compute_pinned_statistic(store, manifest, as_of_date)
    run_2 = compute_pinned_statistic(store, manifest, as_of_date)
    assert run_1 == run_2
